import argparse
import os
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import GoProDataset, get_image_pairs
from src.models.unet import SimpleUNet
from src.utils.metrics import PSNRMetric

def parse_args():
    parser = argparse.ArgumentParser(description="Train Image Deblurring Model")
    
    # Data
    parser.add_argument('--data_root', type=str, default='./data', help='Path to dataset')
    parser.add_argument('--patch_size', type=int, default=256, help='Training patch size')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    
    # Model
    parser.add_argument('--in_channels', type=int, default=3, help='Input channels')
    parser.add_argument('--out_channels', type=int, default=3, help='Output channels')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--loss', type=str, default='charbonnier', choices=['l1', 'mse', 'charbonnier'], help='Loss function')
    
    # Scheduler
    parser.add_argument('--patience', type=int, default=10, help='Scheduler patience')
    parser.add_argument('--factor', type=float, default=0.5, help='Scheduler factor')
    
    # Checkpointing
    parser.add_argument('--save_dir', type=str, default='./experiments', help='Directory to save results')
    parser.add_argument('--exp_name', type=str, default='default', help='Experiment name')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    
    # Device
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'mps', 'cpu'], help='Device to use')
    
    return parser.parse_args()

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (smooth L1 loss)"""
    def __init__(self, epsilon=1e-3):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.epsilon * self.epsilon)
        return loss.mean()

def get_loss_function(loss_type):
    if loss_type == 'l1':
        return nn.L1Loss()
    elif loss_type == 'mse':
        return nn.MSELoss()
    elif loss_type == 'charbonnier':
        return CharbonnierLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    psnr_metric = PSNRMetric()
    
    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    
    for blur, sharp in pbar:
        blur, sharp = blur.to(device), sharp.to(device)
        
        optimizer.zero_grad()
        output = model(blur)
        loss = criterion(output, sharp)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item()
        psnr_metric.update(output.detach(), sharp)
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    return running_loss / len(loader), psnr_metric.compute()

def validate(model, loader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    psnr_metric = PSNRMetric()
    
    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Val]  ", leave=False)
    
    with torch.no_grad():
        for blur, sharp in pbar:
            blur, sharp = blur.to(device), sharp.to(device)
            
            output = model(blur)
            loss = criterion(output, sharp)
            
            running_loss += loss.item()
            psnr_metric.update(output, sharp)
            
            pbar.set_postfix({'psnr': f"{psnr_metric.compute():.2f}"})
            
    return running_loss / len(loader), psnr_metric.compute()

def main():
    args = parse_args()
    set_seed()
    
    # Setup device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Setup directories
    exp_dir = os.path.join(args.save_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Data
    print("Loading data...")
    train_blur, train_sharp = get_image_pairs(args.data_root, 'train')
    test_blur, test_sharp = get_image_pairs(args.data_root, 'test')
    
    train_dataset = GoProDataset(train_blur, train_sharp, patch_size=args.patch_size, is_train=True)
    val_dataset = GoProDataset(test_blur, test_sharp, patch_size=args.patch_size, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    print(f"Train images: {len(train_dataset)}")
    print(f"Val images: {len(val_dataset)}")
    
    # Model
    model = SimpleUNet(in_channels=args.in_channels, out_channels=args.out_channels).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Optimizer & Loss
    criterion = get_loss_function(args.loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.factor, patience=args.patience, verbose=True)
    
    # Resume
    start_epoch = 0
    best_psnr = 0.0
    
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Resuming from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_psnr = checkpoint['best_psnr']
        else:
            print(f"Checkpoint not found: {args.resume}")
    
    # Training Loop
    print("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        train_loss, train_psnr = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_psnr = validate(model, val_loader, criterion, device, epoch)
        
        scheduler.step(val_psnr)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} PSNR: {train_psnr:.2f} | Val Loss: {val_loss:.4f} PSNR: {val_psnr:.2f} | LR: {current_lr:.2e}")
        
        # Save best
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_psnr': best_psnr,
                'args': vars(args)
            }, os.path.join(exp_dir, 'best_model.pth'))
            print(f"✓ Saved best model (PSNR: {best_psnr:.2f} dB)")
            
        # Save latest
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_psnr': best_psnr,
            'args': vars(args)
        }, os.path.join(exp_dir, 'latest_checkpoint.pth'))

if __name__ == '__main__':
    main()
