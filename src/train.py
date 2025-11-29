import os
import sys
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import get_dataloaders
from src.models.lightunet import LightweightUNet
from src.utils.metrics import PSNRMetric


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=1e-3):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.epsilon * self.epsilon)
        return loss.mean()


def get_loss_function(loss_type):
    if loss_type == "l1":
        return nn.L1Loss()
    elif loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "charbonnier":
        return CharbonnierLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    psnr_metric = PSNRMetric()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]")

    for blur, sharp in pbar:
        blur, sharp = blur.to(device), sharp.to(device)

        output = model(blur)
        loss = criterion(output, sharp)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        psnr_metric.update(output.detach(), sharp)

        pbar.set_postfix(
            {"loss": f"{loss.item():.4f}", "psnr": f"{psnr_metric.compute():.2f}"}
        )

    return running_loss / len(train_loader), psnr_metric.compute()


def validate(model, val_loader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    psnr_metric = PSNRMetric()

    pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1} [Val]  ")

    with torch.no_grad():
        for blur, sharp in pbar:
            blur, sharp = blur.to(device), sharp.to(device)
            output = model(blur)
            loss = criterion(output, sharp)

            running_loss += loss.item()
            psnr_metric.update(output, sharp)

            pbar.set_postfix(
                {"loss": f"{loss.item():.4f}", "psnr": f"{psnr_metric.compute():.2f}"}
            )

    return running_loss / len(val_loader), psnr_metric.compute()


def main():
    parser = argparse.ArgumentParser(description="Train LightweightUNet")

    # Config matching notebook defaults
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./experiments")
    parser.add_argument("--experiment_name", type=str, default="unet_light_v4_123")
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_split", type=float, default=0.15)

    parser.add_argument("--start_filters", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--loss", type=str, default="charbonnier")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--early_stop_patience", type=int, default=20)

    args = parser.parse_args()

    set_seed(args.seed)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Directories
    experiment_dir = os.path.join(args.save_dir, args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # Data
    train_loader, val_loader = get_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
        val_split=args.val_split,
    )

    # Model
    model = LightweightUNet(
        in_channels=3,
        out_channels=3,
        global_residual=True,
        start_filters=args.start_filters,
    ).to(device)

    print(
        f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    # Optimizer & Scheduler
    criterion = get_loss_function(args.loss)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # Training Loop
    best_psnr = 0.0
    epochs_without_improvement = 0

    print("Starting training...")
    for epoch in range(args.epochs):
        # Train
        train_loss, train_psnr = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_psnr = validate(model, val_loader, criterion, device, epoch)

        # Scheduler step
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1}/{args.epochs} | Train PSNR: {train_psnr:.2f} | Val PSNR: {val_psnr:.2f} | LR: {current_lr:.2e}"
        )

        # Save best
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_psnr": best_psnr,
                    "args": vars(args),
                },
                os.path.join(experiment_dir, "best_model.pth"),
            )
            print(f"  ✓ Saved best model (PSNR: {best_psnr:.2f} dB)")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    print(f"Training complete. Best PSNR: {best_psnr:.2f} dB")


if __name__ == "__main__":
    main()
