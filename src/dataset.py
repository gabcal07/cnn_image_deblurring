"""
PyTorch Dataset for GoPro Image Deblurring.

This module provides:
- GoProDataset: Custom Dataset class with on-the-fly augmentation
- get_dataloaders: Helper function to create train/val/test dataloaders
"""

import os
import random
from typing import List, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF


class GoProDataset(Dataset):
    """
    GoPro Image Deblurring Dataset.
    
    Features:
    - On-the-fly random cropping (256x256 patches from 1280x720 images)
    - Random horizontal flip augmentation
    - Normalization to [0, 1] range
    - Optional full-resolution mode (no cropping)
    
    Args:
        blur_paths: List of paths to blurred images
        sharp_paths: List of paths to sharp (ground truth) images
        patch_size: Size of random crop (default: 256). Ignored if full_image=True
        is_train: If True, apply augmentation; if False, use center crop
        full_image: If True, use full resolution (no cropping). Default: False
    """
    
    def __init__(
        self,
        blur_paths: List[str],
        sharp_paths: List[str],
        patch_size: int = 256,
        is_train: bool = True,
        full_image: bool = False
    ):
        assert len(blur_paths) == len(sharp_paths), "Mismatch between blur and sharp image counts"
        
        self.blur_paths = blur_paths
        self.sharp_paths = sharp_paths
        self.patch_size = patch_size
        self.is_train = is_train
        self.full_image = full_image
        
    def __len__(self) -> int:
        return len(self.blur_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and process a single image pair.
        
        Returns:
            blur: Blurred image tensor (C, H, W) in range [0, 1]
            sharp: Sharp image tensor (C, H, W) in range [0, 1]
        """
        # Load images as PIL
        blur_img = Image.open(self.blur_paths[idx]).convert('RGB')
        sharp_img = Image.open(self.sharp_paths[idx]).convert('RGB')
        
        # Apply cropping only if not using full images
        if not self.full_image:
            # Apply cropping (random for train, center for val/test)
            if self.is_train:
                # Random crop - same location for both images
                i, j, h, w = self._get_random_crop_params(blur_img)
            else:
                # Center crop for validation/test
                i, j, h, w = self._get_center_crop_params(blur_img)
            
            blur_img = TF.crop(blur_img, i, j, h, w)
            sharp_img = TF.crop(sharp_img, i, j, h, w)
        
        # Random horizontal flip (only during training)
        if self.is_train and random.random() > 0.5:
            blur_img = TF.hflip(blur_img)
            sharp_img = TF.hflip(sharp_img)
        
        # Random vertical flip (only during training)
        if self.is_train and random.random() > 0.5:
            blur_img = TF.vflip(blur_img)
            sharp_img = TF.vflip(sharp_img)
        
        # Random Rotation (0, 90, 180, 270)
        # On choisit aléatoirement un nombre de quarts de tour (0 à 3)
        k_rot = random.randint(0, 3)
        if k_rot > 0:
            # 90 * k_rot
            angle = k_rot * 90
            blur_img = TF.rotate(blur_img, angle)
            sharp_img = TF.rotate(sharp_img, angle)
        
        # Convert to tensor and normalize to [0, 1]
        blur_tensor = TF.to_tensor(blur_img)  # Converts uint8 [0, 255] to float [0, 1]
        sharp_tensor = TF.to_tensor(sharp_img)
        
        return blur_tensor, sharp_tensor
    
    def _get_random_crop_params(self, img: Image.Image) -> Tuple[int, int, int, int]:
        """
        Get random crop parameters.
        
        Returns:
            (top, left, height, width) for cropping
        """
        width, height = img.size
        
        # Ensure the image is large enough
        if width < self.patch_size or height < self.patch_size:
            raise ValueError(
                f"Image size ({width}x{height}) is smaller than patch size ({self.patch_size})"
            )
        
        # Random top-left corner
        top = random.randint(0, height - self.patch_size)
        left = random.randint(0, width - self.patch_size)
        
        return top, left, self.patch_size, self.patch_size
    
    def _get_center_crop_params(self, img: Image.Image) -> Tuple[int, int, int, int]:
        """
        Get center crop parameters.
        
        Returns:
            (top, left, height, width) for cropping
        """
        width, height = img.size
        
        top = (height - self.patch_size) // 2
        left = (width - self.patch_size) // 2
        
        return top, left, self.patch_size, self.patch_size


def get_image_pairs(root_dir: str, split: str = 'train') -> Tuple[List[str], List[str]]:
    """
    Recursively find blur-sharp image pairs in the dataset.
    
    Args:
        root_dir: Root directory of the dataset (e.g., './data')
        split: 'train' or 'test'
    
    Returns:
        blur_paths: List of paths to blurred images
        sharp_paths: List of paths to sharp images
    """
    blur_paths = []
    sharp_paths = []
    
    split_dir = os.path.join(root_dir, split)
    
    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"Directory {split_dir} does not exist")
    
    # Walk through sequence folders (each represents frames from a GoPro video)
    for sequence_folder in sorted(os.listdir(split_dir)):
        sequence_path = os.path.join(split_dir, sequence_folder)
        if not os.path.isdir(sequence_path):
            continue
        
        blur_dir = os.path.join(sequence_path, 'blur')
        sharp_dir = os.path.join(sequence_path, 'sharp')
        
        if os.path.exists(blur_dir) and os.path.exists(sharp_dir):
            images = sorted(os.listdir(blur_dir))
            for img_name in images:
                if img_name.endswith('.png'):
                    blur_paths.append(os.path.join(blur_dir, img_name))
                    sharp_paths.append(os.path.join(sharp_dir, img_name))
    
    return blur_paths, sharp_paths


def get_dataloaders(
    data_root: str = './data',
    batch_size: int = 16,
    patch_size: int = 256,
    num_workers: int = 0,
    pin_memory: bool = True,
    full_image: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        data_root: Root directory of the dataset
        batch_size: Batch size for training
        patch_size: Size of image patches (256x256). Ignored if full_image=True
        num_workers: Number of parallel data loading workers
        val_split: Fraction of training data to use for validation (default: 0.1)
        pin_memory: Whether to pin memory for faster GPU transfer
        full_image: If True, use full resolution images (1280x720) instead of patches
    
    Returns:
        train_loader: DataLoader for training (from train/ folder)
        val_loader: DataLoader for validation (from test/ folder)
    """
    # Load all image paths
    train_blur, train_sharp = get_image_pairs(data_root, 'train')
    test_blur, test_sharp = get_image_pairs(data_root, 'test')
    
    # Use train/ for training and test/ for validation
    # (Competition evaluation will be done separately with evaluate.py)
    train_blur_split = train_blur
    train_sharp_split = train_sharp
    val_blur_split = test_blur
    val_sharp_split = test_sharp
    
    # Create datasets
    train_dataset = GoProDataset(
        train_blur_split, train_sharp_split,
        patch_size=patch_size,
        is_train=True,
        full_image=full_image
    )
    
    val_dataset = GoProDataset(
        val_blur_split, val_sharp_split,
        patch_size=patch_size,
        is_train=False,  # No augmentation for validation
        full_image=full_image
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    img_size = "Full (1280x720)" if full_image else f"Patches ({patch_size}x{patch_size})"
    print(f"\n=== {img_size} ===")
    print("Dataset sizes:")
    print(f"  Train: {len(train_dataset)} images (train/ folder)")
    print(f"  Val: {len(val_dataset)} images (test/ folder)")
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test the dataset
    print("Testing GoProDataset...")
    
    # Load a small subset for testing
    train_blur, train_sharp = get_image_pairs('./data', 'train')
    print(f"Found {len(train_blur)} training pairs")
    
    # Test 1: Patch mode
    print("\n--- Test 1: Patch Mode (256x256) ---")
    dataset_patch = GoProDataset(train_blur[:10], train_sharp[:10], patch_size=256, is_train=True, full_image=False)
    blur, sharp = dataset_patch[0]
    print(f"  Blur: {blur.shape}")
    print(f"  Sharp: {sharp.shape}")
    print(f"  Value range: [{blur.min():.3f}, {blur.max():.3f}]")
    
    # Test 2: Full image mode
    print("\n--- Test 2: Full Image Mode (1280x720) ---")
    dataset_full = GoProDataset(train_blur[:10], train_sharp[:10], is_train=False, full_image=True)
    blur, sharp = dataset_full[0]
    print(f"  Blur: {blur.shape}")
    print(f"  Sharp: {sharp.shape}")
    print(f"  Value range: [{blur.min():.3f}, {blur.max():.3f}]")
