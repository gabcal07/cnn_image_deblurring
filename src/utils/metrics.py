"""
Metrics for evaluating image deblurring quality.

The primary metric is PSNR (Peak Signal-to-Noise Ratio), which measures
the quality of the reconstructed image compared to the ground truth.
"""

import torch
import torch.nn.functional as F


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor, max_value: float = 1.0) -> torch.Tensor:
    """
    Calculate PSNR between two images.
    
    PSNR = 10 * log10(MAX^2 / MSE)
    
    Higher PSNR = better quality (less error).
    Typical ranges:
    - 20-25 dB: Low quality
    - 25-30 dB: Medium quality
    - 30-35 dB: Good quality
    - 35+ dB: Excellent quality
    
    Args:
        img1: First image tensor, shape (B, C, H, W) or (C, H, W)
        img2: Second image tensor, same shape as img1
        max_value: Maximum possible pixel value (1.0 for normalized images)
    
    Returns:
        PSNR value in dB. If batched, returns mean PSNR across batch.
    
    Example:
        >>> pred = model(blur_img)  # (B, 3, 256, 256)
        >>> target = sharp_img      # (B, 3, 256, 256)
        >>> psnr = calculate_psnr(pred, target)
        >>> print(f"PSNR: {psnr:.2f} dB")
    """
    # Ensure same shape
    assert img1.shape == img2.shape, f"Shape mismatch: {img1.shape} vs {img2.shape}"
    
    # Calculate MSE per image if batched
    if img1.dim() == 4:
        mse = torch.mean((img1 - img2) ** 2, dim=[1, 2, 3])
        # Handle perfect matches (mse == 0)
        mse[mse == 0] = 1e-10
        psnr = 10 * torch.log10(max_value**2 / mse)
        return psnr.mean()
    else:
        # Single image
        mse = F.mse_loss(img1, img2, reduction='mean')
        if mse == 0:
            return torch.tensor(100.0) # Cap at 100dB for perfect match
        return 10 * torch.log10(max_value**2 / mse)
    
    return psnr


def calculate_psnr_batch(img1: torch.Tensor, img2: torch.Tensor, max_value: float = 1.0) -> torch.Tensor:
    """
    Calculate PSNR for each image in a batch separately.
    
    Args:
        img1: First image batch, shape (B, C, H, W)
        img2: Second image batch, shape (B, C, H, W)
        max_value: Maximum possible pixel value
    
    Returns:
        Tensor of PSNR values, shape (B,) - one PSNR per image
    
    Example:
        >>> pred = model(blur_batch)  # (4, 3, 256, 256)
        >>> target = sharp_batch       # (4, 3, 256, 256)
        >>> psnr_values = calculate_psnr_batch(pred, target)
        >>> print(f"PSNR per image: {psnr_values}")  # [28.3, 31.2, 29.8, 30.5]
        >>> print(f"Mean PSNR: {psnr_values.mean():.2f} dB")
    """
    assert img1.shape == img2.shape, f"Shape mismatch: {img1.shape} vs {img2.shape}"
    assert img1.ndim == 4, "Expected batched input (B, C, H, W)"
    
    batch_size = img1.shape[0]
    psnr_values = []
    
    for i in range(batch_size):
        mse = F.mse_loss(img1[i], img2[i], reduction='mean')
        if mse == 0:
            psnr = torch.tensor(float('inf'))
        else:
            psnr = 10 * torch.log10(max_value**2 / mse)
        psnr_values.append(psnr)
    
    return torch.stack(psnr_values)


class PSNRMetric:
    """
    PSNR metric tracker for training/validation.
    
    Accumulates PSNR values across batches and computes average.
    
    Example:
        >>> psnr_tracker = PSNRMetric()
        >>> for batch in dataloader:
        >>>     pred = model(batch)
        >>>     psnr_tracker.update(pred, target)
        >>> print(f"Average PSNR: {psnr_tracker.compute():.2f} dB")
        >>> psnr_tracker.reset()
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset accumulated values."""
        self.total_psnr = 0.0
        self.count = 0
    
    def update(self, pred: torch.Tensor, target: torch.Tensor, max_value: float = 1.0):
        """
        Update metric with a new batch.
        
        Args:
            pred: Predicted images (B, C, H, W)
            target: Ground truth images (B, C, H, W)
            max_value: Maximum pixel value
        """
        with torch.no_grad():
            # Calculate PSNR for each image separately
            batch_psnrs = calculate_psnr_batch(pred, target, max_value)
            self.total_psnr += batch_psnrs.sum().item()
            self.count += pred.shape[0]  # Count actual number of images
    
    def compute(self) -> float:
        """
        Compute average PSNR across all updates.
        
        Returns:
            Average PSNR in dB
        """
        if self.count == 0:
            return 0.0
        return self.total_psnr / self.count
