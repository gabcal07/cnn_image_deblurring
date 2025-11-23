"""
Simple U-Net for Image Deblurring.

Architecture:
- Encoder: 4 levels with strided convolutions (no max pooling)
- Bottleneck: Deepest feature extraction
- Decoder: 4 levels with transpose convolutions
- Skip connections: Concatenate encoder features to decoder

Design choices for deblurring:
- Stride-2 Conv instead of MaxPool (preserves spatial info)
- Instance Normalization (preserves per-image contrast)
- LeakyReLU (prevents dead neurons)
- Reflection padding (avoids border artifacts)
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Basic convolutional block: Conv → InstanceNorm → LeakyReLU.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        stride: Convolution stride
        padding: Padding size (default: kernel_size // 2 for same size)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1
    ):
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode='reflect',  # Better than zero padding for images
            bias=False  # Not needed when using normalization
        )
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))


class ResBlock(nn.Module):
    """
    Residual Block: x + Conv(Norm(Act(Conv(Norm(Act(x))))))
    Helps with gradient flow and learning identity functions.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, padding_mode='reflect', bias=False)
        self.norm1 = nn.InstanceNorm2d(channels, affine=True)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, padding_mode='reflect', bias=False)
        self.norm2 = nn.InstanceNorm2d(channels, affine=True)
        
    def forward(self, x):
        residual = x
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return out + residual


class EncoderBlock(nn.Module):
    """
    Encoder block: Conv + ResBlock + Downsampling.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        # Initial projection
        self.conv = ConvBlock(in_channels, out_channels)
        
        # Residual refinement
        self.res = ResBlock(out_channels)
        
        # Downsampling
        self.downsample = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            padding_mode='reflect',
            bias=False
        )
    
    def forward(self, x):
        # Extract features
        x = self.conv(x)
        features = self.res(x)  # Save for skip connection
        
        # Downsample
        downsampled = self.downsample(features)
        
        return features, downsampled


class DecoderBlock(nn.Module):
    """
    Decoder block: Upsample + Concatenate + Conv + ResBlock.
    """
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        
        # Upsampling
        self.upsample = nn.ConvTranspose2d(
            in_channels,
            in_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )
        
        # Reduce channels after concatenation
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)
        
        # Residual refinement
        self.res = ResBlock(out_channels)
    
    def forward(self, x, skip):
        # Upsample
        x = self.upsample(x)
        
        # Concatenate
        x = torch.cat([x, skip], dim=1)
        
        # Refine
        x = self.conv(x)
        x = self.res(x)
        
        return x


class SimpleUNet(nn.Module):
    """
    Simple U-Net for image deblurring.
    
    Architecture:
        Input (3, 256, 256)
        ├─ Encoder1: (3, 256, 256) → (64, 256, 256) → downsample → (64, 128, 128)
        ├─ Encoder2: (64, 128, 128) → (128, 128, 128) → downsample → (128, 64, 64)
        ├─ Encoder3: (128, 64, 64) → (256, 64, 64) → downsample → (256, 32, 32)
        ├─ Encoder4: (256, 32, 32) → (512, 32, 32) → downsample → (512, 16, 16)
        ├─ Bottleneck: (512, 16, 16) → (512, 16, 16)
        ├─ Decoder4: (512, 16, 16) + skip(512) → upsample → (256, 32, 32)
        ├─ Decoder3: (256, 32, 32) + skip(256) → upsample → (128, 64, 64)
        ├─ Decoder2: (128, 64, 64) + skip(128) → upsample → (64, 128, 128)
        ├─ Decoder1: (64, 128, 128) + skip(64) → upsample → (64, 256, 256)
        └─ Output: (64, 256, 256) → (3, 256, 256)
    
    Args:
        in_channels: Number of input channels (default: 3 for RGB)
        out_channels: Number of output channels (default: 3 for RGB)
        global_residual: If True, learns the residual (output = input + model(input)). Default: True.
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 3, global_residual: bool = True):
        super().__init__()
        
        self.global_residual = global_residual
        
        # Encoder (downsampling path)
        self.encoder1 = EncoderBlock(in_channels, 64)    # 256 → 128
        self.encoder2 = EncoderBlock(64, 128)            # 128 → 64
        self.encoder3 = EncoderBlock(128, 256)           # 64 → 32
        self.encoder4 = EncoderBlock(256, 512)           # 32 → 16
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock(512, 512),
            ConvBlock(512, 512)
        )
        
        # Decoder (upsampling path)
        self.decoder4 = DecoderBlock(512, 512, 256)      # 16 → 32
        self.decoder3 = DecoderBlock(256, 256, 128)      # 32 → 64
        self.decoder2 = DecoderBlock(128, 128, 64)       # 64 → 128
        self.decoder1 = DecoderBlock(64, 64, 64)         # 128 → 256
        
        # Output layer (1x1 conv to produce RGB image)
        self.output = nn.Conv2d(64, out_channels, kernel_size=1)

        # Initialize output layer to zero for residual learning
        # This ensures the initial output is 0, so the network starts as an identity mapping
        if global_residual:
            nn.init.constant_(self.output.weight, 0)
            if self.output.bias is not None:
                nn.init.constant_(self.output.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: Input blurred image (B, 3, H, W)
        
        Returns:
            Deblurred image (B, 3, H, W)
        """
        # Save input for global residual
        input_image = x if self.global_residual else None
        
        # Encoder
        skip1, x = self.encoder1(x)   # (64, 256, 256), (64, 128, 128)
        skip2, x = self.encoder2(x)   # (128, 128, 128), (128, 64, 64)
        skip3, x = self.encoder3(x)   # (256, 64, 64), (256, 32, 32)
        skip4, x = self.encoder4(x)   # (512, 32, 32), (512, 16, 16)
        
        # Bottleneck
        x = self.bottleneck(x)        # (512, 16, 16)
        
        # Decoder (with skip connections)
        x = self.decoder4(x, skip4)   # (256, 32, 32)
        x = self.decoder3(x, skip3)   # (128, 64, 64)
        x = self.decoder2(x, skip2)   # (64, 128, 128)
        x = self.decoder1(x, skip1)   # (64, 256, 256)
        
        # Output
        x = self.output(x)            # (3, 256, 256)
        
        if self.global_residual:
            x = x + input_image
        
        return x


def count_parameters(model):
    """Count trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    print("Testing SimpleUNet...")
    
    model = SimpleUNet(in_channels=3, out_channels=3)
    print(f"\nTotal parameters: {count_parameters(model):,}")
    
    # Test forward pass
    x = torch.randn(2, 3, 256, 256)  # Batch of 2 images
    print(f"\nInput shape: {x.shape}")
    
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    print("\n✓ Model architecture is correct!")
