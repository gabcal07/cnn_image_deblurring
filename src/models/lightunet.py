import torch
import torch.nn as nn
import torch.nn.functional as F

class DSConv(nn.Module):
    """
    Depthwise Separable Convolution.
    Remplace une Conv2D standard pour diviser les paramètres par ~8.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        # 1. Convolution Spatiale (Depthwise): traite chaque canal séparément
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding,
            groups=in_channels, padding_mode='reflect', bias=False
        )
        # 2. Convolution Ponctuelle (Pointwise): mixe les canaux (1x1)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        return self.activation(x)

class LightResBlock(nn.Module):
    """
    Version allégée du ResBlock utilisant des DSConv.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = DSConv(channels, channels)
        self.conv2 = DSConv(channels, channels) # Note: DSConv inclut déjà Norm+Act

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        return out + residual

class LightEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Projection initiale
        self.conv = DSConv(in_channels, out_channels)
        # Bloc résiduel pour bien apprendre les features
        self.res = LightResBlock(out_channels)
        # Downsampling via convolution (stride 2) standard (plus efficace pour réduire la taille)
        # On garde une conv standard ici pour ne pas perdre trop d'info spatiale lors de la réduction
        self.downsample = nn.Conv2d(
            out_channels, out_channels, kernel_size=4, stride=2, padding=1,
            padding_mode='reflect', bias=False
        )

    def forward(self, x):
        x = self.conv(x)
        features = self.res(x)
        down = self.downsample(features)
        return features, down

class LightDecoder(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # Upsample sans paramètres (Bilinear) au lieu de ConvTranspose
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # Conv pour mixer après concaténation (réduction des canaux)
        self.conv = DSConv(in_channels + skip_channels, out_channels)
        self.res = LightResBlock(out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)
        
        # Gestion safe des tailles (au cas où H/W impair)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.res(x)
        return x

class LightweightUNet(nn.Module):
    """
    Optimized U-Net for Colab Free Tier.
    Target: 3 Million parameters.
    """
    def __init__(self, in_channels=3, out_channels=3, global_residual=True, start_filters=32):
        super().__init__()
        self.global_residual = global_residual
        
        # Encoder
        self.enc1 = LightEncoder(in_channels, start_filters)       # 32
        self.enc2 = LightEncoder(start_filters, start_filters*2)   # 64
        self.enc3 = LightEncoder(start_filters*2, start_filters*4) # 128
        self.enc4 = LightEncoder(start_filters*4, start_filters*8) # 256
        
        # Bottleneck (Le fond du U)
        self.bottleneck = nn.Sequential(
            DSConv(start_filters*8, start_filters*16), # 256 -> 512
            LightResBlock(start_filters*16),
            DSConv(start_filters*16, start_filters*16) # 512 -> 512
        )
        
        # Decoder
        self.dec4 = LightDecoder(start_filters*16, start_filters*8, start_filters*8) # 512 + 256 -> 256
        self.dec3 = LightDecoder(start_filters*8, start_filters*4, start_filters*4)  # 256 + 128 -> 128
        self.dec2 = LightDecoder(start_filters*4, start_filters*2, start_filters*2)  # 128 + 64 -> 64
        self.dec1 = LightDecoder(start_filters*2, start_filters, start_filters)      # 64 + 32 -> 32
        
        # Output
        self.output = nn.Conv2d(start_filters, out_channels, kernel_size=3, padding=1, padding_mode='reflect')
        
        if global_residual:
            nn.init.constant_(self.output.weight, 0)
            nn.init.constant_(self.output.bias, 0)

    def forward(self, x):
        input_image = x
        
        # Encodage
        s1, x = self.enc1(x)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)
        s4, x = self.enc4(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Décodage
        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        
        x = self.output(x)
        
        if self.global_residual:
            x = x + input_image
            
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    model = LightweightUNet()
    print(f"Original parameters: ~28,000,000")
    print(f"Lightweight parameters: {count_parameters(model):,}")
    
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")