import torch
import torch.nn as nn
import torch.nn.functional as F


class DSConv(nn.Module):
    """
    Depthwise Separable Convolution.
    Due to limited computation ressource we use this trick do divide the number of params by 8~.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            groups=in_channels,
            padding_mode="reflect",
            bias=False,
        )

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
    Lightweight version of the ResBlock using dephtwise separable convolution
    """

    def __init__(self, channels):
        super().__init__()
        self.conv1 = DSConv(channels, channels)
        self.conv2 = DSConv(channels, channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        return out + residual


class LightEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DSConv(in_channels, out_channels)

        self.res = LightResBlock(out_channels)

        self.downsample = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            padding_mode="reflect",
            bias=False,
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
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )

        # Conv pour mixer après concaténation (réduction des canaux)
        self.conv = DSConv(in_channels + skip_channels, out_channels)
        self.res = LightResBlock(out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)

        # Gestion safe des tailles (au cas où H/W impair)
        if x.shape != skip.shape:
            x = F.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=False
            )

        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.res(x)
        return x


class LightweightUNet(nn.Module):
    """
    LightweightUnet with residual connections for Macbook Air training
    """

    def __init__(
        self, in_channels=3, out_channels=3, global_residual=True, start_filters=32
    ):
        super().__init__()
        self.global_residual = global_residual

        self.enc1 = LightEncoder(in_channels, start_filters)  # 48
        self.enc2 = LightEncoder(start_filters, start_filters * 2)  # 96
        self.enc3 = LightEncoder(start_filters * 2, start_filters * 4)  # 192
        self.enc4 = LightEncoder(start_filters * 4, start_filters * 8)  # 384

        self.bottleneck = nn.Sequential(
            DSConv(start_filters * 8, start_filters * 16),  # 384 -> 768
            LightResBlock(start_filters * 16),
            DSConv(start_filters * 16, start_filters * 16),  # 768 -> 768
        )

        self.dec4 = LightDecoder(
            start_filters * 16, start_filters * 8, start_filters * 8
        )  # 768 + 384 -> 384
        self.dec3 = LightDecoder(
            start_filters * 8, start_filters * 4, start_filters * 4
        )  # 384 + 192 -> 192
        self.dec2 = LightDecoder(
            start_filters * 4, start_filters * 2, start_filters * 2
        )  # 192 + 96 -> 96
        self.dec1 = LightDecoder(
            start_filters * 2, start_filters, start_filters
        )  # 96 + 48 -> 48

        self.output = nn.Conv2d(
            start_filters,
            out_channels,
            kernel_size=3,
            padding=1,
            padding_mode="reflect",
        )

        if global_residual:
            nn.init.constant_(self.output.weight, 0)
            nn.init.constant_(self.output.bias, 0)

    def forward(self, x):
        input_image = x

        s1, x = self.enc1(x)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)
        s4, x = self.enc4(x)

        x = self.bottleneck(x)

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
    print("Original parameters: ~28,000,000")
    print(f"Lightweight parameters: {count_parameters(model):,}")

    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
