import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


# noinspection PyTypeChecker
class DoubleConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv2d, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv1(x)


# noinspection PyTypeChecker
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        # Contraction Layer
        self.downsampling = nn.ModuleList()  # Double Conv List
        self.maxpool = nn.MaxPool2d(2, 2, padding=1)
        self.downsampling.append(DoubleConv2d(in_channels, 64))
        self.downsampling.append(DoubleConv2d(64, 128))
        self.downsampling.append(DoubleConv2d(128, 256))
        self.downsampling.append(DoubleConv2d(256, 512))
        self.downsample_final = DoubleConv2d(512, 1024)

        # Expansion Layer
        self.upsampling1 = nn.ModuleList()  # ConvTranspose2d
        self.upsampling2 = nn.ModuleList()  # Double Conv
        self.upsampling1.append(nn.ConvTranspose2d(1024, 512, 2, 2))  # 512 * 2 because of skip connection
        self.upsampling1.append(nn.ConvTranspose2d(512, 256, 2, 2))
        self.upsampling1.append(nn.ConvTranspose2d(256, 128, 2, 2))
        self.upsampling1.append(nn.ConvTranspose2d(128, 64, 2, 2))
        self.upsampling2.append(DoubleConv2d(1024, 512))
        self.upsampling2.append(DoubleConv2d(512, 256))
        self.upsampling2.append(DoubleConv2d(256, 128))
        self.upsampling2.append(DoubleConv2d(128, 64))
        # Final 1x1 Convolution
        self.conv_final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        # Holder for skip connections
        skips = []
        # Contraction Layer
        for layer in self.downsampling:
            x = layer(x)
            skips.append(x)
            x = self.maxpool(x)
        x = self.downsample_final(x)

        # Expansion Layer
        skips = skips[::-1]
        x = self.upsampling1[0](x)
        # This is kind of confusing; need to remove first layer from module list
        for i, layer in enumerate(self.upsampling1[1::]):
            # Check dimensions for torch.cat - correct as needed
            if x.shape != skips[i].shape:
                x = TF.resize(x, skips[i].shape[2:])
            x = self.upsampling2[i](torch.cat((skips[i], x), dim=1))
            x = layer(x)
        if x.shape != skips[-1].shape:
            x = TF.resize(x, skips[-1].shape[2:])
        x = self.upsampling2[-1](torch.cat((skips[-1], x), dim=1))
        # Final 1x1 conv
        return self.conv_final(x)
