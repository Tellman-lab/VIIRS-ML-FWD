from fastai.vision.all import *


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

# New Up class
class Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
         # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# No initial late fusion convolution, but flexible to resampled input resolution at 128 or 256     
class LateFusionUNetReg(nn.Module):
    def __init__(self, n_i_bands, n_m_bands, n_classes,
                 input_resolution=128, fusion=True):
        super(LateFusionUNetReg, self).__init__()

        self.n_i_bands = n_i_bands
        self.n_m_bands = n_m_bands
        self.n_classes = n_classes
        self.input_resolution = input_resolution

        if input_resolution not in [128, 256]:
            raise ValueError("input_resolution must be 128 or 256")

        if input_resolution == 256 and fusion:
            raise ValueError("Late fusion convolution not compatible with 256 input resolution")

        n_input_channels = n_i_bands + n_m_bands

        # Late fusion convolution to downsample I bands from 256 to 128
        if n_i_bands > 0 and fusion:
            self.firstStep = nn.Conv2d(
                n_i_bands, n_i_bands,
                kernel_size=3,
                padding=1,
                padding_mode='reflect',
                stride=2
            )

        # ---------------------------------------------------------
        # Encoder
        # ---------------------------------------------------------
        self.inc = DoubleConv(n_input_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # For 256x256 input, add one additional downsampling step
        if input_resolution == 256:
            self.down5 = Down(1024, 1024)

        # ---------------------------------------------------------
        # Decoder
        # ---------------------------------------------------------
        if input_resolution == 128:

            self.up1 = Up(1024, 512, 512)
            self.up2 = Up(512, 256, 256)
            self.up3 = Up(256, 128, 128)
            self.up4 = Up(128, 64, 64)

        elif input_resolution == 256:

            self.up1 = Up(1024, 1024, 1024)
            self.up2 = Up(1024, 512, 512)
            self.up3 = Up(512, 256, 256)
            self.up4 = Up(256, 128, 128)
            self.up5 = Up(128, 64, 64)

        self.outc = nn.Sequential(
            OutConv(64, n_classes),
            SigmoidRange(0, 1)
        )

    def forward(self, x):

        # ---------------------------------------------------------
        # Prepare inputs
        # ---------------------------------------------------------

        if self.n_i_bands > 0 and self.n_m_bands > 0:

            # Both inputs should already have the same spatial
            # resolution by the time they reach the model
            x = torch.cat((x[0], x[1]), dim=1)

        elif self.n_i_bands > 0:

            x = x[0]

        elif self.n_m_bands > 0:

            x = x[1]

        else:
            raise ValueError(
                "At least one I or M band must be provided."
            )

        # ---------------------------------------------------------
        # Encoder
        # ---------------------------------------------------------

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # ---------------------------------------------------------
        # Decoder
        # ---------------------------------------------------------

        if self.input_resolution == 128:

            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)

            # 128 -> 256
            x = F.interpolate(
                x,
                scale_factor=2,
                mode='bilinear',
                align_corners=False
            )

        else:

            # Additional encoder level:
            # 256 -> 128 -> 64 -> 32 -> 16 -> 8
            x6 = self.down5(x5)

            x = self.up1(x6, x5)
            x = self.up2(x, x4)
            x = self.up3(x, x3)
            x = self.up4(x, x2)
            x = self.up5(x, x1)

        return self.outc(x)

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)

        if self.input_resolution == 256:
            self.down5 = torch.utils.checkpoint(self.down5)
            self.up1 = torch.utils.checkpoint(self.up1)
            self.up2 = torch.utils.checkpoint(self.up2)
            self.up3 = torch.utils.checkpoint(self.up3)
            self.up4 = torch.utils.checkpoint(self.up4)
            self.up5 = torch.utils.checkpoint(self.up5)

        else:
            self.up1 = torch.utils.checkpoint(self.up1)
            self.up2 = torch.utils.checkpoint(self.up2)
            self.up3 = torch.utils.checkpoint(self.up3)
            self.up4 = torch.utils.checkpoint(self.up4)

        self.outc = torch.utils.checkpoint(self.outc)