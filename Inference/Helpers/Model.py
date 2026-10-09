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

class ChannelConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# No initial late fusion convolution, but flexible to resampled input resolution at 128 or 256     
class LateFusionUNetReg(nn.Module):
    def __init__(self, n_i_bands, n_m_bands, n_classes,
                 input_resolution=128, learned_downsampling=True, input_skip=False):
        super(LateFusionUNetReg, self).__init__()

        self.n_i_bands = n_i_bands
        self.n_m_bands = n_m_bands
        self.n_classes = n_classes
        self.input_resolution = input_resolution
        self.learned_downsampling = learned_downsampling
        self.input_skip = input_skip

        # Verify settings
        if input_resolution not in [128, 256]:
            raise ValueError("input_resolution must be 128 or 256")

        if learned_downsampling and input_resolution != 128:
            raise ValueError("learned_downsampling=True requires input_resolution=128")

        if learned_downsampling and n_i_bands == 0:
            raise ValueError("learned_downsampling=True requires at least one I band")

        if input_skip and input_resolution == 128 and not learned_downsampling:
            raise ValueError("input_skip=True with input_resolution=128 requires "
                             "learned_downsampling=True to preserve the 256x256 I-band input")

        n_input_channels = n_i_bands + n_m_bands

        # Late fusion convolution to downsample I bands from 256 to 128
        if n_i_bands > 0 and learned_downsampling:
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
            self.up5 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)

        elif input_resolution == 256:

            self.up1 = Up(1024, 1024, 1024)
            self.up2 = Up(1024, 512, 512)
            self.up3 = Up(512, 256, 256)
            self.up4 = Up(256, 128, 128)
            self.up5 = Up(128, 64, 64)

        # Separate full-resolution I-band skip connection
        if n_i_bands > 0 and input_skip:
            self.learned_channels_ds = ChannelConv(n_i_bands, 1)

        # Final prediction combines decoder output and x0 and skip_input True
        decoder_channels = 32 if input_resolution == 128 else 64
        out_channels = decoder_channels + (1 if input_skip and n_i_bands > 0 else 0)
        self.outc = nn.Sequential(OutConv(out_channels, n_classes), SigmoidRange(0, 1))

    def forward(self, x):

        # ---------------------------------------------------------
        # Prepare inputs
        # ---------------------------------------------------------

        x0 = None

        if self.n_i_bands > 0 and self.input_skip:
            # Full-resolution skip from the original I bands
            x0 = self.learned_channels_ds(x[0])

        if self.n_i_bands > 0:
            i = x[0]
            if self.learned_downsampling:
                i = self.firstStep(i)

        if self.n_m_bands > 0:
            m = x[1]

        if self.n_i_bands > 0 and self.n_m_bands > 0:
            x = torch.cat((i, m), dim=1)
        elif self.n_i_bands > 0:
            x = i
        else:
            x = m

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
            x = self.up5(x)

        else:

            # Additional encoder level:
            # 256 -> 128 -> 64 -> 32 -> 16 -> 8
            x6 = self.down5(x5)

            x = self.up1(x6, x5)
            x = self.up2(x, x4)
            x = self.up3(x, x3)
            x = self.up4(x, x2)
            x = self.up5(x, x1)

        # Combine decoder features with original I-band skip
        if x0 is not None:
            x = torch.cat((x, x0), dim=1)

        return self.outc(x)

    # def use_checkpointing(self):
    #     self.inc = torch.utils.checkpoint(self.inc)
    #     self.down1 = torch.utils.checkpoint(self.down1)
    #     self.down2 = torch.utils.checkpoint(self.down2)
    #     self.down3 = torch.utils.checkpoint(self.down3)
    #     self.down4 = torch.utils.checkpoint(self.down4)

    #     if self.input_resolution == 256:
    #         self.down5 = torch.utils.checkpoint(self.down5)
    #         self.up1 = torch.utils.checkpoint(self.up1)
    #         self.up2 = torch.utils.checkpoint(self.up2)
    #         self.up3 = torch.utils.checkpoint(self.up3)
    #         self.up4 = torch.utils.checkpoint(self.up4)
    #         self.up5 = torch.utils.checkpoint(self.up5)

    #     else:
    #         self.up1 = torch.utils.checkpoint(self.up1)
    #         self.up2 = torch.utils.checkpoint(self.up2)
    #         self.up3 = torch.utils.checkpoint(self.up3)
    #         self.up4 = torch.utils.checkpoint(self.up4)

    #     self.outc = torch.utils.checkpoint(self.outc)