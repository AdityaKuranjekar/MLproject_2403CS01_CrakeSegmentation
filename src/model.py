"""Standard U-Net (Ronneberger et al., 2015) with BatchNorm, 4 down/up levels."""
import torch
import torch.nn as nn


class DoubleConv(nn.Sequential):
    def __init__(self, c_in, c_out):
        super().__init__(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    """Input (B,3,256,256) -> one logit per pixel (B,1,256,256); sigmoid gives P(crack)."""

    def __init__(self, in_ch=3, base=32):
        super().__init__()
        ch = [base * 2 ** i for i in range(5)]  # 32,64,128,256,512
        self.inc = DoubleConv(in_ch, ch[0])
        self.downs = nn.ModuleList(DoubleConv(ch[i], ch[i + 1]) for i in range(4))
        self.ups = nn.ModuleList(nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2) for i in reversed(range(4)))
        self.dec = nn.ModuleList(DoubleConv(ch[i] * 2, ch[i]) for i in reversed(range(4)))
        self.head = nn.Conv2d(ch[0], 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skips = [self.inc(x)]
        for down in self.downs:
            skips.append(down(self.pool(skips[-1])))
        x = skips.pop()
        for up, dec in zip(self.ups, self.dec):
            x = dec(torch.cat([skips.pop(), up(x)], dim=1))
        return self.head(x)
