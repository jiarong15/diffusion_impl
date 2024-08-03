from .helper_module import DoubleConv, Down, Up, SelfAttention
import torch.nn as nn
import torch


## Implementing one of the commonly used architectures in diffusion models
class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, device='cuda'):
        ## c_in and c_out denotes the channels of the image
        ## In this case, both take the value 3.
        super().__init__()
        self.device = device
        self.time_dim = time_dim

        ## Self attention arguments:
        ## 1st arg: channel dimension
        ## 2nd arg: current image resolution

        ## Down sample arguments:
        ## 1st arg: input channels
        ## 2nd arg: output channels

        ## Bottle neck layer: Convolutional layers

        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 32)

        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)

        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)


        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 16)
        
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 32)

        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 64)

        self.outc = nn.Conv2d(64, c_out, kernel_size=1)



    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (10000**(torch.arange(0, channels, 2, device=self.device).float() / channels))
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
    

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        ## The upsampling here has skip connections
        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output













