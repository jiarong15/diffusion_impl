import torch.nn as nn
from torch.nn import functional as F
import torch

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):

        super().__init__()
        self.residual = residual
        self.normalizing_group_num = 1
        if not mid_channels:
            mid_channels = out_channels
        
        ## The following layers are applied sequentially in the
        ## defined order as follows.
        self.double_conv = nn.Sequential(

            ## Weights of conv are learned through backpropagation
            ## We are transforming the channels number from in_channels to mid_channels
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),

            ## We normalize the values group wise by separating the channels
            ## the group number. In this case it is 1. Output's shape
            ## remains unchanged.
            nn.GroupNorm(self.normalizing_group_num, mid_channels),

            ## Apply Gaussian error linear units
            nn.GELU(),

            ## We are transforming the channels number from mid_channels to out_channels
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),

            ## Perform group normalization once again
            nn.GroupNorm(self.normalizing_group_num, out_channels)
        )

    def forward(self, x):
        ## We apply gelu when we have skip connections to the UNet decoder
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        return self.double_conv(x) 
        




class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels)
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels)
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)

        ## Linear projection to bring the time embedding 
        ## to the proper dimension 
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        ## We perform bilinear interpolation to increase the 
        ## size of the image data that we parse through this layer
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2)
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels)
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb



class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()

        self.channels = channels
        self.size = size

        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([self.channels])

        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )


    def forward(self, x):
        x.view(-1, self.channels, self.size * self.size).swapaxes(1,2)
        x_ln = self.ln(x)

        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value

        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)
    




