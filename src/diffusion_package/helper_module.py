import torch.nn as nn
from torch.nn import functional as F
import torch

class DoubleConv(nn.Module):
    '''
    Used in initial convolution of images as well as 
    at the bottom of the the U shape net.
    '''
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
            ## remains unchanged from the input shape.
            nn.GroupNorm(self.normalizing_group_num, mid_channels),

            ## Apply Gaussian error linear units
            nn.GELU(),

            ## We are transforming the channels number from mid_channels to out_channels
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),

            ## Perform group normalization once again
            ## n×c×w×h -> swh×gn where c = sg
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
            ## We perform a maxpooling using a kernel of size 2x2
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels)
        )

        self.emb_layer = nn.Sequential(

            # Sigmoid linear Unit
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels)
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)

        ## Linear projection to bring the time embedding 
        ## to the proper dimension. emb_dim should be the same
        ## as the time_dim for the time tensors (emb_dim == time_dim).
        ## Out channels is the new no. of channels to be down sampled to
        ## self.emb_layer(t) outputs a new tensor of shape (n, time_dim)
        ## where n would be the number of images trained in the network
        ## [:, :, None, None] expands the dim to (n, time_dim, 1, 1)
        ## Finally, repeat(1, 1, x.shape[-2], x.shape[-1]) keeps n, time_dim
        ## while repeating the date of the time tensor to achieve a final shape
        ## of (n, time_dim, x, y) where x and y denotes the new shape of the
        ## maxpooled image
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        ## We perform bilinear interpolation to increase the 
        ## size of the image data that we parse through this layer
        ## since scale factor is 2, each data point is doubled
        ## across x and y axes since we are dealing with 2D image data vectors
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, mid_channels=in_channels // 2)
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels)
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)

        ## Concatenate the skip connections from the encoder layer
        ## after self-attention and down sampling is applied
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)

        ## Refer to the similar explanation for Down class for this step
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb



class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()

        self.channels = channels
        self.size = size

        ## Continuing from the channels fed in from the
        ## upstream layers, we split into 4 attention heads in this case
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)

        ## For each batch of picture data examples trained
        ## We normalize across the channels, width and height of image
        ## c x w x h over n (n×c×w×h -> cwh×n)
        self.ln = nn.LayerNorm([self.channels])

        ## Feed Forward layer
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )


    def forward(self, x):

        ## We first change it such that IF an epoch of data trained 
        ## follows this dimensions [5, 5, 64, 64] -> 5 images of 5 channels
        ## with an image of width and height 64 x 64. We get a resulting
        ## [80, 5, 256] as the channel number is preserved (5), 256 * 80
        ## must make up 5 * 64 * 64. we restrict the last item to be
        ## self.size ** 2 and the left over divided value to be the first
        ## where 5 * 64 * 64 / (16^2) assuming self.size is 16.
        ## Finally we swap the y (2nd) and z (3rd) axes. This results in
        ## the final output [80, 256, 5].
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1,2)

        ## We normalize over the 256 x 5 dimension
        ## output dimension remains the same at [80, 256, 5] 
        ## for an example of data whose dim is [5, 5, 64, 64].
        x_ln = self.ln(x)

        ## Returns tuple of attention value and attention weights
        ## The attention_value has the same dimension as x_ln.
        ## Since the embedding dimension for the attention layer is
        ## equal to the number of channels, we can visualize that for
        ## head(i) = Attention(QW(q), KW(k), VW(v)), the learnable weights
        ## W(q) / W(k) / W(v) will be the shape of (channels, channels).
        ## We then perform a dot product with all the concatenated heads
        ## and a learnable parameter W(0) follows: 
        ## CONCAT(head(i), ..., head(n)) @ W(0) where n is 4 in this case.
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value

        ## We output in the desired up or down sampled size
        ## in the dimension as follows with the example above: [80, 5, 16, 16]
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)
    




