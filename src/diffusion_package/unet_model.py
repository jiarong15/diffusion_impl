from .helper_module import DoubleConv, Down, Up, SelfAttention
import torch.nn as nn
import torch

## Implementing one of the commonly used architectures in diffusion models
class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, num_classes=None, time_dim=256, device='cuda'):
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

        if num_classes is not None:
            ## We try to condition on the classes that we may know of 
            ## to help improve our model as part of Classifier Free Guidance
            ## The number of embeddings will be the image classes and
            ## it has to be the same dimension as our time tensors
            ## as we are conditioning the noise learnt on both to improve
            ## training accuracy.
            self.label_embedding = nn.Embedding(num_classes, time_dim)


    def pos_encoding(self, t, channels):
        '''
        Usage of the positional embedding formula
        '''
        even_inv_freq = 1.0 / (10000**(torch.arange(0, channels, 2, device=self.device).float() / channels))
        odd_inv_freq = 1.0 / (10000**(torch.arange(1, channels, 2, device=self.device).float() / channels))

        ## We encode the odd and even indices in their respective 
        ## manner following the positional encoding methodology. The encoding is
        ## as from [[A], [B], [C]] -> [[(channels // 2) no. of As], 
        ##                             [(channels // 2) no. of Bs],
        ##                             [(channels // 2) no. of Cs]]
        pos_even_enc_a = torch.sin(t.repeat(1, channels // 2) * even_inv_freq)
        pos_odd_enc_b = torch.cos(t.repeat(1, channels // 2) * odd_inv_freq)

        ## Stack the even and odd tensors along y axis
        pos_enc = torch.cat([pos_even_enc_a, pos_odd_enc_b], dim=-1)
        return pos_enc


    def forward(self, x, t, y):
        '''
        x: x_t function (e.g. x(75) = sqrt(alpha(75)) * x(0) +  sqrt(1 - alpha(75)) * epsilon)
        t: randomly chosen timestep expressed as a 1D tensor
        y: image label data that are provided
        '''

        ## Transform the flat time tensors into their appropriate shape
        ## to encode its positional embeddings. Unsqueeze(-1)
        ## transforms it from [a,b,c] -> [[a], [b], [c]] with each
        ## value being float.
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        if y is not None:
            t += self.label_embedding(y)

        x1 = self.inc(x) # DoubleConv
        x2 = self.down1(x1, t) # Down
        x2 = self.sa1(x2) # SelfAttention
        x3 = self.down2(x2, t) # Down
        x3 = self.sa2(x3) # SelfAttention
        x4 = self.down3(x3, t) # Down
        x4 = self.sa3(x4) # SelfAttention

        x4 = self.bot1(x4) # DoubleConv
        x4 = self.bot2(x4) # DoubleConv
        x4 = self.bot3(x4) # DoubleConv

        ## The upsampling here has skip connections
        x = self.up1(x4, x3, t) # Up
        x = self.sa4(x) # SelfAttention
        x = self.up2(x, x2, t) # Up
        x = self.sa5(x) # SelfAttention
        x = self.up3(x, x1, t) # Up
        x = self.sa6(x) # SelfAttention
        output = self.outc(x) # Conv2D for to match final output image dimension
        return output













