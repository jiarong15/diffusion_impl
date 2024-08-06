from .helper_module import DoubleConv, Down, Up, SelfAttention
import torch.nn as nn
import torch

class ModuleHelpers:

    def initial_input_convolutions(channels_in, channels_out):
        return DoubleConv(channels_in, channels_out)
    
    def down_sampling_with_attention(down_in_channels, down_out_channels, att_size):
        return nn.Sequential(
            Down(down_in_channels, down_out_channels),
            SelfAttention(down_out_channels, att_size)
        )
    
    def up_sampling_with_attention(up_in_channels, up_out_channels, att_size):
        return nn.Sequential(
            Up(up_in_channels, up_out_channels),
            SelfAttention(up_out_channels, att_size)
        )
    
    def middle_blocks(in_channel, middle_channel, out_channel):
        return nn.Sequential(
            DoubleConv(in_channel, middle_channel),
            DoubleConv(middle_channel, middle_channel),
            DoubleConv(middle_channel, out_channel)
        )
    

def make_zero_module(module):
    '''
    Loop through the tensors that make up the module
    and remove it from the gradient computation graph. 
    No gradient will be backpropagated along this variable.
    Then, we set the tensor to 0 tensor i.e <[0., 0., ..., 0.]>
    '''
    for p in module.parameters():
        p.detach().zero_()
    return module

class ParentUNet:

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



## Implementing one of the commonly used architectures in diffusion models
class UNet(nn.Module, ParentUNet):
    def __init__(self, c_in=3, c_out=3, num_classes=None,
                 time_dim=256, device='cuda', use_up_blocks=True):
        ## c_in and c_out denotes the channels of the image
        ## In this case, both take the value 3.
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.use_up_blocks = use_up_blocks

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
        
        if self.use_up_blocks:
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

        ## Related to self.conv_in
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

        if self.use_up_blocks:
            ## The upsampling here has skip connections
            x = self.up1(x4, x3, t) # Up
            x = self.sa4(x) # SelfAttention

            x = self.up2(x, x2, t) # Up
            x = self.sa5(x) # SelfAttention

            x = self.up3(x, x1, t) # Up
            x = self.sa6(x) # SelfAttention

            output = self.outc(x) # Conv2D for to match final output image dimensio
        else:
            output = x4

        return output





class ControlNet(nn.Module, ParentUNet):

    ## Copy weight of trained UNet model

    ## Create a zero parameters convolution for the last layer of the hint block

    ## Down and mid blocks will be made with 0 parameters throughout all layers

    def __init__(self, num_classes=None, model_locked=True, model_ckpt=None, device='cuda'):
        super().__init__()

        ## Load the trained UNet's parameters
        ## Should already be trained. We can load the model 
        ## checkpoint into this model
        self.trained_unet = UNet(num_classes=num_classes).to(device)

        if model_ckpt is not None:
            self.trained_unet.load_state_dict(torch.load(model_ckpt,
                                                        map_location=device),
                                                        strict=True)
        
        self.should_model_be_locked = model_locked

        ## Need to somehow indicate that when loading the
        ## model, we want to exclude the Up sampling Modules in
        ## this model.
        self.copy_control_net = UNet(num_classes=num_classes,
                                     use_up_blocks=False).to(device)

        ## We set strict = False because we will not enforce 
        ## that the keys of the original model must match this 
        ## control net model since we are omitting the Downsampling blocks !!!
        if model_ckpt is not None:
            self.copy_control_net.load_state_dict(torch.load(model_ckpt,
                                                        map_location=device),
                                                        strict=False)
            

        ## Hint block for extra features to be added to training,
        ## where for instance it could be canary edges, hough transforms
        ## and so on. Dimensions should match those of the first conv layer

        self.copy_control_net_hint_block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=(1,1)),
            nn.SiLU(),
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=(1,1)),
            nn.SiLU(),
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=(1,1)),
            nn.SiLU(),
            make_zero_module(nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0))
        )


        self.copy_control_net_down_block_convs = nn.Sequential(
            ## Need to scale to how many down block channels there are
            make_zero_module(nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0))
        )


        self.copy_control_net_mid_block_convs = nn.Sequential(
            ## Need to scale to how many down block channels there are
            make_zero_module(nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0))
        )

    
    def add_params(self):
        params = list(self.copy_control_net.parameters())
        params += list(self.copy_control_net_hint_block.parameters())
        params += list(self.copy_control_net_down_block_convs.parameters())
        params += list(self.copy_control_net_mid_block_convs.parameters())


        if not self.should_model_be_locked:
            ## Add the trained Upsampling output blocks
            ## if needed
            params += list(self.trained_unet.parameters())
            params += list(self.trained_unet.parameters())
            params += list(self.trained_unet.parameters())
        return params
    

    
    def forward(self, x, t, y, hint):
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
        
        trained_unet_t = t.clone()
        trained_unet_down_outs = []
        with torch.no_grad():
            train_unet_out = self.trained_unet.inc(x)

            trained_unet_down_outs.append(train_unet_out)
            train_unet_out = self.trained_unet.down1(train_unet_out, trained_unet_t)
            train_unet_out = self.trained_unet.sa1(train_unet_out)

            trained_unet_down_outs.append(train_unet_out)
            train_unet_out = self.trained_unet.down2(train_unet_out, trained_unet_t)
            train_unet_out = self.trained_unet.sa2(train_unet_out)

            trained_unet_down_outs.append(train_unet_out)
            train_unet_out = self.trained_unet.down3(train_unet_out, trained_unet_t)
            train_unet_out = self.trained_unet.sa3(train_unet_out)

        
        hint_out = self.copy_control_net_hint_block(hint)
        x1 = self.inc(x) # DoubleConv
        x1 += hint_out # Add the hint block for further downblock processing

        control_copy_down_outs = []

        control_copy_down_outs.append(self.copy_control_net_down_block_convs(x1))
        x2 = self.down1(x1, t) # Down
        x2 = self.sa1(x2) # SelfAttention

        control_copy_down_outs.append(self.copy_control_net_down_block_convs(x2))
        x3 = self.down2(x2, t) # Down
        x3 = self.sa2(x3) # SelfAttention

        control_copy_down_outs.append(self.copy_control_net_down_block_convs(x3))
        x4 = self.down3(x3, t) # Down
        x4 = self.sa3(x4) # SelfAttention



        x4 = self.bot1(x4) # DoubleConv
        train_unet_out = self.trained_unet.bot1(train_unet_out)
        train_unet_out += self.copy_control_net_mid_block_convs(x4)

        x4 = self.bot2(x4) # DoubleConv
        train_unet_out = self.trained_unet.bot2(train_unet_out)
        train_unet_out += self.copy_control_net_mid_block_convs(x4)

        x4 = self.bot3(x4) # DoubleConv
        train_unet_out = self.trained_unet.bot3(train_unet_out)
        train_unet_out += self.copy_control_net_mid_block_convs(x4)

        

        ## We have 3 up layer which follows the number
        ## of down layers to connect the skip connections
        trained_unet_down_out = trained_unet_down_outs.pop()
        control_copy_down_out = control_copy_down_outs.pop()
        train_unet_out = self.trained_unet.up1(train_unet_out,
                                               trained_unet_down_out + control_copy_down_out,
                                               trained_unet_t)
        train_unet_out = self.trained_unet.sa4(train_unet_out)


        trained_unet_down_out = trained_unet_down_outs.pop()
        control_copy_down_out = control_copy_down_outs.pop()
        train_unet_out = self.trained_unet.up2(train_unet_out,
                                               trained_unet_down_out + control_copy_down_out,
                                               trained_unet_t)
        train_unet_out = self.trained_unet.sa5(train_unet_out)


        trained_unet_down_out = trained_unet_down_outs.pop()
        control_copy_down_out = control_copy_down_outs.pop()
        train_unet_out = self.trained_unet.up3(train_unet_out,
                                               trained_unet_down_out + control_copy_down_out,
                                               trained_unet_t)
        train_unet_out = self.trained_unet.sa6(train_unet_out)

        output = self.outc(train_unet_out) # Conv2D for to match final output image dimension
        return output






