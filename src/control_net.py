
import torch.nn as nn
import torch
from diffusion_package.unet_model import UNet

class Args:
    pass

args = Args()

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

class ControlNet(nn.Module):

    ## Copy weight of trained UNet model

    ## Create a zero parameters convolution for the last layer of the hint block

    ## Down and mid blocks will be made with 0 parameters throughout all layers

    def __init__(self, model_locked=True, model_ckpt=None, device=None):
        super().__init__()


        ## Load the trained UNet's parameters
        ## Should already be trained. We can load the model 
        ## checkpoint into this model
        self.trained_unet = UNet(num_classes=args.num_classes).to(device)
        self.trained_unet.load_state_dict(torch.load(model_ckpt,
                                                     map_location=device),
                                                     strict=True)
        
        self.should_model_be_locked = model_locked

        ## Need to somehow indicate that when loading the this 
        ## model, we want to exclude the Up sampling Modules in
        ## this model.
        self.copy_control_net = UNet(num_classes=args.num_classes).to(device)

        ## We set strict = False because we will not enforce 
        ## that the keys of the original model must match this 
        ## control net model since we are omitting the Downsampling blocks
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








    


