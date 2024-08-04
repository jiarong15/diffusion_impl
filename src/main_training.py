import torch
from torch import optim
from tqdm import tqdm
import copy
import torch.nn as nn
import numpy as np

from diffusion_package.unet_model import UNet
from diffusion_package.diffusion import Diffusion
from diffusion_package.utils import get_data_loader
from diffusion_package.helper_module import EMA

class Args:
    pass

def train(args):
    device = args.device

    ## Load the UNet model
    model = UNet(num_classes=args.num_classes).to(device)

    ## Dataloader for input image data for training
    dataloader = get_data_loader(args.batch_size, args.is_data_loader_shuffle)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    ## Utilize mean square error as the loss function
    mse = nn.MSELoss()

    ## Load the Diffusion class
    diffusion = Diffusion(img_size=args.image_size, device=device)

    ema = EMA(beta=0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    for epoch in range(args.epochs):
        progress_bar = tqdm(dataloader)

        all_images_in_this_epoch = torch.tensor(())
        all_labels_in_this_epoch = torch.tensor(())

        for i, (images, labels) in enumerate(progress_bar):
            all_images_in_this_epoch = torch.cat((all_images_in_this_epoch, images), 0)
            all_labels_in_this_epoch = torch.cat((all_labels_in_this_epoch, labels), 0)

            images = images.to(device)
            labels = labels.to(device)

            ## We build a timestep tensor of size as huge as the number
            ## of image training data that we have. This time t is a 
            ## random integer generated.
            t = diffusion.sample_timesteps(images.shape[0]).to(device)

            ## We sample x(t) at the random timestep we have just found 
            ## with the timestep tensor above. This means that for the random
            ## integer of time t say t = 75, we get an arbitrary image 
            ## at x(75) = sqrt(alpha(75)) * x(0) +  sqrt(1 - alpha(75)) * epsilon.
            ## 75 steps of gaussian noise added from the original image at x(0).
            x_t, noise = diffusion.noise_images(images, t)

            is_uncond_sampli_prob = np.random.random() < 0.1
            if is_uncond_sampli_prob:
                labels = None

            ## Run the ML model (UNet) and get the predicted noise
            ## Algorithm 1 of the paper and depending on uncond_sampli_prob,
            ## We train the model with labels and without labels at 
            ## 90% and 10% of the time respectively.
            predicted_noise = model(x_t, t, labels)

            ## Calculate the mean squared error of
            ## predicted noise and the actual noise as the loss value
            loss = mse(noise, predicted_noise)

            ## Prevent gradient accumulation by clearing
            ## all the gradient computation the optimizer is tracking.
            ## Gradient is tracked throughout a network 
            ## because of backprop and thus we need to clear it after 
            ## each epoch is trained.
            optimizer.zero_grad()

            ## Computes loss for every parameter in the network.
            loss.backward()

            ## Iterates over all parameters it should update and 
            ## and use the stored gradient from backward step
            ## to update the parameters
            optimizer.step()

            ## Update ema model params to the model params
            ## appropriately, either smoothed or just copying
            ## the parameters early on
            ema.step(ema_model, model)

        ## At each epoch, as the UNet model is being trained,
        ## we run the algorithm described in the paper to see how close
        ## are the images generated from a random gaussian to the original
        ## image. The images created should get better as the model learns.
        ## Can add a helper function to save these sampled images.
        sampled_images = diffusion.sample(model, all_labels_in_this_epoch, n=all_images_in_this_epoch.shape[0])
        ema_sampled_images = diffusion.sample(ema_model, all_labels_in_this_epoch, n=all_images_in_this_epoch.shape[0])



def launch():
    args = Args()
    args.run_name = 'diffusion_model'
    args.num_classes = 10
    args.epochs = 500
    args.batch_size = 12
    args.is_data_loader_shuffle = True
    args.image_size = 32
    args.device = 'cuda'
    args.lr = 3e-4
    train(args)

if __name__ == '__main__':
    launch()