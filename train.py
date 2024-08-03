from .unet_model import UNet
from .diffusion import Diffusion
from torch import optim
from tqdm import tqdm
import torch.nn as nn

def train(args):
    device = args.device

    ## Load the UNet model
    model = UNet().to(device)

    ## Dataloader for input image data for training
    dataloader = get_data(args)

    optimizer = optim.adamw(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()

    ## Load the Diffusion class
    diffusion = Diffusion(img_size=args.image_size, device=device)

    l = len(dataloader)


    for epoch in range(args.epochs):
        progress_bar = tqdm(dataloader)
        for i, (images, _) in enumerate(progress_bar):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

