import torch
from torch import optim
from tqdm import tqdm
import torch.nn as nn

from diffusion_package.unet_model import UNet
from diffusion_package.diffusion import Diffusion
from diffusion_package.utils import get_data_loader

MODEL_CHECKPOINT_PATH = './model_storage/trained_unet_model.pt'

class Args:
    pass

def train_distillation(args):
    device = args.device

    ## Load the UNet model
    teacher_model = UNet(num_classes=args.num_classes,
                            device=device,
                            dropout_prob=args.dropout_prob).to(device)
    student_model = UNet(num_classes=args.num_classes,
                            device=device,
                            dropout_prob=args.dropout_prob).to(device)

    ## Dataloader for input image data for training
    dataloader = get_data_loader(args.batch_size,
                                 args.is_data_loader_shuffle)
    

    teacher_model.load_state_dict(torch.load(MODEL_CHECKPOINT_PATH,
                                                        map_location=device),
                                                        strict=True)
        
    student_optimizer = optim.AdamW(student_model.parameters(), lr=args.lr)


    ## Utilize mean square error as the loss function
    mse = nn.MSELoss()
    teacher_noise_steps = 1000
    student_noise_steps = teacher_noise_steps // 2

    ## Load the Diffusion class
    teacher_diffusion = Diffusion(noise_steps=teacher_noise_steps, img_size=args.image_size,
                          use_parameterization=args.use_parameterization,
                          device=device)

    student_diffusion = Diffusion(noise_steps=student_noise_steps,
                                    img_size=args.image_size,
                                    use_parameterization=args.use_parameterization,
                                    device=device)
    
    ## We set teach model to evaluation mode
    ## while student model on train mode as
    ## we do not want gradient computation for 
    ## the teacher
    teacher_model.eval()
    student_model.train()

    for epoch in range(args.epochs):
        progress_bar = tqdm(dataloader)

        all_images_in_this_epoch = torch.tensor(())
        all_labels_in_this_epoch = torch.tensor(())

        for i, (images, labels, _) in enumerate(progress_bar):
            all_images_in_this_epoch = torch.cat((all_images_in_this_epoch, images), 0)
            all_labels_in_this_epoch = torch.cat((all_labels_in_this_epoch, labels), 0)

            images = images.to(device)
            labels = labels.to(device)

            ## Prevent gradient accumulation by clearing
            ## all the gradient computation the optimizer is tracking.
            ## Gradient is tracked throughout a network 
            ## because of backprop and thus we need to clear it after 
            ## each epoch is trained.
            student_optimizer.zero_grad()

            t = 2 * student_diffusion.sample_timesteps(images.shape[0]).to(device)

            ## Teacher diffusion with 
            ## timestep of 1 ahead
            x_t_plus_one, alpha_t_plus_one, sigma_t_plus_one, _ = teacher_diffusion.noise_images(images, t + 1)

            ## Student diffusion with 
            ## half the timestep of teacher behind
            _, alpha_s_half, sigma_s_half, _ = student_diffusion.noise_images(images, t // 2)

            ## Teacher diffusion on current timestep
            _, alpha_t, sigma_t, _ = teacher_diffusion.noise_images(images, t)

            v = teacher_model.forward_decision(x_t_plus_one, t + 1,
                                               labels, cfg_scale=args.cfg_scale)
            x_hat = (alpha_t_plus_one * x_t_plus_one - sigma_t_plus_one * v).clip(-1, 1)
            z_1 = alpha_t * x_hat + (sigma_t / sigma_t_plus_one) * (x_t_plus_one - alpha_t_plus_one * x_hat)
            
            v_1 = teacher_model.forward_decision(z_1, t, labels, cfg_scale=args.cfg_scale)

            x_hat_2 = (alpha_t * z_1 - sigma_t * v_1).clip(-1, 1)

            eps_2 = (x_t_plus_one - alpha_s_half * x_hat_2) / sigma_s_half
            v_2 = alpha_s_half * eps_2 - sigma_s_half * x_hat_2

            w = torch.pow(1 + alpha_s_half / sigma_s_half, 0.3)
            predicted_noise = student_model.forward_decision(x_t_plus_one, t, labels, cfg_scale=args.cfg_scale)

            loss = mse(w * predicted_noise, w * v_2)
            loss.backward()
            student_optimizer.step()

        ## At each epoch, as the UNet model is being trained,
        ## we run the algorithm described in the paper to see how close
        ## are the images generated from a random gaussian to the original
        ## image. The images created should get better as the model learns.
        ## Can add a helper function to save these sampled images.
        ## Additionally, as labels ought to be categorical in nature, 
        ## we explicitly cast labels to int64 to ensure that the type 
        ## integrity is retained and will work when we run it though the
        ## nn.Embedding later
        sampled_images = student_diffusion.sample(student_model,
                                                  all_labels_in_this_epoch.to(torch.int64),
                                                  None,
                                                  all_images_in_this_epoch.shape[0],
                                                  args.use_control_net,
                                                  cfg_scale=args.cfg_scale)


def launch_distillation():
    args = Args()
    args.run_name = 'diffusion_model_distilled'
    args.use_parameterization = True
    args.num_classes = 10
    args.epochs = 500
    args.batch_size = 12

    ## Hardcode use_control_net parameter as 0 
    args.use_control_net = 0
    args.is_data_loader_shuffle = True
    args.image_size = 32

    ## set dropout_prob to 0 to not use 
    ## classifier free guidance training run
    args.dropout_prob = 0.35
    args.cfg_scale = 8
    args.device = None
    args.lr = 3e-4
    train_distillation(args)

if __name__ == '__main__':
    launch_distillation()