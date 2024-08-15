import torch
from torch import optim
from tqdm import tqdm
import copy
import torch.nn as nn


from diffusion_package.unet_model import UNet, ControlNet, ConvNextV2ForImageClassificationWithAttributes
from diffusion_package.diffusion import Diffusion
from diffusion_package.utils import get_data_loader
from diffusion_package.helper_module import EMA

MODEL_CHECKPOINT_PATH = './model_storage/trained_unet_model.pt'

class Args:
    pass

def train(args):
    device = args.device

    ## Load the UNet model or the ControlNet model
    if args.use_control_net == 1:
        model = ControlNet(num_classes=args.num_classes,
                           device=device,
                           dropout_prob=args.dropout_prob).to(device)
    else:
        model = UNet(num_classes=args.num_classes,
                     device=device,
                     dropout_prob=args.dropout_prob).to(device)
        classifier_model = ConvNextV2ForImageClassificationWithAttributes(num_classes=args.num_classes,
                                                                          device=device).to(device)

    ## Dataloader for input image data for training
    dataloader = get_data_loader(args.batch_size,
                                 args.is_data_loader_shuffle)

    if args.use_control_net == 1:
        optimizer = optim.AdamW(model.get_control_net_params(), lr=args.lr)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        classifier_optimizer = optim.AdamW(classifier_model.parameters(), lr=1e-5) 


    ## Utilize mean square error as the loss function
    mse = nn.MSELoss()

    ## Load the Diffusion class
    diffusion = Diffusion(img_size=args.image_size,
                          use_parameterization=args.use_parameterization,
                          device=device)

    ema = EMA(beta=0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    for epoch in range(args.epochs):
        progress_bar = tqdm(dataloader)

        all_images_in_this_epoch = torch.tensor(())
        all_labels_in_this_epoch = torch.tensor(())
        all_edges_in_this_epoch = torch.tensor(())

        for i, (images, labels, edges) in enumerate(progress_bar):
            all_images_in_this_epoch = torch.cat((all_images_in_this_epoch, images), 0)
            all_labels_in_this_epoch = torch.cat((all_labels_in_this_epoch, labels), 0)
            all_edges_in_this_epoch = torch.cat((all_edges_in_this_epoch, edges), 0)

            images = images.to(device)
            labels = labels.to(device)
            edges = edges.to(device)

            ## We build a timestep tensor of size as huge as the number
            ## of image training data that we have. This time t is a 
            ## random integer generated.
            t = diffusion.sample_timesteps(images.shape[0]).to(device)

            ## We sample x(t) at the random timestep we have just found 
            ## with the timestep tensor above. This means that for the random
            ## integer of time t say t = 75, we get an arbitrary image 
            ## at x(75) = sqrt(alpha(75)) * x(0) +  sqrt(1 - alpha(75)) * epsilon.
            ## 75 steps of gaussian noise added from the original image at x(0).
            x_t, classifier_guided_weights, noise = diffusion.noise_images(images, t)
       
            ## Prevent gradient accumulation by clearing
            ## all the gradient computation the optimizer is tracking.
            ## Gradient is tracked throughout a network 
            ## because of backprop and thus we need to clear it after 
            ## each epoch is trained.
            optimizer.zero_grad()

            ## Run the ML model (UNet) and get the predicted noise
            ## Algorithm 1 of the paper and depending on uncond_sampli_prob,
            ## We train the model with labels and without labels at 
            ## 90% and 10% of the time respectively.
            if args.use_control_net == 1:
                predicted_noise = model.forward_decision(x_t, t, labels, edges, cfg_scale=args.cfg_scale)
            else:
                predicted_noise = model.forward_decision(x_t, t, labels, cfg_scale=args.cfg_scale)
                
                ## Perform classifier guided diffusionn
                classifier_optimizer.zero_grad()  # Clear any existing gradients
                classifer_outputs = classifier_model.forward_decision(x_t, t, labels, cfg_scale=args.cfg_scale)
                classifier_loss = classifer_outputs.loss
                classifier_loss.backward()

                # Calculate the overall gradient norm
                total_grad_norm_curr_round = 0
                for classifier_params in classifier_model.parameters():
                    if classifier_params.grad is not None:
                        param_norm = classifier_params.grad.detach().data.norm(2)
                        total_grad_norm_curr_round += param_norm.item() ** 2
                total_grad_norm_curr_round = total_grad_norm_curr_round ** 0.5

                ## Classifier guidance weight w
                ## can range from 0.1 to 10. The more it 
                ## controls the strength of the class guidance
                w = 7.5
                predicted_noise = predicted_noise - classifier_guided_weights * w * total_grad_norm_curr_round

            ## Calculate the mean squared error of
            ## predicted noise and the actual noise as the loss value
            loss = mse(noise, predicted_noise)

            ## Computes loss for every parameter in the network.
            loss.backward()

            ## Iterates over all parameters it should update and 
            ## and use the stored gradient from backward step
            ## to update the parameters
            optimizer.step()

            ## Update ema model params to the model params
            ## appropriately, either smoothed or just copying
            ## the parameters early on
            ema.step_ema(ema_model, model)

        ## At each epoch, as the UNet model is being trained,
        ## we run the algorithm described in the paper to see how close
        ## are the images generated from a random gaussian to the original
        ## image. The images created should get better as the model learns.
        ## Can add a helper function to save these sampled images.
        ## Additionally, as labels ought to be categorical in nature, 
        ## we explicitly cast labels to int64 to ensure that the type 
        ## integrity is retained and will work when we run it though the
        ## nn.Embedding later
        sampled_images = diffusion.sample(model, all_labels_in_this_epoch.to(torch.int64),
                                          all_edges_in_this_epoch,
                                          all_images_in_this_epoch.shape[0],
                                          args.use_control_net,
                                          cfg_scale=args.cfg_scale)
        ema_sampled_images = diffusion.sample(ema_model, all_labels_in_this_epoch.to(torch.int64),
                                              all_edges_in_this_epoch,
                                              all_images_in_this_epoch.shape[0],
                                              args.use_control_net,
                                              cfg_scale=args.cfg_scale)
    
    ## Only save when we run the normal model
    ## since part of the initialization in the 
    ## controlnet model requires the strict
    ## frozen state dicts mapping of the  
    ## normal Unet model
    if args.use_control_net != 1:
        torch.save(model.state_dict(), MODEL_CHECKPOINT_PATH)


def launch():
    args = Args()
    args.run_name = 'diffusion_model'
    args.use_parameterization = True
    args.num_classes = 10
    args.epochs = 500
    args.batch_size = 12
    args.use_control_net = 1
    args.is_data_loader_shuffle = True
    args.image_size = 32

    ## set dropout_prob to 0 to not use 
    ## classifier free guidance training run
    args.dropout_prob = 0.35
    args.cfg_scale = 8
    args.device = None
    args.lr = 3e-4
    train(args)

if __name__ == '__main__':
    launch()