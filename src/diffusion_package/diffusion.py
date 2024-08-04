from tqdm import tqdm
import torch

class Diffusion:

    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=64, device='cuda'):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        ## Beta is the values for noise variance parameter
        ## used at each time step. It influences the amount of noise
        ## introduced to the image data at some timestep t
        self.beta = self.prepare_schedule().to(self.device)
        self.alpha = 1. - self.beta

        ## Get the cumulative product of alphas from timestep 1 to timestep n.
        ## Used to derive the formula of 
        ## x(t) = sqrt(alpha(t)) * x(0) +  sqrt(1 - alpha(t)) * epsilon
        ## in the noise image function.
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)


    def prepare_schedule(self):
        '''
        Create 1D tensor evenly space from start to end over
        the number noise steps
        '''
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    

    def noise_images(self, x, t):
        '''
        Build the function 
        x(t) = sqrt(alpha(t)) * x(0) +  sqrt(1 - alpha(t)) * epsilon
        x(t) denotes the image at timestep t
        x(0) denotes the original image
        '''

        ## After sampling alpha and beta at time step i, it becomes a 
        ## 1D tensor. [:, None, None, None] changes the dimensions to
        ## this shape (n, 1, 1, 1)
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alpha_hat[t])[:, None, None, None]

        ## Build a tensor of same tensor size as x sampled 
        ## from a gaussian of mean 0 and variance 1
        epsilon = torch.rand_like(x)

        ## Return a tuple of 2 parameters where the first parameter is 
        ## the function x(t) ustilizing the reparameterization trick
        ## Second parameter is the noise.
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon
    
    def sample_timesteps(self, n):
        '''
        Generates a tensor of size n where the numbers in tensor
        ranges from (1, self.noise_steps). In this case self.noise_steps
        is the amount of steps going from an image x(t) of pure gaussian noise
        to x(0) of the original image used for training.
        '''
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, labels, n, cfg_scale=8):

        ## Used to switch off behaviours in layers/parts
        ## of the ML model even as we inference it in the loop
        ## below to predict noise with model(x, t)
        model.eval()

        ## Disables gradient computation
        with torch.no_grad():
            ## Creating sample images from gaussian distribution of 
            ## size similar to the sizes of trained images 
            ## e.g. if images trained are 64 x 64, so will the random normal images
            ## We will 3 channels with n of such samples
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)

            ## Loop used to reverse the diffusion process
            ## from timestep t back to achieve the original image at timestep 0
            ## This follows algorithm from the Ho et al. 2020 paper
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):

                ## Tensor t is used to sample noise introduced at 
                ## timestep i for n times below as used in self.alpha[t]
                ## as well as for indicating model prediction at a particular timestep.
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)

                ## We don't want to include the labels, so
                ## we are just conditioning on the time
                ## and leaving out the labels
                unconditioned_predicted_noise = model(x, t, None)

                ## We then perform linear interpolation to move towards
                ## the conditional sample over the unconditional sample
                ## as following the CFG formula. Also a usual CFG scale
                ## of 7.5 - 10 is used. I chose 8
                predicted_noise = torch.lerp(unconditioned_predicted_noise,
                                             predicted_noise,
                                             cfg_scale)

                ## After sampling alpha, alpha_hat and beta at time step i, it 
                ## becomes a 1D tensor.
                ## [:, None, None, None] changes the dimensions to
                ## this shape (n, 1, 1, 1)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]

                ## Creating tensors the size of training images
                ## The noise used appropriately as the loop reaches x(0)
                ## where there should be no noise terms when i equals 1
                ## as it ought to converge to the original image at x(0).
                if i > 1:
                    ## Create tensor of uniform distribution
                    noise = torch.rand_like(x)
                else:
                    ## Create tensor of values 0
                    noise = torch.zeros_like(x)

                ## Formula derived from the proof using bayes rule for q(x(t-1) | x(t), x(0)) 
                ## Formula is conditioned on x(0) such that it make the training more stable
                ## knowing that we need to achieve the original image from
                ## an image constructed from gaussian noise. Additionally, we are trying to
                ## train mu(theta) to be as close to the mu(t) formula derived mathematically
                ## in the proof of q(x(t-1) | x(t), x(0)), hence we are using the predicted_noise
                ## in this scenario. Algorithm 2 of the paper
                x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        
        ## Turning back to training mode after inference
        model.train()

        ## Pixel normalization allowing better neural network convergence
        ## Also, (0, 255) is the usual pixel range where 0 represents
        ## black while 255 represents white. This is similar to using 
        ## cv2.Normalize() on the image
        x = (x.clamp(-1, 1) + 1) / 2

        ## Altering the images such that they are in the valid pixel range
        x = (x * 255).type(torch.uint8)
        return x



