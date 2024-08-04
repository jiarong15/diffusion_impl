# Implementation of diffusion model for learning purpose. 

## Included the explanations for many of the variables, the use case of the different functions and how it relates back to the research paper in the link above.
Mathematical source for diffusion models: https://lilianweng.github.io/posts/2021-07-11-diffusion-models/

Download nvidia driver for GPU accelerated modelling: https://heads0rtai1s.github.io/2021/02/25/gpu-setup-r-python-ubuntu/



wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub


# Add NVIDIA repos
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
sudo apt-get update && sudo apt-get upgrade


# Install NVIDIA driver & CUDA
# + stable for RTX 4090 + TF 2.13
sudo apt-get install cuda-drivers cuda-11-8 libcudnn8=8.9.2.26-1+cuda11.8 libcudnn8-dev=8.9.2.26-1+cuda11.8 libnccl2 libnccl-dev

# - latest for TF 2.16.1
# sudo apt-get install cuda-drivers cuda-12-2 libcudnn8 libcudnn8-dev libnccl2 libnccl-dev


# Reboot. Check that GPUs are visible
nvidia-smi