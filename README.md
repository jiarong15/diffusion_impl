# Implementation of diffusion model for learning purpose. 

> Included the explanations for many of the variables, the use case of the different functions and how it relates back to the research paper in the link above.
Mathematical source for diffusion models: https://lilianweng.github.io/posts/2021-07-11-diffusion-models/

Download nvidia driver for GPU accelerated modelling: 
* https://heads0rtai1s.github.io/2021/02/25/gpu-setup-r-python-ubuntu/
* https://forums.developer.nvidia.com/t/installing-cuda-on-ubuntu-22-04-rxt4080-laptop/292899

Training with EMA blog:
* https://developer.nvidia.com/blog/rethinking-how-to-train-diffusion-models/


Personal Notes:
* 1st optimization: Classifier Free Guidance added helps to aid additional information during training by indicating the class label of the images trained assuming we have information to the classes when training. We train the model with labels and without labels at 90% and 10% of the time respectively.
* 2nd optimization: We apply EMA smoothing only after some training is done by the original model. This is for reasons explained in the EMA blog by nvidia above



