# Implementation of diffusion model for learning purpose. 

> Included the explanations for many of the variables, use case of the different functions and how it relates back to the research paper in the link below.
Mathematical source for reference: https://lilianweng.github.io/posts/2021-07-11-diffusion-models/

> ** The trained_unet_model.pt saved in model_storage dir is generated with
only a test run of small epochs size and noise steps as a check. Please re-run the training proper with UNet first to get an updated model checkpoint that can be used!

Download nvidia driver for GPU accelerated modelling: 
* https://heads0rtai1s.github.io/2021/02/25/gpu-setup-r-python-ubuntu/
* https://forums.developer.nvidia.com/t/installing-cuda-on-ubuntu-22-04-rxt4080-laptop/292899

Training with EMA blog:
* https://developer.nvidia.com/blog/rethinking-how-to-train-diffusion-models/

Creating hint blocks to aid training utilizing control net
* **Canny Edge**: 
1. https://medium.com/@rohit-krishna/coding-canny-edge-detection-algorithm-from-scratch-in-python-232e1fdceac7
2. https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html

Personal Notes:
* **1st optimization**: Classifier Free Guidance added helps to aid additional information during training by indicating the class label of the images trained assuming we have information to the classes when training. We train the model with labels and without labels at 90% and 10% of the time respectively
* **2nd optimization**: We apply EMA smoothing only after some training is done by the original model. This is for reasons explained in the EMA blog by nvidia above
* **3rd optimization**: Use of control net that can allow more features of the image to be part of training. e.g. Canny edges, Hough lines, user scribbles, segmentation maps, depths

Good to have:
* We can use a pretrained model or build another model and use it as a classifier model to predict the image lables given the noisy images and as it the classifier is trained, we use the gradient to help in diffusion training



