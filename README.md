# srresnet4x
Super resolution with srresnet using TensorFlow.
the attached model is supesialized in cartoons.

I referrd this paper:https://arxiv.org/abs/1609.04802
this implementation is not GAN.


left:input right:output

<img src = 'output/0_val.png' >


## Issue
It takes very long time for model building in this implementation.

The cause is probably subpixel convolution (pixel shuffler).

I will fix it as soon as a better way is found.
