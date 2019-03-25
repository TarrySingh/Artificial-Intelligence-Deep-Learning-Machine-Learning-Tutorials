# BigGAN-PyTorch TFHub converter
This dir contains scripts for taking the [pre-trained generator weights from TFHub](https://tfhub.dev/s?q=biggan) and porting them to BigGAN-Pytorch.

In addition to the base libraries for BigGAN-PyTorch, to run this code you will need:

TensorFlow
TFHub
parse

Note that this code is only presently set up to run the ported models without truncation--you'll need to accumulate standing stats at each truncation level yourself if you wish to employ it.

To port the 128x128 model from tfhub, produce a pretrained weights .pth file, and generate samples using all your GPUs, run

`python converter.py -r 128 --generate_samples --parallel`