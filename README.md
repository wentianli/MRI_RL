# MRI_RL

This is the implementation of the AAAI 2020 paper.<br>

```
..
```

Parts of the code are borrowed from other repos, including  

The code  is based on [pixelRL](https://github.com/BVLC/caffe/blob/master/include/caffe/layers/softmax_loss_layer.hpp) (for a2c algorithm)
the code  is based on [DAGAN]() (for MRI and MICCAI dataset), [fastMRI] (for unet and fastMRI dataset), and (for learning rate scheme).

## Environment

I used Python 3.6.1, Pytorch 0.3.1.post2, torchvision 0.2.0, numpy 1.14.2, and tensorboardX 1.7.
The code usually works fine on my machine with 2 gpus (GeForce GTX 1080), 
but some weird bugs appeared occasionally (ValueError from numpy and segmentation fault from running Unet).

## Data Preparation

For MICCAI, please download the data and extract images by running `prepare_data.py` in `MICCAI/`.

For fastMRI, please download the data somewhere.

For ..., please unzip the file.

To properly set the data path, you need to modify the variables `dataset` and `root` in the `config.py` file.

## Training

