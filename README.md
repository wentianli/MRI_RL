# MRI_RL

This is the implementation of our AAAI 2020 paper:<br>
MRI Reconstruction with Interpretable Pixel-Wise Operations Using Reinforcement Learning

```
@inproceedings{li2020mri,
  title={MRI Reconstruction with Interpretable Pixel-Wise Operations Using Reinforcement Learning},
  author={Li, Wentian and Feng, Xidong and An, Haotian and Ng, Xiang Yao and Zhang, Yu-Jin},
  booktitle={AAAI},
  year={2020}
}
```

Parts of the code are borrowed from other repos, including [pixelRL](https://github.com/rfuruta/pixelRL) (for a2c algorithm), [DAGAN](https://github.com/nebulaV/DAGAN/) (for MICCAI and MRI dataset), [fastMRI](https://github.com/facebookresearch/fastMRI) (for fastMRI dataset and unet), and some others.

## Environment

I used Python 3.6.1, Pytorch 0.3.1.post2, torchvision 0.2.0, numpy 1.14.2, and tensorboardX 1.7.  
The code usually works fine on my machine with two GeForce GTX 1080, 
but some weird bugs appeared occasionally (ValueError from numpy and segmentation fault from running Unet).

## Data Preparation

For [MICCAI 2013 Grand Challenge](https://my.vanderbilt.edu/masi/workshops/) dataset, please download the data and extract images by running `prepare_data.py` in `MICCAI/`.

For [fastMRI](https://fastmri.med.nyu.edu/) dataset, the h5 files will directly be read.

## Training

To properly set the data path, you need to modify the variables `dataset` and `root` in the `config.py` file in `MICCAI/` or `fastMRI/` accordingly. The hyper-parameters are also set in `config.py`.

For MICCAI, run
```
python train.py --dataset MICCAI
```

For fastMRI, run
```
python train.py --dataset fastMRI
```

To train Unet on fastMRI, go to `unet/` and run
```
sh train.sh
```
See `unet/` for more details.

## Trained models

We povide our model trained on MICCAI with 30% mask, our model and Unet trained on fastMRI with 40% mask.

The trained models can be downloaded here: [百度网盘](https://pan.baidu.com/s/1y2OXdERwmeYZEGDsI-r2UQ).


## Testing

Run
```
python test.py --dataset MICCAI_or_fastMRI --model path_to_the_model
```

We also provide an example of testing on a custom dataset. Please see `hemorrhage/`.
