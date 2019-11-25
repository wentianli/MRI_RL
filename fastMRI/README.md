### Notes

Different from [fastMRI official code](https://github.com/facebookresearch/fastMRI/blob/38bbfe2117905f5a246714739e7d6bedbdaba649/models/unet/train_unet.py), 
we do down-sampling after cropping the central region of the image, which allows data consistency step.

The original normalization of the data `x` is
```
x = (x - x.mean()) / (x.std() + eps)
x = x.clip(-6, 6)
```
After this, we scale the image into the range [0, 1]
```
m = min(float(x.min()), 0)
x = (x - m) / (6 - m)
```
