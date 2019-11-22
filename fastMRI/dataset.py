import numpy as np

import torch
from torch.utils.data import DataLoader

from utils import Downsample
from mri_data import SliceData
from subsample import MaskFunc
from transforms import normalize_instance, to_tensor

class DataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(self, mask_func, resolution, which_challenge, use_seed=True, normalize=False):
        """
        Args:
            mask_func (common.subsample.MaskFunc): A function that can create a mask of
                appropriate shape.
            resolution (int): Resolution of the image.
            which_challenge (str): Either "singlecoil" or "multicoil" denoting the dataset.
            use_seed (bool): If true, this class computes a pseudo random number generator seed
                from the filename. This ensures that the same mask is used for all the slices of
                a given volume every time.
        """
        if which_challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
        self.mask_func = mask_func
        self.resolution = resolution
        self.which_challenge = which_challenge
        self.use_seed = use_seed
        self.normalize = normalize

    def __call__(self, kspace, target, attrs, fname, slice):
        """
        Args:
            kspace (numpy.array): Input k-space of shape (num_coils, rows, cols, 2) for multi-coil
                data or (rows, cols, 2) for single coil data.
            target (numpy.array): Target image
            attrs (dict): Acquisition related information stored in the HDF5 object.
            fname (str): File name
            slice (int): Serial number of the slice.
        Returns:
            (tuple): tuple containing:
                image (torch.Tensor): Zero-filled input image.
                target (torch.Tensor): Target image converted to a torch Tensor.
                mean (float): Mean value used for normalization.
                std (float): Standard deviation value used for normalization.
                norm (float): L2 norm of the entire volume.
        """

        # this is the original normalization method from fastMRI official code
        def normalize_image(x):
            x, mean, std = normalize_instance(x, eps=1e-11)
            x = x.clip(-6, 6)
            return x

        if target is not None:
            target = normalize_image(target)
        else:
            target = [0]

        # Apply mask
        seed = None if not self.use_seed else tuple(map(ord, fname))
        mask = self.mask_func(target.shape + (2,), seed)
        mask = mask[:, :, 0].numpy()

        m = min(float(np.min(target)), 0)
        target_01 = (target - m) / (6 - m) # normalization into the range [0, 1]
        image, _, _ = Downsample(target_01, mask)
        if self.normalize:
            target = target_01
        else:
            image = image * (6 - m) + m # for unet, to scale back
        #else:
        #    image, _, _ = Downsample(target - m, mask) # make sure that the data are non-negative before downsampling
        #    image += m

        target = to_tensor(target)
        image = to_tensor(image)
        mask = to_tensor(mask)
        return target.unsqueeze(0).float(), image.unsqueeze(0).float(), mask.float()


def MRIDataset(image_set, transform, config):
    '''
    transform: rescale the image into [0, 1]
    For our model, set transform True.
    For unet, set transform False.
    No data augmentation is implemented for fastMRI.
    '''

    train_mask = MaskFunc(*config.sampling_scheme)
    
    if image_set == 'train':
        root = config.root + '/singlecoil_train'
    elif image_set == 'test':
        root = config.root + '/singlecoil_val'
    dataset = SliceData(
        root=root,
        transform=DataTransform(train_mask, config.resolution, 'singlecoil', normalize=transform),
        sample_rate=1.,
        challenge='singlecoil',
    )
    return dataset

if __name__ == '__main__':
    from config import config
    test_loader = DataLoader(
        dataset=MRIDataset('test', True, config),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )
    
    import cv2
    for _, data in enumerate(test_loader):
        ori_image, image, _ = data
        image = image.numpy()
        target = ori_image.numpy()
        cv2.imshow('test.jpg', image[0, 0] * 255)
        cv2.imshow('test_ori.jpg', target[0, 0] * 255)
        cv2.waitKey(0)
