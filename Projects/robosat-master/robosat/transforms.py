'''PyTorch-compatible transformations.
'''

import torch
import numpy as np

import torchvision


# Callable to convert a RGB image into a PyTorch tensor.
ImageToTensor = torchvision.transforms.ToTensor


class MaskToTensor:
    '''Callable to convert a PIL image into a PyTorch tensor.
    '''

    def __call__(self, image):
        '''Converts the image into a tensor.

        Args:
          image: the PIL image to convert into a PyTorch tensor.

        Returns:
          The converted PyTorch tensor.
        '''

        return torch.from_numpy(np.array(image, dtype=np.uint8)).long()


class ConvertImageMode:
    def __init__(self, mode):
        self.mode = mode

    def __call__(self, image):
        return image.convert(self.mode)
