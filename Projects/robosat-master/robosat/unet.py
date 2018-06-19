'''The "U-Net" architecture for semantic segmentation.

See:
- https://arxiv.org/abs/1505.04597 - U-Net: Convolutional Networks for Biomedical Image Segmentation
- https://arxiv.org/abs/1411.4038  - Fully Convolutional Networks for Semantic Segmentation

'''

import torch
import torch.nn as nn


def Block(num_in, num_out):
    '''Creates a single U-Net building block.

    Args:
      num_in: number of input feature maps for the convolutional layer.
      num_out: number of output feature maps for the convolutional layer.

    Returns:
      The U-Net's building block module.
    '''
    return nn.Sequential(
        nn.Conv2d(num_in, num_out, kernel_size=3, padding=1),
        nn.BatchNorm2d(num_out),
        nn.PReLU(num_parameters=num_out),
        nn.Conv2d(num_out, num_out, kernel_size=3, padding=1),
        nn.BatchNorm2d(num_out),
        nn.PReLU(num_parameters=num_out))


def Downsample():
    '''Downsamples the spatial resolution by a factor of two.

    Returns:
      The downsampling module.
    '''

    return nn.MaxPool2d(kernel_size=2, stride=2)


def Upsample(num_in):
    '''Upsamples the spatial resolution by a factor of two.

    Args:
      num_in: number of input feature maps for the transposed convolutional layer.

    Returns:
      The upsampling module.
    '''

    return nn.ConvTranspose2d(num_in, num_in // 2, kernel_size=2, stride=2)


class UNet(nn.Module):
    '''The "U-Net" architecture for semantic segmentation.

    See: https://arxiv.org/abs/1505.04597
    '''

    def __init__(self, num_classes):
        '''Creates an `UNet` instance for semantic segmentation.

        Args:
          num_classes: number of classes to predict.
        '''

        super().__init__()

        self.block1 = Block(3, 64)
        self.down1 = Downsample()

        self.block2 = Block(64, 128)
        self.down2 = Downsample()

        self.block3 = Block(128, 256)
        self.down3 = Downsample()

        self.block4 = Block(256, 512)
        self.down4 = Downsample()

        self.block5 = Block(512, 1024)
        self.up1 = Upsample(1024)

        self.block6 = Block(1024, 512)
        self.up2 = Upsample(512)

        self.block7 = Block(512, 256)
        self.up3 = Upsample(256)

        self.block8 = Block(256, 128)
        self.up4 = Upsample(128)

        self.block9 = Block(128, 64)
        self.block10 = nn.Conv2d(64, num_classes, kernel_size=1)

        self.initialize()

    def forward(self, x):
        '''The networks forward pass for which autograd synthesizes the backwards pass.

        Args:
          x: the input tensor

        Returns:
          The networks output tensor.
        '''

        block1 = self.block1(x)
        down1 = self.down1(block1)

        block2 = self.block2(down1)
        down2 = self.down2(block2)

        block3 = self.block3(down2)
        down3 = self.down3(block3)

        block4 = self.block4(down3)
        down4 = self.down4(block4)

        block5 = self.block5(down4)
        up1 = self.up1(block5)

        block6 = self.block6(torch.cat([block4, up1], dim=1))
        up2 = self.up2(block6)

        block7 = self.block7(torch.cat([block3, up2], dim=1))
        up3 = self.up3(block7)

        block8 = self.block8(torch.cat([block2, up3], dim=1))
        up4 = self.up4(block8)

        block9 = self.block9(torch.cat([block1, up4], dim=1))
        block10 = self.block10(block9)

        return block10

    def initialize(self):
        '''Initializes the network's layers.
        '''

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                nn.init.constant_(module.bias, 0)
            if isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
