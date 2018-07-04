'''The "Pyramid Scene Parsing Network" (PSPNet) architecture for semantic segmentation.

See:
- https://arxiv.org/abs/1612.01105 - Pyramid Scene Parsing Network
- https://arxiv.org/abs/1706.05587 - Rethinking Atrous Convolution for Semantic Image Segmentation
- https://arxiv.org/abs/1511.07122 - Multi-Scale Context Aggregation by Dilated Convolutions
- https://arxiv.org/abs/1606.02147 - ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation
- https://arxiv.org/abs/1512.03385 - Deep Residual Learning for Image Recognition
'''

import torch
import torch.nn as nn
from torchvision.models import resnet50


def PyramidBlock(scale, num_in, num_out):
    '''Creates a single pyramid module for a specific scale.

    Args:
      scale: the pyramid's scale for pooling.
      num_in: number of input feature maps for the convolutional layer.
      num_out: number of output feature maps for the convolutional layer.

    Returns:
      The pyramid module.
    '''

    return nn.Sequential(
        nn.AdaptiveAvgPool2d(scale),
        nn.Conv2d(num_in, num_out, kernel_size=1, bias=False),
        nn.BatchNorm2d(num_out),
        nn.ReLU(inplace=True))


class PSPNet(nn.Module):
    '''The "Pyramid Scene Parsing Network" (PSPNet) architecture.

    See: https://arxiv.org/abs/1612.01105
    '''

    def __init__(self, num_classes, pretrained=False):
        '''Creates an `PSPNet` instance for semantic segmentation.

        Args:
          num_classes: number of classes to predict.
          pretrained: use a pre-trained `ResNet` backbone for convolutional feature extraction.
        '''

        super().__init__()

        # Backbone network we use to harvest convolutional image features from
        self.resnet = resnet50(pretrained=pretrained)

        # https://github.com/pytorch/vision/blob/c84aa9989f5256480487cafe280b521e50ddd113/torchvision/models/resnet.py#L101-L105
        self.block0 = nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu, self.resnet.maxpool)

        # https://github.com/pytorch/vision/blob/c84aa9989f5256480487cafe280b521e50ddd113/torchvision/models/resnet.py#L106-L109
        self.block1 = self.resnet.layer1
        self.block2 = self.resnet.layer2
        self.block3 = self.resnet.layer3
        self.block4 = self.resnet.layer4

        # See https://arxiv.org/abs/1606.02147v1 section 4: Information-preserving dimensionality changes
        #
        # "When downsampling, the first 1x1 projection of the convolutional branch is performed with a stride of 2
        # in both dimensions, which effectively discards 75% of the input. Increasing the filter size to 2x2 allows
        # to take the full input into consideration, and thus improves the information flow and accuracy."
        #
        # We can not change the kernel_size on the fly but we can change the stride instead from (2, 2) to (1, 1).

        assert self.block3[0].downsample[0].stride == (2, 2)
        assert self.block4[0].downsample[0].stride == (2, 2)

        self.block3[0].downsample[0].stride = (1, 1)
        self.block4[0].downsample[0].stride = (1, 1)

        # See https://arxiv.org/abs/1511.07122 and https://arxiv.org/abs/1706.05587 for dilated convolutions.
        # ResNets reduce spatial dimension too much for segmentation => patch in dilated convolutions.

        for name, module in self.block3.named_modules():
            if 'conv2' in name:
                module.dilation = (2, 2)
                module.padding = (2, 2)
                module.stride = (1, 1)

        for name, module in self.block4.named_modules():
            if 'conv2' in name:
                module.dilation = (4, 4)
                module.padding = (4, 4)
                module.stride = (1, 1)

        # PSPNet's pyramid: 2048 feature maps from conv net => pool into scales of {1, 2, 3, 6}
        self.pyramid1 = PyramidBlock(1, 2048, 2048 // 4)
        self.pyramid2 = PyramidBlock(2, 2048, 2048 // 4)
        self.pyramid3 = PyramidBlock(3, 2048, 2048 // 4)
        self.pyramid6 = PyramidBlock(6, 2048, 2048 // 4)

        # Pyramid pooling doubles feature maps via concatenation
        self.logits = nn.Sequential(
            nn.Conv2d(4096, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, num_classes, kernel_size=1))

        self.initialize()

    def forward(self, x):
        '''The networks forward pass for which autograd synthesizes the backwards pass.

        Args:
          x: the input tensor

        Returns:
          The networks output tensor.
        '''

        size = x.size()

        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        pyramid1 = nn.functional.upsample(self.pyramid1(x), size=x.size()[2:], mode='bilinear')
        pyramid2 = nn.functional.upsample(self.pyramid2(x), size=x.size()[2:], mode='bilinear')
        pyramid3 = nn.functional.upsample(self.pyramid3(x), size=x.size()[2:], mode='bilinear')
        pyramid6 = nn.functional.upsample(self.pyramid6(x), size=x.size()[2:], mode='bilinear')

        x = torch.cat([x, pyramid1, pyramid2, pyramid3, pyramid6], dim=1)

        x = self.logits(x)

        x = nn.functional.upsample(x, size=size[2:], mode='bilinear')

        return x

    def initialize(self):
        '''Initializes the network's layers.
        '''

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
