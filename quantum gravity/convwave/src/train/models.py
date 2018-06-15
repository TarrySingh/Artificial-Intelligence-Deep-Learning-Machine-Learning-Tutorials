# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as func


# -----------------------------------------------------------------------------
# STANDARD FCN FOR SPECTROGRAMS
# -----------------------------------------------------------------------------

class SpectrogramFCN(nn.Module):

    # -------------------------------------------------------------------------
    # Initialize the net and define functions for the layers
    # -------------------------------------------------------------------------

    def __init__(self):

        # Inherit from the PyTorch neural net module
        super(SpectrogramFCN, self).__init__()

        # Convolutional layers: (in_channels, out_channels, kernel_size)
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=64,
                               kernel_size=(3, 7), padding=(1, 3), stride=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=(3, 7), padding=(1, 6),
                               stride=1, dilation=(1, 2))
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128,
                               kernel_size=(3, 7), padding=(1, 6),
                               stride=1, dilation=(1, 2))
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128,
                               kernel_size=(3, 7), padding=(1, 6),
                               stride=1, dilation=(1, 2))
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128,
                               kernel_size=(3, 7), padding=(1, 6),
                               stride=1, dilation=(1, 2))
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128,
                               kernel_size=(3, 7), padding=(1, 6),
                               stride=1, dilation=(1, 2))
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=128,
                               kernel_size=(3, 7), padding=(1, 9),
                               stride=1, dilation=(1, 3))
        self.conv8 = nn.Conv2d(in_channels=128, out_channels=1,
                               kernel_size=(1, 1), padding=(0, 0), stride=1)

        # Batch norm layers
        self.batchnorm1 = nn.BatchNorm2d(num_features=128)
        self.batchnorm2 = nn.BatchNorm2d(num_features=128)
        self.batchnorm3 = nn.BatchNorm2d(num_features=128)
        self.batchnorm4 = nn.BatchNorm2d(num_features=128)
        self.batchnorm5 = nn.BatchNorm2d(num_features=128)
        self.batchnorm6 = nn.BatchNorm2d(num_features=128)

        # Pooling layers
        self.pool = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

    # -------------------------------------------------------------------------
    # Define a forward pass through the network (apply the layers)
    # -------------------------------------------------------------------------

    def forward(self, x):

        # Layer 1
        # ---------------------------------------------------------------------
        x = self.conv1(x)
        x = func.elu(x)

        # Layers 2 to 3
        # ---------------------------------------------------------------------
        convolutions = [self.conv2, self.conv3, self.conv4, self.conv5,
                        self.conv6, self.conv7]
        batchnorms = [self.batchnorm1, self.batchnorm2, self.batchnorm3,
                      self.batchnorm4, self.batchnorm5, self.batchnorm6]

        for conv, batchnorm in zip(convolutions, batchnorms):
            x = conv(x)
            x = batchnorm(x)
            x = func.elu(x)
            x = self.pool(x)

        # Layer 8
        # ---------------------------------------------------------------------
        x = self.conv8(x)
        x = func.sigmoid(x)

        return x

    # -------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# STANDARD FCN FOR TIME SERIES
# -----------------------------------------------------------------------------

class TimeSeriesFCN(nn.Module):

    # -------------------------------------------------------------------------
    # Initialize the net and define functions for the layers
    # -------------------------------------------------------------------------

    def __init__(self):

        # Inherit from the PyTorch neural net module
        super(TimeSeriesFCN, self).__init__()

        size = 64

        # Convolutional layers: (in_channels, out_channels, kernel_size)
        self.conv00 = nn.Conv1d(in_channels=2, out_channels=size,
                                kernel_size=1, dilation=1, padding=0)
        self.convolutions = nn.ModuleList()

        self.conv01 = nn.Conv1d(in_channels=size, out_channels=size,
                                kernel_size=3, dilation=1, padding=1)
        self.convolutions.append(self.conv01)

        self.conv02 = nn.Conv1d(in_channels=size, out_channels=size,
                                kernel_size=2, dilation=2, padding=1)
        self.convolutions.append(self.conv02)

        self.conv03 = nn.Conv1d(in_channels=size, out_channels=size,
                                kernel_size=2, dilation=4, padding=2)
        self.convolutions.append(self.conv03)

        self.conv04 = nn.Conv1d(in_channels=size, out_channels=size,
                                kernel_size=2, dilation=8, padding=4)
        self.convolutions.append(self.conv04)

        self.conv05 = nn.Conv1d(in_channels=size, out_channels=size,
                                kernel_size=2, dilation=16, padding=8)
        self.convolutions.append(self.conv05)

        self.conv06 = nn.Conv1d(in_channels=size, out_channels=size,
                                kernel_size=2, dilation=32, padding=16)
        self.convolutions.append(self.conv06)

        self.conv07 = nn.Conv1d(in_channels=size, out_channels=size,
                                kernel_size=2, dilation=64, padding=32)
        self.convolutions.append(self.conv07)

        self.conv08 = nn.Conv1d(in_channels=size, out_channels=size,
                                kernel_size=2, dilation=128, padding=64)
        self.convolutions.append(self.conv08)

        self.conv09 = nn.Conv1d(in_channels=size, out_channels=size,
                                kernel_size=2, dilation=256, padding=128)
        self.convolutions.append(self.conv09)

        self.conv10 = nn.Conv1d(in_channels=size, out_channels=size,
                                kernel_size=2, dilation=512, padding=256)
        self.convolutions.append(self.conv10)

        self.conv11 = nn.Conv1d(in_channels=size, out_channels=size,
                                kernel_size=2, dilation=1024, padding=512)
        self.convolutions.append(self.conv11)

        self.conv12 = nn.Conv1d(in_channels=size, out_channels=size,
                                kernel_size=2, dilation=2048, padding=1024)
        self.convolutions.append(self.conv12)
        # This should give a receptive field of size 4096?

        self.conv13 = nn.Conv1d(in_channels=size, out_channels=1,
                                kernel_size=1, dilation=1, padding=0)

        # Batch norm layers
        self.batchnorms = nn.ModuleList()
        self.batchnorm00 = nn.BatchNorm1d(num_features=size)

        self.batchnorm01 = nn.BatchNorm1d(num_features=size)
        self.batchnorms.append(self.batchnorm01)

        self.batchnorm02 = nn.BatchNorm1d(num_features=size)
        self.batchnorms.append(self.batchnorm02)

        self.batchnorm03 = nn.BatchNorm1d(num_features=size)
        self.batchnorms.append(self.batchnorm03)

        self.batchnorm04 = nn.BatchNorm1d(num_features=size)
        self.batchnorms.append(self.batchnorm04)

        self.batchnorm05 = nn.BatchNorm1d(num_features=size)
        self.batchnorms.append(self.batchnorm05)

        self.batchnorm06 = nn.BatchNorm1d(num_features=size)
        self.batchnorms.append(self.batchnorm06)

        self.batchnorm07 = nn.BatchNorm1d(num_features=size)
        self.batchnorms.append(self.batchnorm07)

        self.batchnorm08 = nn.BatchNorm1d(num_features=size)
        self.batchnorms.append(self.batchnorm08)

        self.batchnorm09 = nn.BatchNorm1d(num_features=size)
        self.batchnorms.append(self.batchnorm09)

        self.batchnorm10 = nn.BatchNorm1d(num_features=size)
        self.batchnorms.append(self.batchnorm10)

        self.batchnorm11 = nn.BatchNorm1d(num_features=size)
        self.batchnorms.append(self.batchnorm11)

        self.batchnorm12 = nn.BatchNorm1d(num_features=size)
        self.batchnorms.append(self.batchnorm12)

    # -------------------------------------------------------------------------
    # Define a forward pass through the network (apply the layers)
    # -------------------------------------------------------------------------

    def forward(self, x):

        # Layer 0
        # ---------------------------------------------------------------------
        x = self.conv00(x)
        x = self.batchnorm00(x)
        x = func.elu(x)

        # Layers 1 to 12
        # ---------------------------------------------------------------------

        # Apply all layers (either forward or backward)
        for conv, batchnorm in list(zip(self.convolutions, self.batchnorms)):

            x = conv(x)
            x = batchnorm(x)
            x = func.elu(x)

        # Layer 13
        # ---------------------------------------------------------------------
        x = self.conv13(x)
        x = func.sigmoid(x)

        return x

    # -------------------------------------------------------------------------
