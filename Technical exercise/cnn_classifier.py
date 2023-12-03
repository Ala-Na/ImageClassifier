import torch
import numpy as np
from torch import nn
from tqdm.auto import tqdm
from torch.utils.data import DataLoader


class ResidualBlock(nn.Module):
    """
    Block to perform a skip connection / residual learning following
    https://arxiv.org/pdf/1512.03385.pdf.
    Init parameters:
        in_channels: number of channels in the input
        hidden_out_channels: size/channels of hidden layers outputs
        stride: stride to use on first convolutionnal operation
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = self._convolution_block(
            in_channels, out_channels, stride=stride, bias=False
        )
        self.conv2 = self._convolution_block(
            out_channels, out_channels, kernel_size=3, padding=1
        )
        self.relu = nn.LeakyReLU(0.2)
        self.downsample = None
        if stride != 1:
            self.downsample = self._downsample_block(in_channels, out_channels, stride)

    def _convolution_block(
        self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True
    ):
        """
        Block of operations including a convolution.
        Parameters:
            in_channels: number of channels in the input
            out_channels: number of channels produced by the convolution
            kernel_size: filter size of convolving kernel as an integer, more precisely (kernel_size, kernel_size)
            stride: stride of the convolution
            padding: padding to add prior to convolution operation
            bias: if a bias is used
        """
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            nn.BatchNorm2d(out_channels),
        )

    def _downsample_block(self, in_channels, out_channels, stride):
        """
        Block of operations to perform on identity weights for residual connection,
        for adapting dimensions.
        Parameters:
            in_channels: number of channels in the input
            out_channels: number of channels produced by the convolution
            stride: stride of the convolution
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, input):
        """
        Forward pass, using a residual connection. A downsample function is used on the
        identity input when needed.
        Parameters:
            input: input to perform operations on
        """
        identity = input
        out = self.conv1(input)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        out = self.relu(out)
        return out


class Classifier(nn.Module):
    """
    Classifier class for Road vs. Field dataset with ResNet18 architecture.
    Init parameters:
        in_channels: the size/number of channels of the picture
        hidden_out_channels: size/channels of hidden layers outputs
        nbr_classes: the number of classes for this classifier
    """

    def __init__(self, in_channels=3, hidden_out_channels=64, nbr_classes=2):
        super(Classifier, self).__init__()
        if nbr_classes < 2:
            raise Exception("Must at least have 2 classes for classification !")
        self.model = nn.Sequential(
            self._conv_pool_block(in_channels, hidden_out_channels),
            self._residual_conv_block(
                hidden_out_channels, hidden_out_channels, stride=1
            ),
            self._residual_conv_block(hidden_out_channels, hidden_out_channels * 2),
            self._residual_conv_block(hidden_out_channels * 2, hidden_out_channels * 4),
            self._residual_conv_block(hidden_out_channels * 4, hidden_out_channels * 8),
            self._fully_connected_block(nbr_classes),
        )

    def _conv_pool_block(self, in_channels, out_channels):
        """
        Block of initial operations including convolutionnal operation and maxpooling.
        Parameters:
            in_channels: the size/number of channels of the picture
            out_channels:  number of channels produced by the convolution
        """
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

    def _residual_conv_block(self, in_channels, out_channels, stride=2):
        """
        Block which perform convolutions and skip connections.
        Parameters:
            in_channels: the size/number of the input
            hidden_out_channels: size/channels of output
            stride: stride of the convolution
        """
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels, stride=stride),
            ResidualBlock(out_channels, out_channels),
        )

    def _fully_connected_block(self, nbr_classes):
        """
        Final block with fully connected layer to perform classification.
        An adaptive average pooling is performed on input to avoid uncompatible dimensions.
        Parameters:
            nbr_classes: the number of classes for this classifier
        """
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(512, nbr_classes)
        )

    def forward(self, image):
        out = self.model(image)
        return out
