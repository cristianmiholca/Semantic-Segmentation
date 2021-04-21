import torch
import torch.nn as nn


class ConvBatchNormRelu(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, batch_norm=True):
        super(ConvBatchNormRelu, self).__init__()
        conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                         kernel_size=kernel_size, stride=stride, padding=padding, )
        self.layer = None
        if batch_norm:
            self.layer = nn.Sequential(conv, nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        else:
            self.layer = nn.Sequential(conv, nn.ReLU(inplace=True))

    def forward(self, x):
        return self.layer(x)


class Down2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down2, self).__init__()
        self.conv1 = ConvBatchNormRelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvBatchNormRelu(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        size = x.size()
        x, indices = self.pool(x)
        return x, indices, size


class Down3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down3, self).__init__()
        self.conv1 = ConvBatchNormRelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvBatchNormRelu(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = ConvBatchNormRelu(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        size = x.size()
        x, indices = self.pool(x)
        return x, indices, size


class Up2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up2, self).__init__()
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.conv1 = ConvBatchNormRelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvBatchNormRelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, indices, output_size):
        x = self.unpool(input=x, indices=indices, output_size=output_size)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Up3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up3, self).__init__()
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.conv1 = ConvBatchNormRelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvBatchNormRelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = ConvBatchNormRelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, indices, output_size):
        x = self.unpool(input=x, indices=indices, output_size=output_size)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x
