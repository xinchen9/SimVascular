""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class HiddenConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HiddenConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1,bias=False)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(out_channels)

    def forward(self, x):
        output = self.conv(x)
        output = self.batchnorm(output)
        output = self.relu(output)
        return output


class OutFusion(nn.Module):
    """Custom weight accumulation layer
    for learned weight output
    """
    def __init__(self, n_classes, height, width):
        super(OutFusion, self).__init__()
        self.n_classes = n_classes
        self.height = height
        self.width = width
        weight = torch.ones((self.n_classes,self.height,self.width), dtype=torch.float32)
        weight = weight * 0.5
        if torch.cuda.is_available():
            weight = weight.cuda()
        self.weight = weight
        
    def forward(self, input1, input2):
        result = torch.zeros(input1.size())
        if torch.cuda.is_available():
            result = result.cuda()
        # import pdb
        # pdb.set_trace()
        weight_rem = 1.0 - self.weight
        for i in range(self.n_classes):
            result[:,i,:,:] = torch.mul(input1[:,i,:,:], self.weight[i,:,:])
            result[:,i,:,:] = torch.mul(input2[:,i,:,:], weight_rem[i,:,:])
        return result



        