import functools
import inspect
import sys
from collections import OrderedDict

import torch
from torch.nn import functional as F
from torch.nn.parameter import Parameter

class SeparableConv2d(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 depth_multiplier=1,
                 stride=1,
                 padding=0,
                 dilation=0,
                 bias=True):
        super(SeparableConv2d, self).__init__()
        self.dw_conv2d = torch.nn.Conv2d(in_channels, depth_multiplier,
                                         kernel_size, stride, padding,
                                         dilation, in_channels, False)
        self.pw_conv2d = torch.nn.Conv2d(depth_multiplier * in_channels,
                                         out_channels, 1, 1, 0, 1, 1, bias)

    def forward(self, x):
        return self.pw_conv2d(self.dw_conv2d(x))


class SeparableConv1d(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 depth_multiplier=1,
                 stride=1,
                 padding=0,
                 dilation=0,
                 bias=True):
        super(SeparableConv1d, self).__init__()
        self.dw_conv1d = torch.nn.Conv1d(in_channels, depth_multiplier,
                                         kernel_size, stride, padding,
                                         dilation, in_channels, False)
        self.pw_conv1d = torch.nn.Conv1d(depth_multiplier * in_channels,
                                         out_channels, 1, 1, 0, 1, 1, bias)

    def forward(self, x):
        return self.pw_conv1d(self.dw_conv1d(x))