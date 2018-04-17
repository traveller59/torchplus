import functools
import inspect
import sys
from collections import OrderedDict

import torch
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torchplus.nn.functional import group_norm

class _GroupBatchNorm(torch.nn.Module):
    def __init__(self,
                 num_features,
                 group=1,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True):
        super(_GroupBatchNorm, self).__init__()
        assert num_features % group == 0
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.group = group
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(group))
            self.register_buffer('running_var', torch.ones(group))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        return NotImplemented

    def forward(self, input):
        self._check_input_dim(input)
        ret = group_norm(input, self.running_mean, self.running_var,
                         self.weight, self.bias, self.training
                         or not self.track_running_stats, self.momentum,
                         self.eps, self.group)
        return ret

    def __repr__(self):
        return ('{name}({num_features}, eps={eps}, momentum={momentum},'
                ' affine={affine}, track_running_stats={track_running_stats})'
                .format(name=self.__class__.__name__, **self.__dict__))


class GroupNorm2d(_GroupBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(
                input.dim()))


class GroupNorm1d(_GroupBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 3:
            raise ValueError('expected 3D input (got {}D input)'.format(
                input.dim()))


class GroupNorm3d(_GroupBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(
                input.dim()))
