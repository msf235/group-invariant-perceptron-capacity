import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.common_types import _size_2_t, _size_3_t
from typing import *
import collections.abc
from itertools import repeat
import math


container_abcs = collections.abc


def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


_pair = _ntuple(2)
_triple = _ntuple(3)

# %% 
class PeriodicAggregate2D(nn.Module):
    def __init__(self, period, agg_func='mean'):
        super().__init__()
        self.period = period
        if agg_func in {'avg', 'mean'}:
            self.agg_func = torch.sum
        elif agg_func == 'sum':
            self.agg_func = torch.sum
        elif agg_func == 'max':
            self.agg_func = torch.max
        else:
            self.agg_func = agg_func


    def forward(self, input):
        output_dim_2 = self.period[1]
        output_dim_1 = self.period[0]
        output = torch.zeros(*input.shape[:-2], output_dim_1, output_dim_2)
        for k2 in range(output_dim_2):
            for k1 in range(output_dim_1):
                t1 = input[...,:,k2::self.period[1]]
                input_agg_1 = self.agg_func(t1, dim=-1)
                t2 = input_agg_1[...,k1::self.period[0]]
                input_agg_12 = self.agg_func(t2, dim=-1)
                output[..., k1, k2] = input_agg_12
        return output


class MultiplePeriodicAggregate2D(nn.Module):
    def __init__(self, periods, agg_func='mean'):
        super().__init__()
        self.AggLayers = [PeriodicAggregate2D(period, agg_func)
                          for period in periods]


    def forward(self, input):
        output = [L(input) for L in self.AggLayers]
        return output


if __name__ == '__main__':
    x = torch.randn(5, 3, 24, 24)
    convL = torch.nn.Conv2d(3, 64, (3, 3), padding='same') 
    aggL = PeriodicAggregate2D((6,6))
    maggL = MultiplePeriodicAggregate2D(((6,6), (4,4)))
    y1 = convL(x)
    y2 = aggL(y1)
    y3 = maggL(y1)
