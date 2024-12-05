import copy
from pathlib import Path
from math import log2, ceil, sqrt
from functools import wraps, partial

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList
from torch.autograd import grad as torch_grad

import torchvision
from torchvision.models import VGG16_Weights

from collections import namedtuple

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange

from beartype import beartype
from beartype.typing import Union, Tuple, Optional, List
import numpy as np
#

def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


# helper

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def safe_get_index(it, ind, default = None):
    if ind < len(it):
        return it[ind]
    return default

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def identity(t, *args, **kwargs):
    return t

def divisible_by(num, den):
    return (num % den) == 0

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def append_dims(t, ndims: int):
    return t.reshape(*t.shape, *((1,) * ndims))

def is_odd(n):
    return not divisible_by(n, 2)

def maybe_del_attr_(o, attr):
    if hasattr(o, attr):
        delattr(o, attr)

def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else ((t,) * length)

#

def weight_init(m):
    if isinstance(m, nn.Linear):
        in_num = m.in_features
        out_num = m.out_features
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(
            m.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std
        )
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        space = m.kernel_size[0] * m.kernel_size[1]
        in_num = space * m.in_channels
        out_num = space * m.out_channels
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(
            m.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std
        )
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.LayerNorm):
        m.weight.data.fill_(1.0)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
