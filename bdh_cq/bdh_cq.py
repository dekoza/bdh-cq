from __future__ import annotations

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from einops import rearrange

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if v is not None else d

# classes

class BDH(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError
