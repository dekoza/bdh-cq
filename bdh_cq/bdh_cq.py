from __future__ import annotations

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Embedding, Linear, Sequential, RMSNorm

from einops import rearrange

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if v is not None else d

def LinearNoBias(dim, dim_out):
    return Linear(dim, dim_out, bias = False)

# classes

class BDH(Module):
    def __init__(
        self,
        *,
        dim,
        num_tokens
    ):
        super().__init__()
        self.token_embed = Embedding(dim, num_tokens)

        self.to_logits = Sequential(
            RMSNorm(dim),
            LinearNoBias(dim, num_tokens)
        )

    def forward(
        self,
        ids
    ):
        tokens = self.token_embed(ids)

        logits = self.to_logits(tokens)

# quick test

if __name__ == '__main__':

    model = BDH(
        dim = 512,
        num_tokens = 256
    )

    ids = torch.randint(0, 256, (2, 1024))

    logits = model(ids)

    assert logits.shape == (2, 1024, 256)
