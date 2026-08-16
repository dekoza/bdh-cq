from __future__ import annotations

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Embedding, Linear, LayerNorm, Sequential, RMSNorm

from einops import rearrange
from einops.layers.torch import Rearrange

from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(n, d):
    return (n % d) == 0

def LinearNoBias(dim, dim_out):
    return Linear(dim, dim_out, bias = False)

def LayerNormNoParams(dim):
    return LayerNorm(dim, elementwise_affine = False)

# classes

class LinearAttention(Module):
    def __init__(
        self,
        dim,
        *,
        heads,
        dim_queries_keys,
        dim_values,
        qk_activation = nn.ReLU()
    ):
        super().__init__()
        dim_inner_qk = dim_queries_keys * heads
        dim_inner_values = dim_values * heads

        self.prenorm = LayerNormNoParams(dim)

        self.to_qk = LinearNoBias(dim, dim_inner_qk)
        self.to_v = LinearNoBias(dim, dim_inner_values)

        self.qk_activation = qk_activation

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.to_out = LinearNoBias(dim_inner_values, dim)

    def forward(
        self,
        tokens,
        memories = None,
        rotary_emb = None,
        return_memories = False
    ):
        device = tokens.device

        tokens = self.prenorm(tokens)

        # queries and keys, relu activated

        q = k = self.to_qk(tokens)
        q, k = map(self.qk_activation, (q, k))

        # value

        v = self.to_v(tokens)

        # split

        q, k, v = (self.split_heads(t) for t in (q, k, v))

        # relative positions

        if exists(rotary_emb):
            q, k = (apply_rotary_emb(rotary_emb, t) for t in (q, k))

        # linear attention, omitting attention to self

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        i, j = sim.shape[-2:]
        causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).tril(-1) # omit self, seen in Reformer shared qk attention years ago

        attn = sim.masked_fill(~causal_mask, 0.)

        agg = einsum('b h i j, b h j d -> b h i d', attn, v)

        # past memories

        if exists(memories):
            retrieved = einsum('b h n d, b h d e -> b h n e', q, memories)
            agg = agg + retrieved

        # out

        out = self.merge_heads(agg)

        out = self.to_out(out)

        if not return_memories:
            return out

        memories = einsum('b h n d, b h n e -> b h d e', k, v)

        return out, memories

class BDH(Module):
    def __init__(
        self,
        *,
        dim,
        num_tokens,
        depth = 8,
        heads = 4,
        dim_qk_heads = 32_768 # their neurons is the dim_qk * heads
    ):
        super().__init__()
        assert divisible_by(dim_qk_heads, heads)
        dim_qk = dim_qk_heads // heads

        assert divisible_by(dim, heads)
        dim_v = dim // heads

        self.token_embed = Embedding(num_tokens, dim)

        self.rope = RotaryEmbedding(dim_qk // 2)
        self.depth = depth

        self.postnorm = LayerNormNoParams(dim)

        self.layer = LinearAttention(
            dim,
            heads = heads,
            dim_queries_keys = dim_qk,
            dim_values = dim_v
        )

        self.to_logits = Sequential(
            LayerNormNoParams(dim),
            LinearNoBias(dim, num_tokens)
        )

    def forward(
        self,
        ids,
        memories = None,
        return_memory = False
    ):
        tokens = self.token_embed(ids)

        seq_len, depth, device = tokens.shape[-2], self.depth, tokens.device

        # positions

        seq = torch.arange(seq_len, device = device)

        pos_emb = self.rope(seq)

        # memories

        memories = iter(default(memories, (None,) * depth))
        next_memories = []

        # layers

        for _ in range(depth):
            layer_out, layer_memory = self.layer(tokens, memories = next(memories, None), rotary_emb = pos_emb, return_memories = True)

            tokens = self.postnorm(tokens + layer_out)

            next_memories.append(layer_memory)

        # readout

        logits = self.to_logits(tokens)

        # return

        if not return_memory:
            return logits

        return logits, next_memories

# quick test

if __name__ == '__main__':

    model = BDH(
        dim = 512,
        num_tokens = 256
    )

    ids = torch.randint(0, 256, (2, 1024))

    logits = model(ids)

    assert logits.shape == (2, 1024, 256)
