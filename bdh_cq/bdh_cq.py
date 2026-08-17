from __future__ import annotations

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Embedding, Linear, LayerNorm, Sequential, Parameter

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

class BDHBlock(Module):
    def __init__(
        self,
        dim,
        *,
        heads,
        dim_queries_keys,
        dim_values,
        qk_activation = nn.ReLU(),
        ff_activation = nn.ReLU(),
    ):
        super().__init__()
        dim_inner_qk = dim_queries_keys * heads
        dim_inner_values = dim_values * heads

        self.pre_norm = LayerNormNoParams(dim)

        self.to_qk = LinearNoBias(dim, dim_inner_qk)

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.qk_activation = qk_activation

        self.post_attn_norm = LayerNormNoParams(dim)

        # the feedforward part

        self.proj_up = Parameter(torch.randn(heads, dim, dim_queries_keys) * 0.02)

        self.ff_act = ff_activation

        self.merge_heads = Rearrange('b h n d -> b n (h d)')
        self.proj_out = LinearNoBias(dim_queries_keys * heads, dim)

        self.post_norm = LayerNormNoParams(dim)

    def forward(
        self,
        tokens,
        memories = None,
        rotary_emb = None,
        return_memories = False
    ):
        device = tokens.device

        normed_tokens = self.pre_norm(tokens)

        # queries and keys, relu activated

        sparse_input = self.qk_activation(self.to_qk(normed_tokens))

        # split heads

        q = k = ff_gates = self.split_heads(sparse_input)

        # the values are the normed tokens

        v = normed_tokens

        # relative positions

        if exists(rotary_emb):
            q, k = (apply_rotary_emb(rotary_emb, t) for t in (q, k))

        # linear attention, omitting attention to self

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        i, j = sim.shape[-2:]
        causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).tril(-1) # omit self, seen in Reformer shared qk attention years ago

        attn = sim.masked_fill(~causal_mask, 0.)

        # they directly aggregate on the tokens as the values, no projection

        agg = einsum('b h i j, b j d -> b h i d', attn, v)

        # past memories

        if exists(memories):
            retrieved = einsum('b h n d, b h d e -> b h n e', q, memories)
            agg = agg + retrieved

        # post attn norm

        attn_out = self.post_attn_norm(agg)

        # the interesting ff glu variant

        projected = einsum('b h n d, h d e -> b h n e', attn_out, self.proj_up)

        # they use the projected sparse input itself (q, k) as the gates

        projected = self.ff_act(projected * ff_gates)

        out = self.merge_heads(projected)

        out = self.proj_out(out)

        # maybe return memories

        if not return_memories:
            return out

        memories = einsum('b h n d, b n e -> b h d e', k, v)

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

        self.pre_norm = LayerNormNoParams(dim)

        self.block = BDHBlock(
            dim,
            heads = heads,
            dim_queries_keys = dim_qk,
            dim_values = dim_v
        )

        self.post_norm = LayerNormNoParams(dim)

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
            prev_memory = next(memories, None)

            normed = self.pre_norm(tokens)

            block_out, layer_memory = self.block(normed, memories = prev_memory, rotary_emb = pos_emb, return_memories = True)

            tokens = self.post_norm(tokens + block_out)

            next_memory = layer_memory + default(prev_memory, 0.)
            next_memories.append(next_memory)

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
