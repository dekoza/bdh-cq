from __future__ import annotations
from collections import namedtuple

import torch
from torch import nn, einsum, is_tensor
from torch.nn import Module, Embedding, Linear, LayerNorm, Sequential, Parameter

from einops.layers.torch import Rearrange

from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb

# constants

Memory = namedtuple('Memory', ('tokens_seen', 'embeds', 'fast_weight_memories'))

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def first(v):
    return v[0]

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
        qk_activation = nn.ReLU(),
        ff_activation = nn.ReLU(),
    ):
        super().__init__()
        dim_inner_qk = dim_queries_keys * heads

        self.to_qk = LinearNoBias(dim, dim_inner_qk)

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.qk_activation = qk_activation

        self.post_attn_norm = LayerNormNoParams(dim)

        self.post_ff_norm = LayerNormNoParams(dim)

        # the feedforward part

        self.proj_up = Parameter(torch.randn(heads, dim, dim_queries_keys) * 0.02)

        self.ff_act = ff_activation

        self.merge_heads = Rearrange('b h n d -> b n (h d)')
        self.proj_out = LinearNoBias(dim_queries_keys * heads, dim)

    def forward(
        self,
        tokens,
        memories = None,
        rotary_emb = None,
        return_memories = False
    ):
        device = tokens.device

        # queries and keys, relu activated

        sparse_input = self.qk_activation(self.to_qk(tokens))

        # split heads

        q = k = ff_gates = self.split_heads(sparse_input)

        # the values are the tokens

        v = tokens

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

        out = self.post_ff_norm(out)

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

        self.token_embed = Embedding(num_tokens, dim)

        self.rope = RotaryEmbedding(dim_qk // 2)
        self.depth = depth

        self.post_embed_norm = LayerNormNoParams(dim)

        self.block = BDHBlock(
            dim,
            heads = heads,
            dim_queries_keys = dim_qk
        )

        self.post_norm = LayerNormNoParams(dim)

        self.to_logits = LinearNoBias(dim, num_tokens)

    def forward(
        self,
        tokens_or_ids,
        memories = None,
        return_memory = False,
        return_logits = True,
        update_memory = True
    ):

        # the input can be tokens, from last forward, for recurrent latent reasoning

        tokens = tokens_or_ids if tokens_or_ids.is_floating_point() else None

        # usual token embed if the input is not floating point

        if not exists(tokens):

            tokens = self.token_embed(tokens_or_ids)

            tokens = self.post_embed_norm(tokens)

        # variables

        seq_len, depth, device = tokens.shape[-2], self.depth, tokens.device

        # destruct memories

        tokens_seen = 0

        if exists(memories):
            tokens_seen, _, memories = memories

        # positions

        seq = torch.arange(seq_len, device = device) + tokens_seen

        pos_emb = self.rope(seq)

        # memories

        memories = iter(default(memories, (None,) * depth))
        next_memories = []

        # layers

        for _ in range(depth):
            prev_memory = next(memories, None)

            block_out, layer_memory = self.block(tokens, memories = prev_memory, rotary_emb = pos_emb, return_memories = True)

            tokens = self.post_norm(tokens + block_out)

            # update the memory, but allow for it to be controlled with `update_memory` kwarg, section 3.3 suggests they kept the past memory constant during the latent recurrent iterations

            next_memory = layer_memory + default(prev_memory, 0.) if update_memory else prev_memory
            next_memories.append(next_memory)

        # readout

        logits = self.to_logits(tokens) if return_logits else None

        # return

        if not return_memory:
            return logits

        next_tokens_seen = tokens_seen + seq_len

        return logits, Memory(next_tokens_seen, tokens, next_memories)

# reasoning wrapper for interleaved parallel token ingestion and recurrent latent reasoning

class BDHReasoningWrapper(Module):
    def __init__(self, bdh: BDH):
        super().__init__()
        self.bdh = bdh

    def forward(
        self,
        *args,
        memories: Memory | None = None,
        return_memory = False,
        update_memory = False
    ):
        # allow for passing a single list or tuple of inputs

        if len(args) == 1 and isinstance(first(args), (list, tuple)):
            args = first(args)

        # loop through parallel tokens and latent reasoning steps

        logits = None

        for item in args:

            # latent reasoning step

            if isinstance(item, int):
                assert exists(memories), 'must ingest tokens before latent reasoning'

                for _ in range(item):
                    latent = memories.embeds[..., -1:, :]
                    _, memories = self.bdh(latent, memories = memories, return_memory = True, return_logits = False, update_memory = update_memory)

            # parallel tokens

            elif is_tensor(item):
                logits, memories = self.bdh(item, memories = memories, return_memory = True)

        # return

        return (logits, memories) if return_memory else logits

# quick test

if __name__ == '__main__':

    model = BDH(
        dim = 512,
        num_tokens = 256
    )

    ids = torch.randint(0, 256, (2, 1024))

    logits = model(ids)

    assert logits.shape == (2, 1024, 256)
