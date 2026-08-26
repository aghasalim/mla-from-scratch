"""A small character level transformer with a pluggable attention block.

Deliberately tiny: the claim being tested is about the KV cache, so every
variant gets the same depth, width, data and step count, and only the attention
module changes. Anything else would confound the comparison.
"""
from __future__ import annotations

import torch
from torch import nn

from mla.absorbed import AbsorbedMLA
from mla.baselines import GroupedAttention
from mla.naive import MLA
from mla.rope import rope_frequencies


class Block(nn.Module):
    def __init__(self, attn, d_model, mult=4):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = attn
        self.mlp = nn.Sequential(nn.Linear(d_model, mult * d_model), nn.GELU(),
                                 nn.Linear(mult * d_model, d_model))

    def forward(self, x, freqs):
        x = x + self.attn(self.n1(x), freqs)
        return x + self.mlp(self.n2(x))


class CharLM(nn.Module):
    def __init__(self, vocab, d_model=192, n_layers=4, n_heads=6, seq=256,
                 variant="MHA", d_c=48, d_rope=16, decoupled=True):
        super().__init__()
        self.seq, self.variant = seq, variant
        d_head = d_model // n_heads

        def make():
            if variant == "MHA":
                return GroupedAttention(d_model, n_heads, n_heads)
            if variant.startswith("GQA"):
                return GroupedAttention(d_model, n_heads, n_heads // int(variant.split("=")[1].rstrip(")")))
            if variant == "MQA":
                return GroupedAttention(d_model, n_heads, 1)
            if variant == "MLA":
                return MLA(d_model, n_heads, d_c=d_c, d_rope=d_rope, use_decoupled_rope=decoupled)
            raise ValueError(variant)

        self.emb = nn.Embedding(vocab, d_model)
        self.blocks = nn.ModuleList([Block(make(), d_model) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)

        rope_dim = d_rope if (variant == "MLA" and decoupled) else d_head
        self.register_buffer("freqs", torch.view_as_real(rope_frequencies(rope_dim, seq)),
                             persistent=False)

    def _freqs(self):
        return torch.view_as_complex(self.freqs)

    def cache_elements_per_token(self) -> int:
        """Summed over layers. This is the number the whole repo is about."""
        return sum(b.attn.cache_elements_per_token() for b in self.blocks)

    def forward(self, idx, targets=None):
        x = self.emb(idx)
        f = self._freqs()
        for b in self.blocks:
            x = b(x, f)
        logits = self.head(self.norm(x))
        if targets is None:
            return logits, None
        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def to_absorbed(self):
        """Swap every MLA block for its absorbed form. Inference only."""
        for b in self.blocks:
            if isinstance(b.attn, MLA):
                b.attn = AbsorbedMLA(b.attn)
        return self
