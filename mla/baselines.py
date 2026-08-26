"""MHA, GQA and MQA. The things MLA is measured against.

All three cache K and V directly. The only lever they have is how many KV heads
to keep, so the cache is n_kv_heads * d_head * 2 elements per token per layer:

    MHA   n_kv = n_heads          full cache
    GQA   n_kv = n_heads / g      g query heads share one KV head
    MQA   n_kv = 1                every query head shares one KV head

Quality falls as n_kv falls, which is the trade MLA is trying to avoid.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .rope import apply_rope


class GroupedAttention(nn.Module):
    """One module covering MHA, GQA and MQA via `n_kv_heads`."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int | None = None):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        assert n_heads % self.n_kv_heads == 0
        self.d_head = d_model // n_heads
        self.repeat = n_heads // self.n_kv_heads

        self.wq = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.wk = nn.Linear(d_model, self.n_kv_heads * self.d_head, bias=False)
        self.wv = nn.Linear(d_model, self.n_kv_heads * self.d_head, bias=False)
        self.wo = nn.Linear(n_heads * self.d_head, d_model, bias=False)

    @property
    def variant(self) -> str:
        if self.n_kv_heads == self.n_heads:
            return "MHA"
        return "MQA" if self.n_kv_heads == 1 else f"GQA(g={self.repeat})"

    def cache_elements_per_token(self) -> int:
        """K and V, one vector each per KV head."""
        return 2 * self.n_kv_heads * self.d_head

    def forward(self, x, freqs, causal=True):
        b, s, _ = x.shape
        q = self.wq(x).view(b, s, self.n_heads, self.d_head).transpose(1, 2)
        k = self.wk(x).view(b, s, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.wv(x).view(b, s, self.n_kv_heads, self.d_head).transpose(1, 2)

        q, k = apply_rope(q, freqs), apply_rope(k, freqs)
        if self.repeat > 1:
            k = k.repeat_interleave(self.repeat, dim=1)
            v = v.repeat_interleave(self.repeat, dim=1)

        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.d_head)
        if causal:
            mask = torch.ones(s, s, dtype=torch.bool, device=x.device).triu(1)
            att = att.masked_fill(mask, float("-inf"))
        out = (att.softmax(-1) @ v).transpose(1, 2).reshape(b, s, -1)
        return self.wo(out)
