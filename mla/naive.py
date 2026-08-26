"""Multi-head latent attention, written the obvious way.

K and V are not stored. Instead each token gets one latent vector

    c_n = W^DKV x_n        (d_c dimensional, d_c << n_heads * d_head)

and K and V are reconstructed on use:

    k_n = W^UK c_n ,   v_n = W^UV c_n

The cache holds c_n only. With d_c = 512 against MHA's 32 heads * 128 dims * 2
tensors = 8192 elements, that is a 16x reduction before any quality is traded.

Decoupled RoPE, `use_decoupled_rope=True`: a small extra key of width d_rope is
computed straight from x, RoPE is applied only to it, and it is cached alongside
the latent. The compressed path stays rotation free so it can be absorbed. The
score is the sum of the two paths. Cache becomes d_c + d_rope per token.

This file materialises k and v on purpose so the maths is legible. `absorbed.py`
is the version that never does, and the two are asserted to agree.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .rope import apply_rope


class MLA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_c: int = 128,
                 d_rope: int = 16, use_decoupled_rope: bool = True,
                 d_head: int | None = None):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head or d_model // n_heads
        self.d_c = d_c
        self.d_rope = d_rope if use_decoupled_rope else 0
        self.use_decoupled_rope = use_decoupled_rope

        self.w_dkv = nn.Linear(d_model, d_c, bias=False)              # down projection
        self.w_uk = nn.Linear(d_c, n_heads * self.d_head, bias=False)  # up, keys
        self.w_uv = nn.Linear(d_c, n_heads * self.d_head, bias=False)  # up, values
        self.wq = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        if use_decoupled_rope:
            self.wk_rope = nn.Linear(d_model, d_rope, bias=False)       # shared across heads
            self.wq_rope = nn.Linear(d_model, n_heads * d_rope, bias=False)
        self.wo = nn.Linear(n_heads * self.d_head, d_model, bias=False)

    @property
    def variant(self) -> str:
        return "MLA" + ("" if self.use_decoupled_rope else " (no decoupled RoPE)")

    def cache_elements_per_token(self) -> int:
        """The latent, plus the shared rope key if decoupled RoPE is on."""
        return self.d_c + self.d_rope

    def forward(self, x, freqs, causal=True):
        b, s, _ = x.shape
        c = self.w_dkv(x)                                             # (b, s, d_c)
        k = self.w_uk(c).view(b, s, self.n_heads, self.d_head).transpose(1, 2)
        v = self.w_uv(c).view(b, s, self.n_heads, self.d_head).transpose(1, 2)
        q = self.wq(x).view(b, s, self.n_heads, self.d_head).transpose(1, 2)

        scale = math.sqrt(self.d_head + self.d_rope)
        att = q @ k.transpose(-1, -2)

        if self.use_decoupled_rope:
            qr = self.wq_rope(x).view(b, s, self.n_heads, self.d_rope).transpose(1, 2)
            kr = self.wk_rope(x).view(b, s, 1, self.d_rope).transpose(1, 2)
            qr = apply_rope(qr, freqs)
            kr = apply_rope(kr, freqs).expand(b, self.n_heads, s, self.d_rope)
            att = att + qr @ kr.transpose(-1, -2)
        else:
            # RoPE straight onto the reconstructed keys. Correct as attention,
            # but it puts a position dependent rotation between W^UK and the
            # query, so absorption is impossible. Kept to measure that claim.
            q = apply_rope(q, freqs)
            k = apply_rope(k, freqs)
            att = q @ k.transpose(-1, -2)

        att = att / scale
        if causal:
            mask = torch.ones(s, s, dtype=torch.bool, device=x.device).triu(1)
            att = att.masked_fill(mask, float("-inf"))
        out = (att.softmax(-1) @ v).transpose(1, 2).reshape(b, s, -1)
        return self.wo(out)
