"""Absorbed MLA inference: never reconstruct K or V at all.

Two identities, both just associativity.

Keys. The content score is

    q_m . k_n = (W^Q x_m) . (W^UK c_n) = x_m^T (W^Q)^T W^UK c_n

so with  W^Q_abs = (W^UK)^T W^Q  computed once, the score is (W^Q_abs x_m) . c_n
and k_n never exists. The query now lives in the latent space, dimension d_c
instead of d_head.

Values. The output is

    o_m = sum_n a_mn v_n = sum_n a_mn W^UV c_n = W^UV (sum_n a_mn c_n)

so attend over the latents and up-project once at the end. Fold W^UV into W^O
and the value path never materialises either.

Consequence: at decode time the cache holds only c (plus the small decoupled
rope key), and the per-step work is a d_c-dimensional dot product per head
instead of a d_head one against a reconstructed key. That is the entire point of
MLA, and it is only possible because no position dependent rotation sits between
the query and W^UK. See rope.py.

This module is checked against naive.py to within float tolerance. If those two
disagree the absorption is wrong, and it is the only test that can tell.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .naive import MLA
from .rope import apply_rope


class AbsorbedMLA(nn.Module):
    """Inference-time view of a trained `MLA`. Shares its weights."""

    def __init__(self, src: MLA):
        super().__init__()
        if not src.use_decoupled_rope:
            raise ValueError(
                "absorption requires decoupled RoPE: a position dependent rotation "
                "between the query and W^UK cannot be folded into a fixed matrix. "
                "See mla/rope.py."
            )
        self.src = src
        self.n_heads, self.d_head = src.n_heads, src.d_head
        self.d_c, self.d_rope = src.d_c, src.d_rope

        with torch.no_grad():
            # W^Q: (n*d_head, d_model), W^UK: (n*d_head, d_c)
            wq = src.wq.weight.view(self.n_heads, self.d_head, -1)
            wuk = src.w_uk.weight.view(self.n_heads, self.d_head, self.d_c)
            # per head: (d_c, d_head) @ (d_head, d_model) -> (d_c, d_model)
            self.register_buffer("wq_abs", torch.einsum("hdc,hdm->hcm", wuk, wq))
            wuv = src.w_uv.weight.view(self.n_heads, self.d_head, self.d_c)
            wo = src.wo.weight.view(-1, self.n_heads, self.d_head)
            # (d_model, n, d_head) x (n, d_head, d_c) -> (d_model, n, d_c)
            self.register_buffer("wo_abs", torch.einsum("mhd,hdc->mhc", wo, wuv))

    def cache_elements_per_token(self) -> int:
        return self.src.cache_elements_per_token()

    def forward(self, x, freqs, causal=True):
        b, s, _ = x.shape
        src = self.src
        c = src.w_dkv(x)                                              # (b, s, d_c)

        q_lat = torch.einsum("bsm,hcm->bhsc", x, self.wq_abs)         # query in latent space
        att = torch.einsum("bhsc,btc->bhst", q_lat, c)                # against cached c

        qr = src.wq_rope(x).view(b, s, self.n_heads, self.d_rope).transpose(1, 2)
        kr = src.wk_rope(x).view(b, s, 1, self.d_rope).transpose(1, 2)
        qr = apply_rope(qr, freqs)
        kr = apply_rope(kr, freqs)
        att = att + torch.einsum("bhsr,bktr->bhst", qr, kr)

        att = att / math.sqrt(self.d_head + self.d_rope)
        if causal:
            att = att.masked_fill(torch.ones(s, s, dtype=torch.bool, device=x.device).triu(1),
                                  float("-inf"))
        ctx = torch.einsum("bhst,btc->bhsc", att.softmax(-1), c)      # still in latent space
        return torch.einsum("bhsc,mhc->bsm", ctx, self.wo_abs)        # one projection out
