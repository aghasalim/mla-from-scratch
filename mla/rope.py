"""Rotary position embeddings, and why they break MLA's absorption trick.

RoPE rotates each 2D pair of a head's channels by an angle proportional to the
token's position:

    q_m -> R_m q_m ,   k_n -> R_n k_n ,   so  q_m . k_n  ->  q_m^T R_{n-m} k_n

The attention score depends only on the relative offset n - m, which is the
whole point of RoPE.

Now the MLA problem. MLA caches a low rank latent c_n and reconstructs
k_n = W^UK c_n at use time. Absorption is the observation that

    q_m^T k_n = q_m^T W^UK c_n = (W^UK{}^T q_m)^T c_n

so you can fold W^UK into the query projection once and never materialise k_n at
all. That works because the two matrices are adjacent.

RoPE puts R_{n-m} between them:

    q_m^T R_{n-m} W^UK c_n

and R depends on n, which is the token index. There is no fixed matrix to
pre-multiply into W^Q, because a different rotation is needed for every position
pair. The absorption is destroyed.

Decoupled RoPE is the fix: carry a small extra set of channels that are *not*
compressed, apply RoPE only there, and leave the compressed part rotation free.
The score becomes the sum of an absorbed content term and a small positional
term, and only the small term needs its keys cached in full.
"""
from __future__ import annotations

import torch


def rope_frequencies(dim: int, end: int, theta: float = 10_000.0) -> torch.Tensor:
    """Complex rotation factors of shape (end, dim//2)."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
    t = torch.arange(end).float()
    return torch.polar(torch.ones_like(torch.outer(t, freqs)), torch.outer(t, freqs))


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Rotate x, shape (batch, heads, seq, dim), using freqs (seq, dim//2)."""
    b, h, s, d = x.shape
    xc = torch.view_as_complex(x.float().reshape(b, h, s, d // 2, 2))
    out = torch.view_as_real(xc * freqs[:s].view(1, 1, s, d // 2)).reshape(b, h, s, d)
    return out.type_as(x)


def relative_score_is_position_invariant(dim=64, seq=16, theta=10_000.0) -> float:
    """Check RoPE's defining property, and return the largest deviation.

    Two query/key pairs with the same offset must score identically regardless
    of where they sit. If this is not ~0 the rotation is wrong.
    """
    torch.manual_seed(0)
    freqs = rope_frequencies(dim, seq, theta)
    q = torch.randn(1, 1, seq, dim)
    k = torch.randn(1, 1, seq, dim)
    qr, kr = apply_rope(q, freqs), apply_rope(k, freqs)
    worst = 0.0
    for offset in range(1, 5):
        scores = [(qr[0, 0, m + offset] @ kr[0, 0, m]).item() for m in range(seq - offset)]
        # same offset, different absolute positions: only true when q,k are the
        # same vector at every position, so compare against a constant-input run
        base = torch.randn(dim)
        qc = base.view(1, 1, 1, dim).expand(1, 1, seq, dim).contiguous()
        kc = base.view(1, 1, 1, dim).expand(1, 1, seq, dim).contiguous()
        qcr, kcr = apply_rope(qc, freqs), apply_rope(kc, freqs)
        s2 = [(qcr[0, 0, m + offset] @ kcr[0, 0, m]).item() for m in range(seq - offset)]
        worst = max(worst, max(s2) - min(s2))
        del scores
    return worst
