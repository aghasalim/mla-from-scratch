"""Tests. The load bearing one is that absorbed MLA equals naive MLA."""
import math

import pytest
import torch

from mla.absorbed import AbsorbedMLA
from mla.baselines import GroupedAttention
from mla.naive import MLA
from mla.rope import apply_rope, rope_frequencies

D_MODEL, N_HEADS, SEQ, D_ROPE = 256, 8, 48, 16


def _freqs(dim=D_ROPE, seq=SEQ):
    return rope_frequencies(dim, seq)


# --- rope -------------------------------------------------------------------
def test_rope_preserves_norm():
    x = torch.randn(2, 4, SEQ, 64)
    assert torch.allclose(apply_rope(x, _freqs(64)).norm(dim=-1), x.norm(dim=-1), atol=1e-4)


def test_rope_is_identity_at_position_zero():
    x = torch.randn(1, 1, 1, 64)
    assert torch.allclose(apply_rope(x, _freqs(64, 1)), x, atol=1e-5)


def test_rope_score_depends_only_on_relative_offset():
    """RoPE's defining property. If this fails everything downstream is noise."""
    f = _freqs(64, 32)
    base = torch.randn(64)
    x = base.view(1, 1, 1, 64).expand(1, 1, 32, 64).contiguous()
    r = apply_rope(x, f)
    for offset in (1, 3, 7):
        scores = [(r[0, 0, m + offset] @ r[0, 0, m]).item() for m in range(32 - offset)]
        assert max(scores) - min(scores) < 1e-3, f"offset {offset} not position invariant"


# --- cache accounting -------------------------------------------------------
@pytest.mark.parametrize("n_kv,expect", [(8, 2 * 8 * 32), (4, 2 * 4 * 32), (1, 2 * 1 * 32)])
def test_grouped_cache_accounting(n_kv, expect):
    m = GroupedAttention(D_MODEL, 8, n_kv)
    assert m.cache_elements_per_token() == expect


def test_mla_cache_is_latent_plus_rope():
    m = MLA(D_MODEL, N_HEADS, d_c=96, d_rope=16)
    assert m.cache_elements_per_token() == 96 + 16


def test_mla_cache_beats_mha_substantially():
    mha = GroupedAttention(D_MODEL, N_HEADS, N_HEADS)
    mla = MLA(D_MODEL, N_HEADS, d_c=96, d_rope=16)
    assert mla.cache_elements_per_token() < mha.cache_elements_per_token() / 4


# --- shapes and causality ---------------------------------------------------
@pytest.mark.parametrize("build", [
    lambda: GroupedAttention(D_MODEL, N_HEADS, N_HEADS),
    lambda: GroupedAttention(D_MODEL, N_HEADS, 2),
    lambda: GroupedAttention(D_MODEL, N_HEADS, 1),
    lambda: MLA(D_MODEL, N_HEADS, d_c=96, d_rope=D_ROPE),
])
def test_output_shape(build):
    m = build()
    dim = D_ROPE if isinstance(m, MLA) else D_MODEL // N_HEADS
    x = torch.randn(2, SEQ, D_MODEL)
    assert m(x, _freqs(dim)).shape == (2, SEQ, D_MODEL)


@pytest.mark.parametrize("build,dim", [
    (lambda: GroupedAttention(D_MODEL, N_HEADS, N_HEADS), D_MODEL // N_HEADS),
    (lambda: MLA(D_MODEL, N_HEADS, d_c=96, d_rope=D_ROPE), D_ROPE),
])
def test_causal_masking_holds(build, dim):
    """Output at position i must not change when tokens after i change.

    A strong causality check that needs no reference implementation, and the one
    that catches an off by one in the mask.
    """
    torch.manual_seed(0)
    m = build().eval()
    f = _freqs(dim)
    x = torch.randn(1, SEQ, D_MODEL)
    y = x.clone()
    y[:, SEQ // 2:] = torch.randn(1, SEQ - SEQ // 2, D_MODEL)
    with torch.no_grad():
        a, b = m(x, f), m(y, f)
    assert torch.allclose(a[:, :SEQ // 2], b[:, :SEQ // 2], atol=1e-5)


# --- absorption: the important one ------------------------------------------
@pytest.mark.parametrize("d_c", [48, 96, 192])
@pytest.mark.parametrize("seq", [1, 7, 48])
def test_absorbed_matches_naive(d_c, seq):
    """Absorbed inference must be numerically identical to the naive form.

    Absorption folds W^UK into W^Q and W^UV into W^O so K and V are never
    materialised. It is pure associativity, so any disagreement means the
    folding is wrong. This is the only test that can catch it: both versions
    produce plausible attention output on their own.
    """
    torch.manual_seed(0)
    m = MLA(D_MODEL, N_HEADS, d_c=d_c, d_rope=D_ROPE).eval()
    a = AbsorbedMLA(m).eval()
    x = torch.randn(3, seq, D_MODEL)
    f = _freqs(D_ROPE, max(seq, 1))
    with torch.no_grad():
        assert torch.allclose(m(x, f), a(x, f), atol=1e-4)


def test_absorption_refused_without_decoupled_rope():
    """The claim in rope.py, enforced.

    With RoPE applied to reconstructed keys, a position dependent rotation sits
    between the query and W^UK, so there is no fixed matrix to fold. Refusing
    loudly is better than silently absorbing something wrong.
    """
    m = MLA(D_MODEL, N_HEADS, d_c=96, use_decoupled_rope=False)
    with pytest.raises(ValueError, match="decoupled RoPE"):
        AbsorbedMLA(m)


def test_absorbed_query_lives_in_latent_space():
    """Shape check with a point: the absorbed query is d_c wide, not d_head."""
    m = MLA(D_MODEL, N_HEADS, d_c=96, d_rope=D_ROPE)
    a = AbsorbedMLA(m)
    assert a.wq_abs.shape == (N_HEADS, 96, D_MODEL)


# --- softmax sanity ---------------------------------------------------------
def test_attention_rows_sum_to_one():
    torch.manual_seed(0)
    m = MLA(D_MODEL, N_HEADS, d_c=96, d_rope=D_ROPE).eval()
    x = torch.randn(1, SEQ, D_MODEL)
    with torch.no_grad():
        c = m.w_dkv(x)
        k = m.w_uk(c).view(1, SEQ, N_HEADS, m.d_head).transpose(1, 2)
        q = m.wq(x).view(1, SEQ, N_HEADS, m.d_head).transpose(1, 2)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(m.d_head + m.d_rope)
        att = att.masked_fill(torch.ones(SEQ, SEQ, dtype=torch.bool).triu(1), float("-inf"))
        assert torch.allclose(att.softmax(-1).sum(-1), torch.ones(1, N_HEADS, SEQ), atol=1e-5)
