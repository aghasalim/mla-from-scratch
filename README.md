# mla-from-scratch

[![ci](https://github.com/aghasalim/mla-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/aghasalim/mla-from-scratch/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![results](https://img.shields.io/badge/results-reproducible-1a9850.svg)](results/)

Multi-head latent attention, built from the DeepSeek V2 paper. Low rank KV
compression, decoupled RoPE, and the absorption trick that lets inference skip
reconstructing keys and values entirely.

Two results here, and they are of very different strength. The cache accounting
is exact and large. The quality comparison is a null, and I think the honest
reading is that my experiment is too small to see the effect rather than that
the effect is absent.

## The problem

A transformer serving long context spends most of its memory on the KV cache,
not on weights. At 128k context a 60 layer model with 32 heads of width 128
needs 120 GB of cache in fp16. That is the wall.

GQA and MQA answer by sharing KV heads. Fewer heads, less cache, some quality
loss. MLA answers differently: project K and V jointly down to one small latent
vector per token, cache only that, and fold the up projections into the
neighbouring weight matrices so inference never decompresses.

![KV cache per token and at 128k context](results/cache.png)

| variant | elements/token/layer | vs MHA | GB at 128k, fp16, 60 layers |
|---|---:|---:|---:|
| MHA | 8192 | 1.00x | 120.00 |
| GQA(g=2) | 4096 | 2.00x | 60.00 |
| GQA(g=4) | 2048 | 4.00x | 30.00 |
| GQA(g=8) | 1024 | 8.00x | 15.00 |
| MQA | 256 | 32.00x | 3.75 |
| **MLA** | **576** | **14.22x** | **8.44** |
| MLA without decoupled RoPE | 512 | 16.00x | 7.50 |

These are arithmetic, not measurements, so there is no error bar. MLA is
`d_c + d_rope = 512 + 64`.

The last row is the price of decoupled RoPE: 64 extra elements per token, 0.94 GB
at 128k context, which buys the ability to absorb the up projections at all. It
is not a variant you would deploy, since without decoupled RoPE you have to
reconstruct every key at every step and the whole point is lost. It is in the
table because the cost of the fix is worth knowing.

MQA is smaller still, which is the point of the quality question below: MQA
already gives you 32x, so MLA only earns its complexity if it holds quality
better at the same budget.

![KV cache filling as the context grows, full cache against the latent cache](results/cache-growth.gif)

The same arithmetic as the table, watched as the context fills. Shape is held
fixed at the DeepSeek V2 numbers (32 heads of width 128, d_c=512, d_rope=64,
60 layers, fp16) and only the sequence length moves. The dashed line is one
80 GB H100.

## Absorption, and why RoPE breaks it

MLA caches `c_n = W_DKV x_n` and reconstructs `k_n = W_UK c_n`. The content
score is

    q_m . k_n = (W_Q x_m) . (W_UK c_n) = x_m^T W_Q^T W_UK c_n

so `W_UK` can be folded into `W_Q` once, and `k_n` never has to exist. The same
works on the value side: attending over the latents and up projecting once at
the end lets `W_UV` fold into `W_O`. Both are just associativity.

RoPE ruins it. Rotary embeddings put a position dependent rotation between the
two matrices,

    q_m^T R_(n-m) W_UK c_n

and `R` depends on the token index, so there is no fixed matrix to pre multiply.
Decoupled RoPE is the fix: carry a small set of extra channels that are not
compressed, apply RoPE only there, and leave the compressed path rotation free.
The score becomes a sum of an absorbed content term and a small positional term.

This repo implements both, and `AbsorbedMLA` refuses to construct itself from a
model that does not use decoupled RoPE rather than silently folding something
wrong.

**The check that matters:** absorbed inference is asserted to be numerically
identical to the naive form, across three latent widths and three sequence
lengths including 1. Measured agreement is 6e-07 max absolute difference. If the
folding were wrong both versions would still produce plausible attention output,
so this test is the only thing that can catch it.

## Quality at matched budget, and why this part is weak

Six variants, same depth, width, data, batch, steps and seed, only the attention
module changing. The six do not separate. All 18 runs land between 4.50 and 4.60
validation perplexity: the spread between variant medians is 0.081, the mean
spread between seeds of one variant is 0.058, and a ratio of 1.41 is not enough
to call anything. At the matched budget of 256 elements per token MQA scores
4.538 and MLA 4.563, a gap of 0.025 while MLA's own three seeds span 0.080. MHA
carries six times the cache of MQA and still cannot beat it, which is the sign
that the experiment is the limit here and not the method.

![quality against cache budget](results/quality.png)
![validation curves](results/curves.png)

Full detail in [notes/METHODS.md](notes/METHODS.md#quality-at-matched-budget-and-why-this-part-is-weak).

## What I got wrong

**I assumed the quality experiment would show something.** I built the whole
matched budget comparison before checking whether the setup had the resolution
to detect the effect. Estimating seed noise first would have cost about ten
minutes and shown that 0.058 of noise buries the 0.025 gap I was looking for, so
the 32 minute sweep went on a table that cannot answer its own question. The
other one is that I nearly shipped `AbsorbedMLA` with no equality test against
the naive path. It produced sensible looking attention from the first run, and
the einsum index order in that first version was wrong.

Full detail in [notes/METHODS.md](notes/METHODS.md#what-i-got-wrong).

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

```bash
python -m pytest tests/ -q
```

```bash
python -m bench.cache
```

```bash
python -m train.run --steps 1500 --seeds 0 1 2
```

```bash
python -m bench.figures
```

The training sweep takes about 32 minutes on an M4 CPU. `bench.cache` is instant
because it is arithmetic. Figures read the committed CSVs and never re measure.

## Layout

```
mla/rope.py        RoPE, and the explanation of why it breaks absorption
mla/baselines.py   MHA, GQA and MQA behind one module
mla/naive.py       MLA with K and V materialised, written to be legible
mla/absorbed.py    MLA with the up projections folded away, asserted equal to naive
bench/cache.py     exact cache accounting
train/             the small char LM and the matched budget sweep
tests/             26 tests
```

## Sources

- **DeepSeek-AI. DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model. 2024.** [arXiv:2405.04434](https://arxiv.org/abs/2405.04434) MLA, decoupled RoPE, and the absorption argument. Section 2.1 is the load bearing part.
- **Su, Lu, Pan, Murtadha, Wen, Liu. RoFormer: Enhanced Transformer with Rotary Position Embedding. 2021.** [arXiv:2104.09864](https://arxiv.org/abs/2104.09864) RoPE itself, and the relative position property tested here.
- **Shazeer. Fast Transformer Decoding: One Write-Head is All You Need. 2019.** [arXiv:1911.02150](https://arxiv.org/abs/1911.02150) MQA.
- **Ainslie, Lee-Thorp, de Jong, Zemlyanskiy, Lebrón, Sanghai. GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints. EMNLP 2023.** [arXiv:2305.13245](https://arxiv.org/abs/2305.13245) GQA, and the uptraining recipe.
- **Pope, Douglas, Chowdhery et al. Efficiently Scaling Transformer Inference. MLSys 2023.** [arXiv:2211.05102](https://arxiv.org/abs/2211.05102) Why the KV cache is the serving bottleneck in the first place.
- Corpus is tiny Shakespeare, from Karpathy's char-rnn.

Related: [flash-attention-from-scratch](https://github.com/aghasalim/flash-attention-from-scratch)
attacks the same memory wall from the kernel side rather than the architecture side.

## Methodology

The rules this follows are in [`METHODOLOGY.md`](METHODOLOGY.md). Rule 12, report variance not
just the point estimate, is the reason the quality section says what it says.

## Author

Aghasalim Mustafazada, third year AI student at Howest, Belgium.

<p align="center">
  <a href="https://github.com/aghasalim">
    <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="github"></a>
  <a href="https://www.kaggle.com/aghasalimmustafazada">
    <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white" alt="kaggle"></a>
  <a href="https://linkedin.com/in/mustafazada">
    <img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="linkedin"></a>
  <a href="https://orcid.org/0009-0001-8746-4582">
    <img src="https://img.shields.io/badge/ORCID-A6CE39?style=for-the-badge&logo=orcid&logoColor=white" alt="orcid"></a>
</p>

## License

MIT, see [LICENSE](LICENSE).
