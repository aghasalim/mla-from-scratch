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
lengths including 1. Measured agreement is 8.3e-07 max absolute difference, the
worst case over that grid. If the folding were wrong both versions would still
produce plausible attention output, so this test is the only thing that can
catch it.

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

## Everything here is computed twice

Every number here came out of exactly one implementation. The cache table came
out of `bench/cache.py`, the perplexity statistics out of one pass over
`results/quality.csv` that I did by hand, and the claim that absorbed MLA equals
the naive form out of the same PyTorch that produces both sides of it. If any of
those were wrong, nothing downstream would notice, because everything downstream
reads the same output. The tests checked that the code ran, not that it was
right.

So the published numbers are recomputed by five independent implementations in
five languages, and CI fails if any two disagree. For the kernel that means
golden vectors: `verify/export_golden.py` writes the weights, the input and both
output tensors to
[`verify/golden/mla_golden.txt`](verify/golden/mla_golden.txt), and the C and the
Rust have to reproduce the outputs from the weights alone, with no access to the
Python at all.

| implementation | what it recomputes | measured agreement |
| --- | --- | --- |
| [`verify/cache.sql`](verify/cache.sql) | all 12 rows of `results/cache.csv`, from n_heads, d_head, d_c, d_rope and layers, in SQLite | exact to 1e-10, 12 rows of 12 |
| [`verify/mla_kernel.c`](verify/mla_kernel.c) | the whole MLA forward pass, naive and absorbed, from the golden weights, tensors resolved by name | 1.103e-07 naive, 1.517e-07 absorbed |
| [`verify/gocheck`](verify/gocheck) | structure of every results CSV, val_ppl against exp(val_loss), cache/token from the shape in `train-meta.json` | val_ppl to 8.9e-16, cache/token exact for all 18 runs |
| [`verify/verify.R`](verify/verify.R) | the nine perplexity statistics that exist only in this prose, plus a permutation test | all nine inside the rounding of the figure quoted |
| [`verify/absorption`](verify/absorption) | the golden vectors again, plus the absorption identity over 50,000 random shapes | 1.103e-07 and 1.517e-07, the same as the C |

Run them all with [`./verify/verify.sh`](verify/verify.sh), which prints
`5 passed, 0 failed, 0 skipped` here. Each check is skipped with a message if its
toolchain is missing, so a partial install still runs the rest.

**It found a wrong number.** This README said the absorbed and naive paths agree
to 6e-07. Recomputed over the same grid the test suite uses, three latent widths
by three sequence lengths, the worst case is 8.345e-07 on torch 2.13.0. The
figure above is now the measured one, and the Go check compares what the README
quotes against the golden file, so it cannot drift again quietly.

**The C and the Rust agree to every digit they print.** Both work in double
precision against float32 golden vectors, so 1.103e-07 and 1.517e-07 are the
width of float32 on the PyTorch side rather than error in either kernel. Inside
double precision the absorption identity is exact: the C measures its own naive
path against its own absorbed path at 3.886e-16.

**The Rust asks the question the test suite could not afford.** `tests/` asserts
naive equals absorbed on nine fixed shapes with one seed. Absorption is algebra,
so it should hold for every shape, and a fault that only appears at n_heads=1 or
seq=1 would pass those nine. 50,000 random draws over n_heads 1 to 4, d_head 2 to
8, d_c 2 to 13, d_rope 2 to 8 and seq 1 to 8, with the crate's own xorshift
generator and no dependencies: worst disagreement 7.105e-15.

**The R runs the test the quality section argues for in words.** If the variant
labels carry no information, how often does a random relabelling of the same 18
runs produce a spread between medians as large as the observed 0.081? Over 20,000
relabellings the median spread is 0.062 and p = 0.07. That is the quantitative
form of "nothing is separated", and it was missing.

**The harness is checked too.** CI moves one GB figure in `results/cache.csv`,
requires the SQL to reject it, restores it, and requires a pass; then moves one
weight of W^UK in the golden file by 1e-3, which the C and the Rust both reject
at 1.894e-05 against their 1e-05 tolerance. Locally I also confirmed the rest of
the coverage: perturbing one `val_ppl` by 0.01 is caught by Go, which recomputes
it from `val_loss`, and by R, which sees the seed spread move; a ragged row, a
NaN, and a wrong `cache_per_token` are caught by Go; a perturbed `out_absorbed`
in the golden file is caught by the C and the Rust while the naive comparison
still passes.

Five languages and not more. This repository publishes a cache table, a set of
perplexity statistics and one kernel, and a sixth implementation of something
already computed three times would be decoration rather than evidence.

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
verify/            the same numbers recomputed in SQL, C, Go, R and Rust
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
