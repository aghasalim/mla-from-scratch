# mla-from-scratch

Multi-head Latent Attention (DeepSeek-V2) implemented from the paper: low-rank joint KV compression, decoupled RoPE, and the matrix-absorption trick that lets you attend in latent space without ever materializing K and V.

> **Status: scaffold. Nothing here is built or measured yet.**
> This repo currently holds the project specification, the shared agent conventions,
> and an empty logbook. Every number in the tables below is a `TODO` because no
> experiment has been run. The `prompts/` task specs referenced in the wave table
> are not written yet either.
>
> Nothing in this repo is estimated or taken from a paper. When a table has a number
> in it, that number came from a run in `results/`.

---

## Why

Repo 01 ended at Flash-Decoding, which makes streaming a huge KV cache as fast as memory allows. MLA asks the better question: why is the cache that big in the first place?

GQA and MQA answer by sharing KV heads — fewer heads, less cache, some quality loss. MLA answers by projecting K and V jointly down to a single low-rank latent vector per token, caching only that, and folding the up-projections into the neighbouring weight matrices so inference never decompresses. Same wall, a different floor of the building.

The part that makes it a real project rather than a weekend one is that **RoPE breaks the absorption**, and the fix — decoupled RoPE — is not obvious and is the thing most reimplementations get wrong or skip.

## Hardware

- **GPU:** `TODO — python -m scripts.env`
- Everything here fits in 8GB. This is a memory-*accounting* project, not a memory-hungry one.

## Results

Per-token, per-layer KV cache in elements, `n_h=32, d_h=128, d_c=512, d_h^R=64`:

| Variant | Cache/token/layer | vs MHA | PPL @ matched budget |
|---|---:|---:|---:|
| MHA | TODO | 1.00× | TODO |
| GQA (g=8) | TODO | TODO | TODO |
| MQA | TODO | TODO | TODO |
| **MLA** | TODO | TODO | TODO |

Fill from `results/`. The interesting column is the last one — cache reduction is arithmetic, quality at matched budget is the claim.

## Waves

```
00 bootstrap + cache accounting          (serial)
   ├─ 01 the math: absorption + RoPE     ┐
   └─ 02 MHA/GQA/MQA baselines           ┘ parallel
        └─ 03 naive MLA (materialized)   (serial — everything below extends it)
             ├─ 04 decoupled RoPE        ┐
             └─ 05 absorbed inference    ┘ parallel
                  └─ 06 quality ablation + writeup
```

| Task | OWNS | READS |
|---|---|---|
| 00 | `scripts/`, `Makefile`, `mla/__init__.py`, `HARDWARE.md` | — |
| 01 | `notes/00-absorption.md`, `mla/ref/math.py` | `scripts/` |
| 02 | `mla/baselines/`, `bench/cache.py` | `scripts/` |
| 03 | `mla/naive.py`, `tests/` | `mla/ref/`, `mla/baselines/` |
| 04 | `mla/rope.py` | `mla/naive.py`, `notes/00-*` |
| 05 | `mla/absorbed.py`, `mla/triton/` | `mla/naive.py` |
| 06 | `train/`, `results/`, `notes/paper.md`, `README.md` | everything |

See [`CONVENTIONS.md`](CONVENTIONS.md) for the rules every task assumes.

## Author

Aghasalim Mustafazada — third-year AI student at Howest, Belgium.

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

MIT — see [LICENSE](LICENSE).
