# The training run behind quality.csv

`results/quality.csv` and `results/curves.csv` are the output of one run of
`train/run.py`. This is what that run was, each value traced to the line
that set it. The command line carried two of them. The rest are defaults
or constants in the code, and they are listed here rather than moved,
because moving them would make it a different experiment.

## Run

    python -m train.run --steps 1500 --seeds 0 1 2

from `README.md`, Reproducing this. `--seeds` overrides the default of
`[0, 1]` (`train/run.py:98`); `--steps 1500` restates the default
(`run.py:89`). `results/train-meta.json` is written from the parsed
arguments plus the wall clock (`run.py:122`) and holds seeds 0 1 2, steps
1500, wall_clock_s 1904.31, torch 2.13.0, device cpu. The logbook entry of
2026-08-26 gives the same 1904 s on an M4 CPU, and `results/` was first
committed that day.

## The recipe every variant got

| | value | where |
|---|---|---|
| corpus | `data/input.txt`, first 90% train, last 10% validation | `train/data.py:12`, `:16` |
| context | 128 tokens | `run.py:90` |
| batch | 24 sequences | `run.py:91` |
| model | d_model 192, 4 layers, 6 heads, so d_head 32; MLP 4x wide | `run.py:92` to `:94`, `train/model.py:36`, `:19` |
| optimiser | AdamW, lr 3e-3, weight decay 0.1 | `run.py:95`, `:57` |
| schedule | OneCycle over the 1500 steps, 10% warmup | `run.py:58` |
| clipping | grad norm 1.0 | `run.py:69` |
| init | `torch.manual_seed(seed)` before the model is built | `run.py:53` |
| batch order | generator seeded `seed + 1` | `run.py:60` |
| curve evals | at steps 0, 250, ..., 1250 and 1499; 20 batches of 16, generator seed 999 | `run.py:96`, `:97`, `:72`, `:73`, `:45` |
| final eval | 40 batches of 16, generator seed 999 | `run.py:61`, `:77` |
| RoPE | decoupled for MLA, on the 16 rope channels; full head width otherwise | `model.py:33`, `:54` |

The evaluation generator is seeded 999 for every variant and seed, so all
18 models were scored on the same validation windows.

## The six variants

`VARIANTS` at `run.py:30` to `:37`. Cache per token is summed over the four
layers (`model.py:63`): `2 * kv_heads * d_head` per layer for the grouped
family (`mla/baselines.py:46`), `d_c + d_rope` for MLA (`mla/naive.py:58`).

| variant | kv heads or latent | `params` | `cache_per_token` | `wall_s`, 3 seeds |
|---|---|---:|---:|---|
| MHA | 6 kv heads | 1,801,728 | 1536 | 107.9 s, all three |
| GQA(g=2) | 3 | 1,654,272 | 768 | 100.0 s to 105.1 s |
| GQA(g=3) | 2 | 1,605,120 | 512 | 97.6 s to 99.2 s |
| MQA | 1 | 1,555,968 | 256 | 95.7 s to 96.1 s |
| MLA(d_c=48) | d_c 48, d_rope 16 | 1,703,424 | 256 | 112.2 s to 112.5 s |
| MLA(d_c=96) | d_c 96, d_rope 16 | 1,814,016 | 448 | 113.9 s to 114.3 s |

The kv head count is `6 // g` (`model.py:42`). `params`, `cache_per_token`
and `wall_s` are columns of `quality.csv`. The 18 runs sum to 1891 s of the
1904 s total; the rest is loading the corpus and the final evaluations,
which sit outside the timed loop (`run.py:63`, `:75`, `:77`).

## What it wrote

- `results/quality.csv`, 18 rows, one per variant and seed.
- `results/curves.csv`, 126 rows, 7 evaluation points per run.
- `results/train-meta.json`.

`run.py:115` to `:124`. The trained model is returned from `train_one`
(`run.py:84`) and dropped at `run.py:108`. Nothing saves it, and there is no
`torch.save` in the repository. `results/cache.csv` is not from this run at
all; it is arithmetic from `bench/cache.py`.
