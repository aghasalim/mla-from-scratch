"""Train every attention variant under identical conditions and report perplexity.

Same depth, width, data, batch, steps and seed for all variants. The only thing
that changes is the attention module, so any difference in validation perplexity
is attributable to the KV cache design and nothing else.

The comparison that matters is MLA against MQA: both are configured to 256 cache
elements per token per model here, so it is quality at a matched budget rather
than quality at whatever budget each happens to use.

    .venv/bin/python -m train.run
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch

from train.data import batch, load
from train.model import CharLM

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

VARIANTS = [
    ("MHA", {}),
    ("GQA(g=2)", {}),
    ("GQA(g=3)", {}),
    ("MQA", {}),
    ("MLA", {"d_c": 48, "d_rope": 16}),
    ("MLA", {"d_c": 96, "d_rope": 16}),
]


@torch.no_grad()
def evaluate(model, split, seq, iters, generator):
    model.eval()
    total = 0.0
    for _ in range(iters):
        x, y = batch(split, 16, seq, generator)
        _, loss = model(x, y)
        total += loss.item()
    model.train()
    return total / iters


def train_one(name, kwargs, vocab, tr, va, args, seed):
    torch.manual_seed(seed)
    model = CharLM(vocab, d_model=args.d_model, n_layers=args.layers,
                   n_heads=args.heads, seq=args.seq, variant=name, **kwargs)
    label = name if not kwargs else f"{name}(d_c={kwargs['d_c']})"
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps,
                                                pct_start=0.1)
    g = torch.Generator().manual_seed(seed + 1)
    ge = torch.Generator().manual_seed(999)          # eval batches fixed across variants

    curve, t0 = [], time.perf_counter()
    for step in range(args.steps):
        x, y = batch(tr, args.batch, args.seq, g)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % args.eval_every == 0 or step == args.steps - 1:
            vl = evaluate(model, va, args.seq, args.eval_iters, torch.Generator().manual_seed(999))
            curve.append({"step": step, "train_loss": loss.item(), "val_loss": vl})
    wall = time.perf_counter() - t0

    val = evaluate(model, va, args.seq, args.eval_iters * 2, ge)
    return {
        "variant": label, "seed": seed,
        "params": sum(p.numel() for p in model.parameters()),
        "cache_per_token": model.cache_elements_per_token(),
        "val_loss": val, "val_ppl": math.exp(val),
        "wall_s": wall, "curve": curve,
    }, model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-iters", type=int, default=20)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    args = ap.parse_args()

    tr, va, vocab = load(args.seq)
    RESULTS.mkdir(exist_ok=True)
    rows, curves = [], []
    started = time.perf_counter()

    for name, kw in VARIANTS:
        for seed in args.seeds:
            out, _model = train_one(name, kw, vocab, tr, va, args, seed)
            print(f"  {out['variant']:16} seed {seed}  cache/token {out['cache_per_token']:5}  "
                  f"val ppl {out['val_ppl']:7.3f}  {out['wall_s']:6.1f}s")
            for c in out.pop("curve"):
                curves.append({"variant": out["variant"], "seed": seed, **c})
            rows.append(out)

    for fname, data in (("quality.csv", rows), ("curves.csv", curves)):
        p = RESULTS / fname
        with p.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
        print(f"wrote {p.relative_to(ROOT)} ({len(data)} rows)")
    (RESULTS / "train-meta.json").write_text(json.dumps({
        **vars(args), "wall_clock_s": time.perf_counter() - started,
        "torch": torch.__version__, "device": "cpu"}, indent=1))
    print(f"total {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
