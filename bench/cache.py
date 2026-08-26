"""Exact KV cache accounting. No measurement error, because it is arithmetic.

Cache per token per layer, in elements:

    MHA / GQA / MQA   2 * n_kv * d_head        (K and V, one vector per KV head)
    MLA               d_c + d_rope             (the latent, plus the shared rope key)

At DeepSeek-V2 scale (n_h=32, d_h=128, d_c=512, d_rope=64) that is 8192 against
576, a 14.2x reduction. The numbers below are computed, not quoted.

    .venv/bin/python -m bench.cache
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

CONFIGS = {
    "deepseek-v2-ish": {"n_heads": 32, "d_head": 128, "d_c": 512, "d_rope": 64, "layers": 60},
    "this-repo": {"n_heads": 6, "d_head": 32, "d_c": 48, "d_rope": 16, "layers": 4},
}


def variants(cfg):
    n, d = cfg["n_heads"], cfg["d_head"]
    out = [("MHA", 2 * n * d)]
    for g in (2, 4, 8):
        if n % g == 0:
            out.append((f"GQA(g={g})", 2 * (n // g) * d))
    out.append(("MQA", 2 * 1 * d))
    out.append(("MLA", cfg["d_c"] + cfg["d_rope"]))
    out.append(("MLA (no decoupled RoPE)", cfg["d_c"]))
    return out


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    rows = []
    for name, cfg in CONFIGS.items():
        base = None
        print(f"\n{name}: n_heads={cfg['n_heads']} d_head={cfg['d_head']} "
              f"d_c={cfg['d_c']} d_rope={cfg['d_rope']} layers={cfg['layers']}")
        print(f"  {'variant':24} {'elem/token/layer':>17} {'vs MHA':>8} "
              f"{'GB @ 128k ctx, fp16':>20}")
        for v, per in variants(cfg):
            base = base or per
            gb = per * cfg["layers"] * 131_072 * 2 / 2**30
            print(f"  {v:24} {per:17} {base / per:7.2f}x {gb:19.2f}")
            rows.append({"config": name, "variant": v, "elem_per_token_per_layer": per,
                         "layers": cfg["layers"], "reduction_vs_mha": base / per,
                         "gb_at_128k_fp16": gb})
    p = RESULTS / "cache.csv"
    with p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
