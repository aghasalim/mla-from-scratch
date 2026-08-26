"""Fail if a number in README.md no longer matches results/."""
from __future__ import annotations

import csv
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    cache = list(csv.DictReader((ROOT / "results" / "cache.csv").open()))
    qual = list(csv.DictReader((ROOT / "results" / "quality.csv").open()))
    body = (ROOT / "README.md").read_text()
    claims, failures = [], []

    for r in cache:
        if r["config"] != "deepseek-v2-ish":
            continue
        claims.append((f"cache {r['variant']}", r["elem_per_token_per_layer"]))
        claims.append((f"reduction {r['variant']}", f"{float(r['reduction_vs_mha']):.2f}"))
        claims.append((f"GB {r['variant']}", f"{float(r['gb_at_128k_fp16']):.2f}"))

    by: dict[str, list[float]] = {}
    cache_by: dict[str, str] = {}
    for r in qual:
        by.setdefault(r["variant"], []).append(float(r["val_ppl"]))
        cache_by[r["variant"]] = r["cache_per_token"]
    for v, vals in by.items():
        claims.append((f"ppl {v}", f"{statistics.median(vals):.3f}"))
        claims.append((f"ppl-min {v}", f"{min(vals):.3f}"))
        claims.append((f"ppl-max {v}", f"{max(vals):.3f}"))
        claims.append((f"cache {v}", cache_by[v]))

    for label, text in claims:
        if not re.search(r"(?<![\d.])" + re.escape(text) + r"(?!\d)", body):
            failures.append(f"{label} should read {text}, not found")

    print(f"checked {len(claims)} quoted figures against results/")
    if failures:
        print("\nDRIFT DETECTED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("no drift")
    return 0


if __name__ == "__main__":
    sys.exit(main())
