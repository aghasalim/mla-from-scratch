"""Figures from results/*.csv. Nothing is re-measured here."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ORDER = ["MHA", "GQA(g=2)", "GQA(g=3)", "MQA", "MLA(d_c=48)", "MLA(d_c=96)"]
COL = {"MHA": "#b2182b", "GQA(g=2)": "#ef8a62", "GQA(g=3)": "#fddbc7",
       "MQA": "#92c5de", "MLA(d_c=48)": "#1a9850", "MLA(d_c=96)": "#66bd63"}


def fig_cache(out: Path) -> Path:
    t = pd.read_csv(RESULTS / "cache.csv")
    t = t[t["config"] == "deepseek-v2-ish"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    names = t["variant"].tolist()
    bars = a.bar(range(len(t)), t["elem_per_token_per_layer"],
                 color=["#b2182b", "#ef8a62", "#fdae61", "#fee08b", "#92c5de", "#1a9850", "#66bd63"][:len(t)])
    for r, v, red in zip(bars, t["elem_per_token_per_layer"], t["reduction_vs_mha"]):
        a.text(r.get_x() + r.get_width() / 2, v * 1.05, f"{int(v)}\n{red:.1f}x", ha="center", fontsize=8)
    a.set_xticks(range(len(t))); a.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    a.set_yscale("log"); a.set_ylabel("cache elements per token per layer")
    a.set_title("KV cache per token, DeepSeek-V2 shape\n(n_h=32, d_h=128, d_c=512, d_rope=64)")
    a.grid(alpha=0.3, axis="y", which="both")

    bars = b.bar(range(len(t)), t["gb_at_128k_fp16"],
                 color=["#b2182b", "#ef8a62", "#fdae61", "#fee08b", "#92c5de", "#1a9850", "#66bd63"][:len(t)])
    for r, v in zip(bars, t["gb_at_128k_fp16"]):
        b.text(r.get_x() + r.get_width() / 2, v + 2, f"{v:.1f}", ha="center", fontsize=8.5)
    b.set_xticks(range(len(t))); b.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    b.set_ylabel("GB"); b.set_title("Cache at 128k context, fp16, 60 layers\nthis is the wall MLA is aimed at")
    b.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
    return out


def fig_quality(out: Path) -> Path:
    t = pd.read_csv(RESULTS / "quality.csv")
    fig, ax = plt.subplots(figsize=(10, 5.6))
    for v in ORDER:
        s = t[t["variant"] == v]
        if s.empty:
            continue
        ax.scatter(s["cache_per_token"], s["val_ppl"], s=70, color=COL[v], label=v, zorder=3)
        ax.errorbar(s["cache_per_token"].iloc[0], s["val_ppl"].median(),
                    yerr=[[s["val_ppl"].median() - s["val_ppl"].min()],
                          [s["val_ppl"].max() - s["val_ppl"].median()]],
                    color=COL[v], capsize=5, zorder=2)
    lo, hi = t["val_ppl"].min(), t["val_ppl"].max()
    ax.axhspan(lo, hi, color="#cccccc", alpha=0.35, zorder=1)
    ax.text(t["cache_per_token"].max() * 0.55, hi,
            f"full range of all 18 runs: {lo:.3f} to {hi:.3f}", fontsize=9, va="bottom")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("KV cache elements per token (all layers)")
    ax.set_ylabel("validation perplexity")
    ax.set_title("Quality against cache budget, 3 seeds each\n"
                 "the variants are not separated at this scale, and the grey band is why")
    ax.grid(alpha=0.3, which="both")
    ax.legend(frameon=False, fontsize=9, ncol=2)
    fig.tight_layout(); fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
    return out


def fig_curves(out: Path) -> Path:
    t = pd.read_csv(RESULTS / "curves.csv")
    fig, ax = plt.subplots(figsize=(10, 5.4))
    for v in ORDER:
        s = t[t["variant"] == v]
        if s.empty:
            continue
        g = s.groupby("step")["val_loss"]
        ax.plot(g.median().index, g.median().values, color=COL[v], label=v, linewidth=1.8)
        ax.fill_between(g.median().index, g.min().values, g.max().values,
                        color=COL[v], alpha=0.15, linewidth=0)
    ax.set_xlabel("training step"); ax.set_ylabel("validation loss")
    ax.set_title("Validation loss, median and range over 3 seeds\n"
                 "the curves sit on top of each other, which is the result")
    ax.grid(alpha=0.3); ax.legend(frameon=False, fontsize=9, ncol=2)
    fig.tight_layout(); fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
    return out


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    for p in (fig_cache(RESULTS / "cache.png"),
              fig_quality(RESULTS / "quality.png"),
              fig_curves(RESULTS / "curves.png")):
        print(f"-> {p.relative_to(ROOT)} ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
