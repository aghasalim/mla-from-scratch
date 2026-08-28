"""Figures from results/*.csv. Nothing is re-measured here.

    .venv/bin/python -m bench.figures

Every panel reads a committed CSV. The animation is arithmetic on cache.csv, so
it has no randomness in it and needs no seed, and it asserts its endpoint
against the committed number before it draws a frame.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.ticker import NullFormatter, NullLocator
from matplotlib.transforms import blended_transform_factory

from bench.style import PALETTE, titled

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

ORDER = ["MHA", "GQA(g=2)", "GQA(g=3)", "MQA", "MLA(d_c=48)", "MLA(d_c=96)"]

# Three colours carry meaning and are the same in every figure here: red is the
# full cache that hits the wall, blue is MQA because it is the real rival at a
# matched budget, green is MLA. The GQA rungs are context and take whatever is
# left in the palette.
COLOR = {
    "MHA": PALETTE[1],
    "GQA(g=2)": PALETTE[3],
    "GQA(g=3)": PALETTE[4],
    "MQA": PALETTE[0],
    "MLA(d_c=48)": PALETTE[2],
    "MLA(d_c=96)": PALETTE[2],
}
# The two MLA widths share a colour, so the wider one is dashed and gets a
# square marker.
DASH = {"MLA(d_c=96)": (0, (5, 2))}
MARK = {"MLA(d_c=96)": "s"}

CACHE_COLOR = {
    "MHA": PALETTE[1],
    "MQA": PALETTE[0],
    "MLA": PALETTE[2],
    "MLA (no decoupled RoPE)": PALETTE[2],
}


def _cache_table() -> pd.DataFrame:
    t = pd.read_csv(RESULTS / "cache.csv")
    return t[t["config"] == "deepseek-v2-ish"].reset_index(drop=True)


def fig_cache(out: Path) -> Path:
    """One bar per variant, with the derived quantities in a column beside it.

    The old version had two panels, but GB at 128k is the element count times a
    constant, so the second panel was the first one again on a different scale.
    """
    t = _cache_table().sort_values("elem_per_token_per_layer")
    y = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    bars = ax.barh(y, t["elem_per_token_per_layer"], height=0.62,
                   color=[CACHE_COLOR.get(v, PALETTE[5]) for v in t["variant"]])
    # The no-RoPE row is arithmetic you would never deploy, so it is faded.
    for bar, v in zip(bars, t["variant"]):
        if v.startswith("MLA (no"):
            bar.set_alpha(0.4)

    ax.set_yticks(y)
    ax.set_yticklabels(t["variant"])
    ax.set_ylim(-0.7, len(t) - 0.1)
    ax.set_xlim(0, t["elem_per_token_per_layer"].max() * 1.02)
    ax.set_xlabel("KV cache elements per token per layer")
    ax.grid(axis="y", visible=False)

    # Value column outside the axes, so nothing sits on top of a bar and the
    # numbers line up with each other instead of with the bar ends.
    tr = blended_transform_factory(ax.transAxes, ax.transData)
    for x, head in ((1.08, "GB at 128k"), (1.21, "vs MHA")):
        ax.text(x, len(t) - 0.42, head, transform=tr, ha="right", va="bottom",
                fontsize=9, color="#5a5a5a")
    for i, r in enumerate(t.itertuples()):
        ax.text(1.08, i, f"{r.gb_at_128k_fp16:.2f} GB", transform=tr,
                ha="right", va="center", fontsize=9.5)
        ax.text(1.21, i, f"{r.reduction_vs_mha:.1f}x", transform=tr,
                ha="right", va="center", fontsize=9.5, color="#5a5a5a")

    g = t.set_index("variant")["gb_at_128k_fp16"]
    titled(ax,
           f"MLA takes the 128k cache from {g['MHA']:.0f} GB down to {g['MLA']:.1f} GB",
           "Exact arithmetic at DeepSeek-V2 shape: 32 heads of width 128, "
           "d_c=512, d_rope=64, 60 layers.")
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_quality(out: Path) -> Path:
    t = pd.read_csv(RESULTS / "quality.csv")
    lo, hi = t["val_ppl"].min(), t["val_ppl"].max()
    fig, ax = plt.subplots(figsize=(9.8, 5.2))
    ax.axhspan(lo, hi, color="#e9e9e9", zorder=0)

    # MQA and MLA(d_c=48) sit at the same budget. Both are nudged 3 percent
    # sideways so the two clusters are readable; the tick still marks 256.
    per_budget = t.groupby("cache_per_token")["variant"].nunique()
    used: dict[int, int] = {}
    for v in ORDER:
        s = t[t["variant"] == v]
        if s.empty:
            continue
        budget = int(s["cache_per_token"].iloc[0])
        k = used.get(budget, 0)
        used[budget] = k + 1
        x = budget * (1.0 if per_budget[budget] == 1 else (0.968, 1.033)[k])
        c = COLOR[v]
        ax.scatter([x] * len(s), s["val_ppl"], s=24, color=c, alpha=0.4,
                   linewidths=0, zorder=2)
        ax.plot([x, x], [s["val_ppl"].min(), s["val_ppl"].max()], color=c,
                linewidth=1.6, zorder=3)
        ax.plot([x], [s["val_ppl"].median()], marker=MARK.get(v, "o"),
                markersize=9, color=c, markeredgecolor="white", linestyle="none",
                markeredgewidth=1.1, label=v, zorder=4)

    ticks = sorted(t["cache_per_token"].unique())
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(int(v)) for v in ticks])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(ticks[0] * 0.85, ticks[-1] * 1.18)
    pad = (hi - lo) * 0.22
    ax.set_ylim(lo - pad, hi + pad * 0.45)
    ax.text(ticks[0] * 0.88, lo - pad * 0.62,
            "MQA and MLA(d_c=48) run at the same budget, nudged apart here",
            fontsize=9, color="#5a5a5a", va="center")
    ax.set_xlabel("KV cache elements per token, all 4 layers")
    ax.set_ylabel("validation perplexity (lower is better)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    titled(ax, "Nothing separates the variants at this budget",
           f"Dots are seeds, bars are min to max. All 18 runs land in the grey "
           f"band, {lo:.3f} to {hi:.3f}.")
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_curves(out: Path) -> Path:
    """Left panel is the whole run, right panel is the tail on a tight axis.

    The right panel exists because the step 0 loss sets the scale on the left
    and hides whether the curves actually separate later. They do not.
    """
    t = pd.read_csv(RESULTS / "curves.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.0, 5.0),
                               gridspec_kw={"width_ratios": [1.15, 1.0]})
    zoom_from = 500
    for ax, start in ((a, 0), (b, zoom_from)):
        w = t[t["step"] >= start]
        for v in ORDER:
            s = w[w["variant"] == v]
            if s.empty:
                continue
            # Raw seeds faint behind the median, rather than a filled band. The
            # band hid how few points there are: 7 evals, 3 seeds, no smoothing.
            for _, run in s.groupby("seed"):
                ax.plot(run["step"], run["val_loss"], color=COLOR[v],
                        alpha=0.22, linewidth=0.9, zorder=2)
            g = s.groupby("step")["val_loss"].median()
            ax.plot(g.index, g.values, color=COLOR[v], linewidth=1.9,
                    linestyle=DASH.get(v, "-"), zorder=3,
                    label=v if ax is a else None)
        ax.set_xlabel("training step")
    a.set_ylabel("validation loss (nats per character)")
    tail = t[t["step"] >= zoom_from]["val_loss"]
    b.set_ylim(tail.min() - 0.01, tail.max() + 0.01)
    b.set_xlim(zoom_from - 30, t["step"].max() + 30)

    titled(a, "Every variant learns the same curve",
           "Median of 3 seeds, the seeds themselves faint underneath.")
    last = t[t["step"] == t["step"].max()].groupby("variant")["val_loss"]
    across = last.median().max() - last.median().min()
    within = (last.max() - last.min()).mean()
    titled(b, "Too close to call, even at the end",
           f"Step {int(t['step'].max())}: variants span {across:.3f} nats, "
           f"one variant's seeds span {within:.3f}.")
    handles, labels = a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def anim_cache(out: Path) -> Path:
    """The cache filling up as the context grows, full KV against the latent.

    Arithmetic on results/cache.csv: frame k is elements * layers * L_k * 2
    bytes. No sampling, no model, nothing to seed. The endpoint is asserted
    against the committed GB figure so this cannot drift from the table.
    """
    t = _cache_table().set_index("variant")
    layers = int(t["layers"].iloc[0])
    ctx_max = 131_072

    def gb(variant: str, ctx: float) -> float:
        return t.at[variant, "elem_per_token_per_layer"] * layers * ctx * 2 / 2**30

    for v in ("MHA", "MLA"):
        assert abs(gb(v, ctx_max) - t.at[v, "gb_at_128k_fp16"]) < 1e-9, v

    hbm = 80.0
    grow, hold = 100, 14
    ctxs = np.concatenate([np.linspace(0.0, ctx_max, grow),
                           np.full(hold, float(ctx_max))])

    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    ax.set_xlim(-0.8, 1.8)
    ax.set_ylim(0, 128)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["MHA\nfull K and V", "MLA\nlatent only"])
    ax.set_ylabel("KV cache (GB, fp16, 60 layers)")
    ax.grid(axis="x", visible=False)
    ax.axhline(hbm, color=PALETTE[5], linestyle=(0, (5, 3)), linewidth=1.3)
    ax.text(1.77, hbm + 1.6, "80 GB, one H100", ha="right", va="bottom",
            fontsize=9, color=PALETTE[5])
    ax.text(1.0, 46, f"{t.at['MLA', 'reduction_vs_mha']:.1f}x smaller\n"
                     "at every length", ha="center", fontsize=9.5,
            color=PALETTE[2])

    bars = ax.bar([0, 1], [0.0, 0.0], width=0.5,
                  color=[PALETTE[1], PALETTE[2]])
    tags = [ax.text(x, 0, "", ha="center", va="bottom", fontsize=10)
            for x in (0, 1)]
    # Under the line and beside the bar. On the bar it would be red on red, and
    # above the line it would sit on the 80 GB label.
    over = ax.text(0.31, hbm - 5.0, "", ha="left", va="top", fontsize=9,
                   color=PALETTE[1])
    # Same layout as style.titled, but the second line has to change per frame.
    ax.set_title("Full KV runs off the card, the latent never gets near it",
                 pad=26)
    ticker = ax.text(0.0, 1.012, "", transform=ax.transAxes, fontsize=9.3,
                     color="#5a5a5a", va="bottom", ha="left")

    def draw(i: int):
        ctx = ctxs[i]
        for bar, tag, v in zip(bars, tags, ("MHA", "MLA")):
            h = gb(v, ctx)
            bar.set_height(h)
            tag.set_position((bar.get_x() + bar.get_width() / 2, h + 1.6))
            tag.set_text(f"{h:.1f} GB")
        ticker.set_text(f"context filled: {ctx:,.0f} of {ctx_max:,} tokens")
        excess = gb("MHA", ctx) - hbm
        # Only once it rounds to a whole GB, so no frame says "over by 0 GB".
        over.set_text(f"over one card by {excess:.0f} GB" if excess >= 1 else "")
        return (*bars, *tags, ticker, over)

    anim = FuncAnimation(fig, draw, frames=len(ctxs), blit=False)
    # dpi is passed here on purpose. The style saves stills at 170, and 114
    # frames at that size would be a several MB GIF for no extra detail.
    anim.save(out, writer=PillowWriter(fps=15), dpi=100)
    plt.close(fig)
    return out


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    for p in (fig_cache(RESULTS / "cache.png"),
              fig_quality(RESULTS / "quality.png"),
              fig_curves(RESULTS / "curves.png"),
              anim_cache(RESULTS / "cache-growth.gif")):
        print(f"-> {p.relative_to(ROOT)} ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
