"""Character level data. Fixed split, fixed seed, so runs are comparable."""
from __future__ import annotations

from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def load(seq: int = 256):
    text = (ROOT / "data" / "input.txt").read_text()
    vocab = sorted(set(text))
    stoi = {c: i for i, c in enumerate(vocab)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    return data[:n], data[n:], len(vocab)


def batch(split: torch.Tensor, size: int, seq: int, generator: torch.Generator):
    i = torch.randint(0, len(split) - seq - 1, (size,), generator=generator)
    x = torch.stack([split[j:j + seq] for j in i])
    y = torch.stack([split[j + 1:j + seq + 1] for j in i])
    return x, y
