"""Export golden reference vectors for the MLA kernel from PyTorch.

The C and Rust reimplementations in verify/ read the file this writes and have
to reproduce the two output tensors from the weights alone. Nothing about the
Python is available to them except the numbers, so an error in mla/naive.py or
mla/absorbed.py would have to be reproduced independently in C and in Rust to
survive.

The file is plain whitespace separated text so a parser is a few lines in any
language:

    config <name> <int>
    scalar <name> <float>
    tensor <name> <ndim> <dim0> [dim1 ...]   then prod(dims) floats

Floats are written with repr precision, so the values a C or Rust parser reads
are bit for bit the float32 values PyTorch held.

    python verify/export_golden.py            rewrite the file
    python verify/export_golden.py --check    fail if the committed file is stale

--check is what CI runs. Without it the golden file could drift away from
mla/ and the C and Rust checks would keep passing against a stale reference.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mla.absorbed import AbsorbedMLA
from mla.naive import MLA
from mla.rope import rope_frequencies

CFG = {"d_model": 32, "n_heads": 4, "d_head": 8, "d_c": 24, "d_rope": 8,
       "seq": 12, "batch": 2}
OUT = ROOT / "verify" / "golden" / "mla_golden.txt"
CHECK_TOL = 1e-5


def build():
    """Return (scalars, tensors) for the reference instance."""
    torch.manual_seed(20240901)
    c = CFG
    m = MLA(c["d_model"], c["n_heads"], d_c=c["d_c"], d_rope=c["d_rope"],
            d_head=c["d_head"]).eval()
    a = AbsorbedMLA(m).eval()
    freqs = rope_frequencies(c["d_rope"], c["seq"])
    x = torch.randn(c["batch"], c["seq"], c["d_model"])
    with torch.no_grad():
        out_naive, out_absorbed = m(x, freqs), a(x, freqs)

    # The agreement the README quotes, over the same grid the test suite uses:
    # three latent widths by three sequence lengths at the test shape.
    grid = 0.0
    for d_c in (48, 96, 192):
        for seq in (1, 7, 48):
            torch.manual_seed(0)
            gm = MLA(256, 8, d_c=d_c, d_rope=16).eval()
            ga = AbsorbedMLA(gm).eval()
            gx = torch.randn(3, seq, 256)
            gf = rope_frequencies(16, seq)
            with torch.no_grad():
                grid = max(grid, (gm(gx, gf) - ga(gx, gf)).abs().max().item())

    scalars = {
        "rope_theta": 10000.0,
        "torch_naive_vs_absorbed": (out_naive - out_absorbed).abs().max().item(),
        "torch_grid_max_abs_diff": grid,
    }
    tensors = {"x": x}
    for name, lin in (("w_dkv", m.w_dkv), ("w_uk", m.w_uk), ("w_uv", m.w_uv),
                      ("wq", m.wq), ("wq_rope", m.wq_rope), ("wk_rope", m.wk_rope),
                      ("wo", m.wo)):
        tensors[name] = lin.weight
    tensors["out_naive"] = out_naive
    tensors["out_absorbed"] = out_absorbed
    return scalars, tensors


def write(scalars, tensors) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as fh:
        fh.write("# golden reference vectors for multi-head latent attention\n")
        fh.write("# written by verify/export_golden.py from mla/naive.py and mla/absorbed.py\n")
        fh.write(f"# torch {torch.__version__}\n")
        for k, v in CFG.items():
            fh.write(f"config {k} {v}\n")
        for k, v in scalars.items():
            fh.write(f"scalar {k} {v!r}\n")
        for name, t in tensors.items():
            v = t.detach().float().reshape(-1).tolist()
            fh.write(f"tensor {name} {t.dim()} " + " ".join(str(d) for d in t.shape) + "\n")
            for i in range(0, len(v), 6):
                fh.write(" ".join(repr(x) for x in v[i:i + 6]) + "\n")


def parse(path: Path):
    cfg, scalars, tensors = {}, {}, {}
    name, want, buf = None, 0, []
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        f = line.split()
        if f[0] == "config":
            cfg[f[1]] = int(f[2])
        elif f[0] == "scalar":
            scalars[f[1]] = float(f[2])
        elif f[0] == "tensor":
            if name:
                tensors[name] = buf
            ndim = int(f[2])
            want = 1
            for d in f[3:3 + ndim]:
                want *= int(d)
            name, buf = f[1], []
        else:
            buf.extend(float(t) for t in f)
    if name:
        tensors[name] = buf
    assert len(buf) == want, f"short tensor {name}"
    return cfg, scalars, tensors


def check(scalars, tensors) -> int:
    if not OUT.exists():
        print(f"{OUT.relative_to(ROOT)} is missing")
        return 1
    cfg, old_scalars, old_tensors = parse(OUT)
    bad = []
    if cfg != CFG:
        bad.append(f"config {cfg} in the file, {CFG} in this script")
    for k, v in scalars.items():
        got = old_scalars.get(k)
        if got is None or abs(got - v) > max(CHECK_TOL * abs(v), 1e-12):
            bad.append(f"scalar {k}: file has {got}, torch now gives {v!r}")
    for name, t in tensors.items():
        fresh = t.detach().float().reshape(-1).tolist()
        old = old_tensors.get(name)
        if old is None:
            bad.append(f"tensor {name} missing from the file")
        elif len(old) != len(fresh):
            bad.append(f"tensor {name}: file has {len(old)} values, torch gives {len(fresh)}")
        else:
            d = max(abs(a - b) for a, b in zip(old, fresh))
            if d > CHECK_TOL:
                bad.append(f"tensor {name}: max abs diff {d:.3e} against the committed file")
    if bad:
        print("the committed golden vectors are not what mla/ produces now:")
        for b in bad:
            print(f"  {b}")
        print("regenerate with: python verify/export_golden.py")
        return 1
    print(f"{OUT.relative_to(ROOT)} still matches mla/ to {CHECK_TOL:.0e}")
    return 0


def main() -> int:
    scalars, tensors = build()
    if "--check" in sys.argv[1:]:
        return check(scalars, tensors)
    write(scalars, tensors)
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"torch naive vs absorbed max abs diff: {scalars['torch_naive_vs_absorbed']:.3e}")
    print(f"torch grid (3 widths x 3 lengths) max abs diff: "
          f"{scalars['torch_grid_max_abs_diff']:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
