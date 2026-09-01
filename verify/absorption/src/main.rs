//! Two checks on the absorption trick, in Rust, sharing nothing with the Python.
//!
//! 1. Reproduce the golden vectors. Same job the C does, from a separate parser
//!    and a separate kernel, so a mistake would have to be made twice.
//!
//! 2. The part Python cannot afford. The test suite asserts naive == absorbed on
//!    nine fixed shapes with one seed. Absorption is an algebraic identity, so
//!    it should hold for every shape and every weight draw, and a bug that only
//!    shows up at, say, n_heads = 1 or seq = 1 would pass the nine. This runs
//!    the identity over thousands of random shapes and random weights with its
//!    own xorshift generator and reports the worst deviation seen.
//!
//!     cargo run --release --quiet -- <repo root>

use std::collections::HashMap;
use std::env;
use std::fs;

const TOL_GOLDEN: f64 = 1e-5;
const TOL_IDENTITY: f64 = 1e-9;
const DRAWS: usize = 50_000;

struct Golden {
    cfg: HashMap<String, i64>,
    scalar: HashMap<String, f64>,
    tensor: HashMap<String, Vec<f64>>,
}

fn parse(path: &str) -> Golden {
    let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
    let mut g = Golden { cfg: HashMap::new(), scalar: HashMap::new(), tensor: HashMap::new() };
    let mut pending: Option<(String, usize)> = None;
    let mut buf: Vec<f64> = Vec::new();
    for line in text.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split_whitespace().collect();
        match f[0] {
            "config" => {
                g.cfg.insert(f[1].to_string(), f[2].parse().expect("bad config"));
            }
            "scalar" => {
                g.scalar.insert(f[1].to_string(), f[2].parse().expect("bad scalar"));
            }
            "tensor" => {
                if let Some((name, n)) = pending.take() {
                    assert_eq!(buf.len(), n, "short tensor {name}");
                    g.tensor.insert(name, std::mem::take(&mut buf));
                }
                let ndim: usize = f[2].parse().expect("bad rank");
                let n: usize = f[3..3 + ndim]
                    .iter()
                    .map(|d| d.parse::<usize>().expect("bad dim"))
                    .product();
                pending = Some((f[1].to_string(), n));
                buf = Vec::with_capacity(n);
            }
            _ => {
                for tok in f {
                    buf.push(tok.parse().expect("bad float"));
                }
            }
        }
    }
    if let Some((name, n)) = pending.take() {
        assert_eq!(buf.len(), n, "short tensor {name}");
        g.tensor.insert(name, buf);
    }
    g
}

impl Golden {
    fn c(&self, k: &str) -> usize {
        *self.cfg.get(k).unwrap_or_else(|| panic!("missing config {k}")) as usize
    }
    fn t(&self, k: &str, n: usize) -> &[f64] {
        let v = self.tensor.get(k).unwrap_or_else(|| panic!("missing tensor {k}"));
        assert_eq!(v.len(), n, "wrong element count for {k}");
        v
    }
}

/// y = W x with W stored (out, in) row major, as PyTorch stores nn.Linear.
fn matvec(w: &[f64], x: &[f64], out: usize, inn: usize) -> Vec<f64> {
    (0..out)
        .map(|o| (0..inn).map(|i| w[o * inn + i] * x[i]).sum())
        .collect()
}

fn rope(v: &mut [f64], pos: usize, theta: f64) {
    let d = v.len();
    for i in 0..d / 2 {
        let f = 1.0 / theta.powf((2 * i) as f64 / d as f64);
        let (s, c) = ((pos as f64 * f).sin(), (pos as f64 * f).cos());
        let (re, im) = (v[2 * i], v[2 * i + 1]);
        v[2 * i] = re * c - im * s;
        v[2 * i + 1] = re * s + im * c;
    }
}

fn softmax(p: &mut [f64]) {
    let m = p.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut s = 0.0;
    for x in p.iter_mut() {
        *x = (*x - m).exp();
        s += *x;
    }
    for x in p.iter_mut() {
        *x /= s;
    }
}

struct Shape {
    dm: usize,
    nh: usize,
    dh: usize,
    dc: usize,
    dr: usize,
    seq: usize,
}

struct Weights {
    w_dkv: Vec<f64>,
    w_uk: Vec<f64>,
    w_uv: Vec<f64>,
    wq: Vec<f64>,
    wq_rope: Vec<f64>,
    wk_rope: Vec<f64>,
    wo: Vec<f64>,
}

/// One sequence through both forms. Returns (naive, absorbed), each seq*d_model.
fn forward(s: &Shape, w: &Weights, x: &[f64], theta: f64) -> (Vec<f64>, Vec<f64>) {
    let scale = ((s.dh + s.dr) as f64).sqrt();
    let (mut c, mut k, mut v, mut q, mut qr, mut kr) =
        (vec![], vec![], vec![], vec![], vec![], vec![]);
    for t in 0..s.seq {
        let xt = &x[t * s.dm..(t + 1) * s.dm];
        let ct = matvec(&w.w_dkv, xt, s.dc, s.dm);
        k.extend(matvec(&w.w_uk, &ct, s.nh * s.dh, s.dc));
        v.extend(matvec(&w.w_uv, &ct, s.nh * s.dh, s.dc));
        c.extend(ct);
        q.extend(matvec(&w.wq, xt, s.nh * s.dh, s.dm));
        let mut qrt = matvec(&w.wq_rope, xt, s.nh * s.dr, s.dm);
        for h in 0..s.nh {
            rope(&mut qrt[h * s.dr..(h + 1) * s.dr], t, theta);
        }
        qr.extend(qrt);
        let mut krt = matvec(&w.wk_rope, xt, s.dr, s.dm);
        rope(&mut krt, t, theta);
        kr.extend(krt);
    }

    // fold: wq_abs (nh, dc, dm) and wo_abs (dm, nh, dc)
    let mut wq_abs = vec![0.0; s.nh * s.dc * s.dm];
    for h in 0..s.nh {
        for cc in 0..s.dc {
            for m in 0..s.dm {
                wq_abs[(h * s.dc + cc) * s.dm + m] = (0..s.dh)
                    .map(|d| w.w_uk[(h * s.dh + d) * s.dc + cc] * w.wq[(h * s.dh + d) * s.dm + m])
                    .sum();
            }
        }
    }
    let mut wo_abs = vec![0.0; s.dm * s.nh * s.dc];
    for m in 0..s.dm {
        for h in 0..s.nh {
            for cc in 0..s.dc {
                wo_abs[(m * s.nh + h) * s.dc + cc] = (0..s.dh)
                    .map(|d| {
                        w.wo[m * s.nh * s.dh + h * s.dh + d] * w.w_uv[(h * s.dh + d) * s.dc + cc]
                    })
                    .sum();
            }
        }
    }

    let mut out_n = vec![0.0; s.seq * s.dm];
    let mut out_a = vec![0.0; s.seq * s.dm];
    for pos in 0..s.seq {
        let mut head_out = vec![0.0; s.nh * s.dh];
        let mut ctx = vec![0.0; s.nh * s.dc];
        for h in 0..s.nh {
            let mut att: Vec<f64> = (0..=pos)
                .map(|t| {
                    let content: f64 = (0..s.dh)
                        .map(|d| q[(pos * s.nh + h) * s.dh + d] * k[(t * s.nh + h) * s.dh + d])
                        .sum();
                    let posn: f64 = (0..s.dr)
                        .map(|d| qr[(pos * s.nh + h) * s.dr + d] * kr[t * s.dr + d])
                        .sum();
                    (content + posn) / scale
                })
                .collect();
            softmax(&mut att);
            for d in 0..s.dh {
                head_out[h * s.dh + d] =
                    (0..=pos).map(|t| att[t] * v[(t * s.nh + h) * s.dh + d]).sum();
            }

            let qlat = matvec(
                &wq_abs[h * s.dc * s.dm..(h + 1) * s.dc * s.dm],
                &x[pos * s.dm..(pos + 1) * s.dm],
                s.dc,
                s.dm,
            );
            let mut att2: Vec<f64> = (0..=pos)
                .map(|t| {
                    let content: f64 =
                        (0..s.dc).map(|cc| qlat[cc] * c[t * s.dc + cc]).sum();
                    let posn: f64 = (0..s.dr)
                        .map(|d| qr[(pos * s.nh + h) * s.dr + d] * kr[t * s.dr + d])
                        .sum();
                    (content + posn) / scale
                })
                .collect();
            softmax(&mut att2);
            for cc in 0..s.dc {
                ctx[h * s.dc + cc] = (0..=pos).map(|t| att2[t] * c[t * s.dc + cc]).sum();
            }
        }
        let o_n = matvec(&w.wo, &head_out, s.dm, s.nh * s.dh);
        out_n[pos * s.dm..(pos + 1) * s.dm].copy_from_slice(&o_n);
        for m in 0..s.dm {
            out_a[pos * s.dm + m] = (0..s.nh)
                .map(|h| {
                    (0..s.dc)
                        .map(|cc| ctx[h * s.dc + cc] * wo_abs[(m * s.nh + h) * s.dc + cc])
                        .sum::<f64>()
                })
                .sum();
        }
    }
    (out_n, out_a)
}

/// xorshift64*, so the random draws need no external crate.
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
    /// Box-Muller normal, so the weights look like the ones training produces.
    fn normal(&mut self) -> f64 {
        let u1 = ((self.next_u64() >> 11) as f64 + 1.0) / (1u64 << 53) as f64;
        let u2 = ((self.next_u64() >> 11) as f64) / (1u64 << 53) as f64;
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
    fn vec(&mut self, n: usize, sd: f64) -> Vec<f64> {
        (0..n).map(|_| self.normal() * sd).collect()
    }
}

fn max_abs(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

fn main() {
    let root = env::args().nth(1).unwrap_or_else(|| ".".to_string());
    let g = parse(&format!("{root}/verify/golden/mla_golden.txt"));
    let (dm, nh, dh, dc, dr, seq, batch) = (
        g.c("d_model"), g.c("n_heads"), g.c("d_head"), g.c("d_c"),
        g.c("d_rope"), g.c("seq"), g.c("batch"),
    );
    let theta = *g.scalar.get("rope_theta").expect("missing rope_theta");
    let shape = Shape { dm, nh, dh, dc, dr, seq };
    let w = Weights {
        w_dkv: g.t("w_dkv", dc * dm).to_vec(),
        w_uk: g.t("w_uk", nh * dh * dc).to_vec(),
        w_uv: g.t("w_uv", nh * dh * dc).to_vec(),
        wq: g.t("wq", nh * dh * dm).to_vec(),
        wq_rope: g.t("wq_rope", nh * dr * dm).to_vec(),
        wk_rope: g.t("wk_rope", dr * dm).to_vec(),
        wo: g.t("wo", dm * nh * dh).to_vec(),
    };
    let x = g.t("x", batch * seq * dm);
    let gold_n = g.t("out_naive", batch * seq * dm);
    let gold_a = g.t("out_absorbed", batch * seq * dm);

    let (mut dn, mut da) = (0.0f64, 0.0f64);
    for b in 0..batch {
        let sl = b * seq * dm..(b + 1) * seq * dm;
        let (on, oa) = forward(&shape, &w, &x[sl.clone()], theta);
        dn = dn.max(max_abs(&on, &gold_n[sl.clone()]));
        da = da.max(max_abs(&oa, &gold_a[sl]));
    }
    println!("Rust, golden vectors, batch {batch} seq {seq} d_model {dm}");
    println!("  Rust naive    vs golden naive    max abs diff {dn:.3e}");
    println!("  Rust absorbed vs golden absorbed max abs diff {da:.3e}");

    // The exhaustive part: the identity over random shapes and random weights.
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut worst = 0.0f64;
    let mut worst_shape = String::new();
    for _ in 0..DRAWS {
        let nh = 1 + rng.below(4);
        let dh = 2 * (1 + rng.below(4));
        let dc = 2 + rng.below(12);
        let dr = 2 * (1 + rng.below(4));
        let dm = nh * dh;
        let s = Shape { dm, nh, dh, dc, dr, seq: 1 + rng.below(8) };
        let sd = 1.0 / (s.dm as f64).sqrt();
        let w = Weights {
            w_dkv: rng.vec(s.dc * s.dm, sd),
            w_uk: rng.vec(s.nh * s.dh * s.dc, sd),
            w_uv: rng.vec(s.nh * s.dh * s.dc, sd),
            wq: rng.vec(s.nh * s.dh * s.dm, sd),
            wq_rope: rng.vec(s.nh * s.dr * s.dm, sd),
            wk_rope: rng.vec(s.dr * s.dm, sd),
            wo: rng.vec(s.dm * s.nh * s.dh, sd),
        };
        let x = rng.vec(s.seq * s.dm, 1.0);
        let label = format!(
            "n_heads={} d_head={} d_c={} d_rope={} seq={}",
            s.nh, s.dh, s.dc, s.dr, s.seq
        );
        let (on, oa) = forward(&s, &w, &x, theta);
        let d = max_abs(&on, &oa);
        if d > worst {
            worst = d;
            worst_shape = label;
        }
    }
    println!("  absorbed == naive over {DRAWS} random shapes and weight draws");
    println!("    worst max abs diff {worst:.3e} at {worst_shape}");

    let bad = dn > TOL_GOLDEN || da > TOL_GOLDEN || worst > TOL_IDENTITY;
    println!(
        "{}: golden tolerance {:.0e}, identity tolerance {:.0e}",
        if bad { "FAIL" } else { "ok" },
        TOL_GOLDEN,
        TOL_IDENTITY
    );
    std::process::exit(bad as i32);
}
