// Recompute results/cache.csv from the shape parameters alone, in JavaScript.
//
// The SQL verifier already does this, but the point of each language is an
// independent reimplementation: a bug would have to appear identically in both
// to survive. Node stdlib only, no packages.

import { readFileSync } from "node:fs";
import { join } from "node:path";

const root = process.argv[2] || ".";

function parseCSV(text) {
  const lines = text.trim().split("\n");
  const header = lines[0].split(",");
  const rows = lines.slice(1).map((l) => {
    const vals = l.split(",");
    const obj = {};
    header.forEach((h, i) => (obj[h] = vals[i]));
    return obj;
  });
  return { header, rows };
}

// --- shape configs, same as cache.sql and bench/cache.py --------------------
const configs = [
  { config: "deepseek-v2-ish", n_heads: 32, d_head: 128, d_c: 512, d_rope: 64, layers: 60 },
  { config: "this-repo",       n_heads: 6,  d_head: 32,  d_c: 48,  d_rope: 16, layers: 4  },
];
const gqa_groups = [2, 4, 8];

function computeRows() {
  const out = [];
  for (const c of configs) {
    const mha_per = 2 * c.n_heads * c.d_head;
    // MHA
    out.push({ config: c.config, variant: "MHA",
      elem: mha_per, layers: c.layers,
      reduction: 1.0,
      gb: (mha_per * c.layers * 131072 * 2) / 1073741824 });
    // GQA variants
    for (const g of gqa_groups) {
      if (c.n_heads % g !== 0) continue;
      const per = 2 * (c.n_heads / g) * c.d_head;
      out.push({ config: c.config, variant: `GQA(g=${g})`,
        elem: per, layers: c.layers,
        reduction: mha_per / per,
        gb: (per * c.layers * 131072 * 2) / 1073741824 });
    }
    // MQA
    const mqaPer = 2 * c.d_head;
    out.push({ config: c.config, variant: "MQA",
      elem: mqaPer, layers: c.layers,
      reduction: mha_per / mqaPer,
      gb: (mqaPer * c.layers * 131072 * 2) / 1073741824 });
    // MLA
    const mlaPer = c.d_c + c.d_rope;
    out.push({ config: c.config, variant: "MLA",
      elem: mlaPer, layers: c.layers,
      reduction: mha_per / mlaPer,
      gb: (mlaPer * c.layers * 131072 * 2) / 1073741824 });
    // MLA (no decoupled RoPE)
    out.push({ config: c.config, variant: "MLA (no decoupled RoPE)",
      elem: c.d_c, layers: c.layers,
      reduction: mha_per / c.d_c,
      gb: (c.d_c * c.layers * 131072 * 2) / 1073741824 });
  }
  return out;
}

// --- load and compare -------------------------------------------------------
const published = parseCSV(readFileSync(join(root, "results", "cache.csv"), "utf8"));
const computed  = computeRows();

let failures = 0;

// Check row count
if (published.rows.length !== computed.length) {
  console.error(`row count: file has ${published.rows.length}, computed ${computed.length}`);
  process.exit(1);
}

// Compare each row
for (let i = 0; i < computed.length; i++) {
  const c = computed[i];
  const p = published.rows[i];

  if (p.config !== c.config || p.variant !== c.variant) {
    console.error(`row ${i + 2}: config/variant mismatch: ` +
      `file has ${p.config}/${p.variant}, computed ${c.config}/${c.variant}`);
    failures++;
    continue;
  }

  const pElem = Number(p.elem_per_token_per_layer);
  if (pElem !== c.elem) {
    console.error(`MISMATCH ${c.config} ${c.variant} elem ${c.elem} vs ${pElem}`);
    failures++;
  }

  const pLayers = Number(p.layers);
  if (pLayers !== c.layers) {
    console.error(`MISMATCH ${c.config} ${c.variant} layers ${c.layers} vs ${pLayers}`);
    failures++;
  }

  const pReduction = Number(p.reduction_vs_mha);
  if (Math.abs(pReduction - c.reduction) > 1e-9) {
    console.error(`MISMATCH ${c.config} ${c.variant} reduction ${c.reduction} vs ${pReduction}`);
    failures++;
  }

  const pGb = Number(p.gb_at_128k_fp16);
  if (Math.abs(pGb - c.gb) > 1e-9) {
    console.error(`MISMATCH ${c.config} ${c.variant} gb ${c.gb} vs ${pGb}`);
    failures++;
  }
}

console.log(`rows ${computed.length} compared ${computed.length}`);

if (failures > 0) {
  console.error(`\n${failures} mismatches`);
  process.exit(1);
}

console.log("JS reproduces every row of results/cache.csv from the shape parameters");
