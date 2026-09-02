# Recompute the quality statistics from results/quality.csv in Ruby.
#
# The R verifier already does this, but the point of each language is an
# independent reimplementation: a bug would have to appear identically in both
# to survive. No gems, just stdlib CSV.

require "csv"

root = ARGV[0] || "."

rows = CSV.read(File.join(root, "results", "quality.csv"), headers: true)
abort "expected 18 rows, got #{rows.size}" unless rows.size == 18

# --- val_ppl must equal exp(val_loss) for every row --------------------------
worst_ppl_diff = 0.0
rows.each_with_index do |row, i|
  loss = row["val_loss"].to_f
  ppl  = row["val_ppl"].to_f
  diff = (Math.exp(loss) - ppl).abs
  if diff > 1e-9 * [1.0, ppl].max
    $stderr.puts "row #{i + 2}: exp(#{loss}) = #{Math.exp(loss)}, file says #{ppl}"
    exit 1
  end
  worst_ppl_diff = [worst_ppl_diff, diff].max
end
puts "val_ppl = exp(val_loss) to %.1e for all #{rows.size} rows" % worst_ppl_diff

# --- recompute the published claims ------------------------------------------
by_variant = rows.group_by { |r| r["variant"] }

medians = {}
spans   = {}
by_variant.each do |v, rs|
  ppls = rs.map { |r| r["val_ppl"].to_f }.sort
  medians[v] = ppls[ppls.size / 2]  # 3 seeds, median is the middle one
  spans[v]   = ppls.last - ppls.first
end

median_spread  = medians.values.max - medians.values.min
mean_seed_span = spans.values.sum / spans.size.to_f
ratio          = median_spread / mean_seed_span
mqa            = medians["MQA"]
mla            = medians["MLA(d_c=48)"]
gap            = (mqa - mla).abs
mla_span       = spans["MLA(d_c=48)"]
lowest         = rows.map { |r| r["val_ppl"].to_f }.min
highest        = rows.map { |r| r["val_ppl"].to_f }.max

puts "\nrecomputed from results/quality.csv"
by_variant.each do |v, _|
  printf "  %-12s median ppl %.3f   seeds span %.3f\n", v, medians[v], spans[v]
end

claims = [
  ["spread between variant medians", median_spread, 0.081, 5e-4],
  ["mean spread between seeds",      mean_seed_span, 0.058, 5e-4],
  ["ratio of the two",               ratio,          1.41,  5e-3],
  ["MQA median perplexity",          mqa,            4.538, 5e-4],
  ["MLA(d_c=48) median perplexity",  mla,            4.563, 5e-4],
  ["gap at the matched budget",      gap,            0.025, 5e-4],
  ["span of MLA's own three seeds",  mla_span,       0.080, 5e-4],
  ["lowest of the 18 runs",          lowest,         4.500, 5e-4],
  ["highest of the 18 runs",         highest,        4.600, 5e-4],
]

failures = 0
puts "\nagainst the published claims"
claims.each do |name, got, want, tol|
  d = (got - want).abs
  ok = d <= tol
  failures += 1 unless ok
  printf "  %-32s recomputed %7.3f  published %7.3f  |d| %.1e  %s\n",
         name, got, want, d, ok ? "ok" : "FAIL"
end

# --- permutation test --------------------------------------------------------
srand(20260901)
perms = 20_000
all_ppls  = rows.map { |r| r["val_ppl"].to_f }
variants  = by_variant.keys
group_sz  = rows.size / variants.size  # 3

perm_spreads = Array.new(perms) do
  shuffled = all_ppls.shuffle
  meds = variants.each_with_index.map do |_, vi|
    chunk = shuffled[vi * group_sz, group_sz].sort
    chunk[chunk.size / 2]
  end
  meds.max - meds.min
end

p_value = (1 + perm_spreads.count { |s| s >= median_spread }).to_f / (perms + 1)
printf "\npermutation test, %d relabellings of the same 18 runs\n", perms
printf "  observed spread between medians %.3f, permuted median %.3f\n",
       median_spread, perm_spreads.sort[perms / 2]
printf "  p = %.3f  (labels shuffled, so this is the spread noise alone produces)\n", p_value

if p_value < 0.05
  puts "  FAIL: the variants separate after all, which the write-up denies"
  failures += 1
end

if failures > 0
  puts "\n#{failures} disagreements"
  exit 1
end
puts "\nok: every quoted statistic reproduces from the CSV"
