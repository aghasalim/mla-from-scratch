# The statistics in the quality section, recomputed in base R.
#
# Those numbers live nowhere but the prose. results/quality.csv holds 18
# perplexities; the spread between variant medians, the mean spread between
# seeds, and their ratio were all computed once, by hand, and written straight
# into the README and notes/METHODS.md. Nothing checked them afterwards. This
# recomputes each from the CSV and compares against what the documents claim.
#
# It also adds the test the write-up argues for in words but never ran: if the
# variant labels carry no information, how often does a random relabelling of
# the same 18 runs produce a spread between medians as large as the observed
# one? That p-value is the quantitative form of "nothing is separated".
#
# No packages, so CI needs nothing beyond r-base-core.

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) > 0) args[1] else "."
set.seed(20260901)

PERMS <- 20000
# Rounding tolerance: half a unit in the last decimal each claim is quoted to.

q <- read.csv(file.path(root, "results", "quality.csv"))
stopifnot(nrow(q) == 18)

variants <- unique(as.character(q$variant))
medians <- sapply(variants, function(v) median(q$val_ppl[q$variant == v]))
spans <- sapply(variants, function(v) diff(range(q$val_ppl[q$variant == v])))

median_spread <- max(medians) - min(medians)
mean_seed_span <- mean(spans)
ratio <- median_spread / mean_seed_span
mqa <- medians[["MQA"]]
mla <- medians[["MLA(d_c=48)"]]
gap <- abs(mqa - mla)
mla_span <- spans[["MLA(d_c=48)"]]

cat("recomputed from results/quality.csv\n")
for (v in variants) {
    cat(sprintf("  %-12s median ppl %.3f   seeds span %.3f\n", v, medians[[v]], spans[[v]]))
}

# The claims, the value each is quoted as, and the tolerance that rounding to
# the quoted number of decimals allows. Every one appears in README.md, in
# notes/METHODS.md, or in both.
claims <- list(
    list("spread between variant medians", median_spread, 0.081, 5e-4),
    list("mean spread between seeds",      mean_seed_span, 0.058, 5e-4),
    list("ratio of the two",               ratio,          1.41,  5e-3),
    list("MQA median perplexity",          mqa,            4.538, 5e-4),
    list("MLA(d_c=48) median perplexity",  mla,            4.563, 5e-4),
    list("gap at the matched budget",      gap,            0.025, 5e-4),
    list("span of MLA's own three seeds",  mla_span,       0.080, 5e-4),
    list("lowest of the 18 runs",          min(q$val_ppl), 4.500, 5e-4),
    list("highest of the 18 runs",         max(q$val_ppl), 4.600, 5e-4)
)

failures <- 0
cat("\nagainst the published claims\n")
for (cl in claims) {
    got <- cl[[2]]; want <- cl[[3]]
    d <- abs(got - want)
    ok <- d <= cl[[4]]
    failures <- failures + !ok
    cat(sprintf("  %-32s recomputed %7.3f  published %7.3f  |d| %.1e  %s\n",
                cl[[1]], got, want, d, if (ok) "ok" else "FAIL"))
}

# Permutation test: shuffle the variant labels, keep the 18 perplexities.
perm_spread <- replicate(PERMS, {
    lab <- sample(as.character(q$variant))
    m <- sapply(variants, function(v) median(q$val_ppl[lab == v]))
    max(m) - min(m)
})
p <- (1 + sum(perm_spread >= median_spread)) / (PERMS + 1)
cat(sprintf("\npermutation test, %d relabellings of the same 18 runs\n", PERMS))
cat(sprintf("  observed spread between medians %.3f, permuted median %.3f\n",
            median_spread, median(perm_spread)))
cat(sprintf("  p = %.3f  (labels shuffled, so this is the spread noise alone produces)\n", p))
if (p < 0.05) {
    cat("  FAIL: the variants separate after all, which the write-up denies\n")
    failures <- failures + 1
}

if (failures > 0) {
    cat(sprintf("\n%d disagreements\n", failures))
    quit(status = 1)
}
cat("\nok: every quoted statistic reproduces from the CSV\n")
