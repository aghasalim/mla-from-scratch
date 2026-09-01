#!/usr/bin/env bash
# Recompute the published numbers in other languages and require agreement.
#
# Everything this repository publishes came out of one implementation. The cache
# table came out of bench/cache.py, the perplexity statistics out of one pass
# over results/quality.csv, and the claim that absorbed MLA equals the naive
# form out of the same PyTorch that produces both. If any of those were wrong,
# nothing downstream would notice, because everything downstream reads the same
# output. These are independent implementations, and a mistake would have to be
# made identically in all of them to survive.
#
# Each check is skipped with a clear message if its toolchain is absent, so this
# runs on a laptop with only some of them. CI has all of them.
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

pass=0 fail=0 skip=0

run () {
    local name="$1" tool="$2"; shift 2
    printf '\n=== %s ===\n' "$name"
    if ! command -v "$tool" >/dev/null 2>&1; then
        printf 'skipped: %s is not installed\n' "$tool"
        skip=$((skip + 1)); return
    fi
    if "$@"; then pass=$((pass + 1)); else fail=$((fail + 1)); fi
}

# SQL prints only what disagrees, so the shell decides. stdin is redirected
# because sqlite3 reads it, and CRLF is stripped because its CSV output has it.
check_sql () {
    local out
    out=$(sqlite3 -init verify/cache.sql :memory: "" < /dev/null 2>&1 | tr -d '\r')
    printf '%s\n' "$out"
    if printf '%s\n' "$out" | grep -qE '^(MISMATCH|MISSING FROM FILE|EXTRA IN FILE)'; then
        echo "SQL disagrees with results/cache.csv"
        return 1
    fi
    if ! printf '%s\n' "$out" | grep -qE '^rows ([1-9][0-9]*) compared \1$'; then
        echo "SQL did not compare every row of results/cache.csv"
        return 1
    fi
    echo "SQL reproduces every row of results/cache.csv from the shape parameters"
}

check_c () {
    cc -std=c99 -O2 -Wall -Wextra -Wpedantic -Werror \
       -o "${TMPDIR:-/tmp}/mla_kernel" verify/mla_kernel.c -lm || return 1
    "${TMPDIR:-/tmp}/mla_kernel" "$root"
}

check_go () { ( cd verify/gocheck && go run . -root "$root" ); }

check_rust () { ( cd verify/absorption && cargo run --release --quiet -- "$root" ); }

run "SQL, cache accounting"     sqlite3 check_sql
run "C, MLA kernel"             cc      check_c
run "Go, file validation"       go      check_go
run "R, quality statistics"     Rscript Rscript verify/verify.R "$root"
run "Rust, absorption identity" cargo   check_rust

printf '\n%s\n' "----------------------------------------"
printf '%d passed, %d failed, %d skipped\n' "$pass" "$fail" "$skip"
[ "$fail" -eq 0 ] || exit 1
[ "$pass" -gt 0 ] || { echo "nothing ran"; exit 1; }
