// Structural validation of every committed results file, plus two recomputes.
//
// The figures in the README are produced by pandas-free Python that also writes
// the CSVs, so a file that is ragged, has a duplicated column, or carries a NaN
// would still plot and still print. This walks the files and rejects that, then
// recomputes two things the Python asserted about itself:
//
//   1. val_ppl in results/quality.csv must be exp(val_loss).
//   2. cache_per_token in results/quality.csv must follow from the shape in
//      results/train-meta.json: 2 * n_kv * d_head * layers for the baselines,
//      (d_c + d_rope) * layers for MLA. That links the trained runs back to the
//      cache accounting rather than trusting the model to report itself.
//
// It also checks the agreement the README quotes for absorbed against naive is
// the number verify/golden/mla_golden.txt actually holds.
//
//	cd verify/gocheck && go run . -root ../..
package main

import (
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

type problem struct{ where, what string }

var problems []problem

func fail(where, format string, a ...any) {
	problems = append(problems, problem{where, fmt.Sprintf(format, a...)})
}

func readCSV(path string) ([]string, [][]string, error) {
	fh, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer fh.Close()
	r := csv.NewReader(fh)
	r.FieldsPerRecord = -1 // check widths here rather than letting the reader hide them
	recs, err := r.ReadAll()
	if err != nil {
		return nil, nil, err
	}
	if len(recs) < 2 {
		return nil, nil, fmt.Errorf("fewer than two lines")
	}
	return recs[0], recs[1:], nil
}

// Every field that parses as a number must be finite, no column may repeat, and
// every row must be as wide as the header.
func structural(name string, header []string, rows [][]string) {
	seen := map[string]bool{}
	for _, h := range header {
		if h == "" {
			fail(name, "empty column name in header")
		}
		if seen[h] {
			fail(name, "duplicate column %q", h)
		}
		seen[h] = true
	}
	for i, row := range rows {
		if len(row) != len(header) {
			fail(name, "row %d has %d fields, header has %d", i+2, len(row), len(header))
			continue
		}
		for j, f := range row {
			if strings.TrimSpace(f) == "" {
				fail(name, "row %d column %s is empty", i+2, header[j])
				continue
			}
			if v, err := strconv.ParseFloat(f, 64); err == nil {
				if math.IsNaN(v) || math.IsInf(v, 0) {
					fail(name, "row %d column %s is %s", i+2, header[j], f)
				}
			} else if strings.EqualFold(f, "nan") || strings.EqualFold(f, "inf") ||
				strings.EqualFold(f, "-inf") {
				fail(name, "row %d column %s is %s", i+2, header[j], f)
			}
		}
	}
}

func col(header []string, name string) int {
	for i, h := range header {
		if h == name {
			return i
		}
	}
	return -1
}

func num(name string, header, row []string, c string, rowNo int) float64 {
	i := col(header, c)
	if i < 0 {
		fail(name, "no column %s", c)
		return math.NaN()
	}
	v, err := strconv.ParseFloat(row[i], 64)
	if err != nil {
		fail(name, "row %d column %s is not a number: %s", rowNo, c, row[i])
		return math.NaN()
	}
	return v
}

type meta struct {
	Steps     int   `json:"steps"`
	DModel    int   `json:"d_model"`
	Layers    int   `json:"layers"`
	Heads     int   `json:"heads"`
	EvalEvery int   `json:"eval_every"`
	Seeds     []int `json:"seeds"`
}

// Cache elements per token for the whole model, from the shape alone.
func expectedCache(variant string, m meta) (int, bool) {
	dHead := m.DModel / m.Heads
	switch {
	case variant == "MHA":
		return 2 * m.Heads * dHead * m.Layers, true
	case variant == "MQA":
		return 2 * 1 * dHead * m.Layers, true
	case strings.HasPrefix(variant, "GQA(g="):
		var g int
		if _, err := fmt.Sscanf(variant, "GQA(g=%d)", &g); err != nil || g == 0 {
			return 0, false
		}
		return 2 * (m.Heads / g) * dHead * m.Layers, true
	case strings.HasPrefix(variant, "MLA(d_c="):
		var dc int
		if _, err := fmt.Sscanf(variant, "MLA(d_c=%d)", &dc); err != nil {
			return 0, false
		}
		return (dc + 16) * m.Layers, true // d_rope = 16 for both MLA runs, see train/run.py
	}
	return 0, false
}

func main() {
	root := flag.String("root", "../..", "repository root")
	flag.Parse()
	res := filepath.Join(*root, "results")

	files, err := filepath.Glob(filepath.Join(res, "*.csv"))
	if err != nil || len(files) == 0 {
		fmt.Println("no results CSVs found")
		os.Exit(1)
	}
	parsed := map[string]struct {
		header []string
		rows   [][]string
	}{}
	for _, f := range files {
		name := filepath.Base(f)
		h, rows, err := readCSV(f)
		if err != nil {
			fail(name, "unreadable: %v", err)
			continue
		}
		structural(name, h, rows)
		parsed[name] = struct {
			header []string
			rows   [][]string
		}{h, rows}
		fmt.Printf("  %-12s %d columns, %d rows\n", name, len(h), len(rows))
	}

	var m meta
	raw, err := os.ReadFile(filepath.Join(res, "train-meta.json"))
	if err != nil {
		fail("train-meta.json", "unreadable: %v", err)
	} else if err := json.Unmarshal(raw, &m); err != nil {
		fail("train-meta.json", "not valid JSON: %v", err)
	}

	// --- quality.csv --------------------------------------------------------
	q, ok := parsed["quality.csv"]
	if !ok {
		fail("quality.csv", "missing")
	} else {
		variants := map[string]int{}
		worstPpl := 0.0
		for i, row := range q.rows {
			v := row[col(q.header, "variant")]
			variants[v]++
			loss := num("quality.csv", q.header, row, "val_loss", i+2)
			ppl := num("quality.csv", q.header, row, "val_ppl", i+2)
			if d := math.Abs(math.Exp(loss) - ppl); d > 1e-9*math.Max(1, ppl) {
				fail("quality.csv", "row %d: exp(%v) = %v, file says %v", i+2,
					loss, math.Exp(loss), ppl)
			} else if d > worstPpl {
				worstPpl = d
			}
			cache := int(num("quality.csv", q.header, row, "cache_per_token", i+2))
			if want, ok := expectedCache(v, m); !ok {
				fail("quality.csv", "row %d: unknown variant %q", i+2, v)
			} else if want != cache {
				fail("quality.csv", "row %d: %s cache/token should be %d, file says %d",
					i+2, v, want, cache)
			}
		}
		if len(q.rows) != len(variants)*len(m.Seeds) {
			fail("quality.csv", "%d rows for %d variants and %d seeds",
				len(q.rows), len(variants), len(m.Seeds))
		}
		for v, n := range variants {
			if n != len(m.Seeds) {
				fail("quality.csv", "%s has %d rows, expected one per seed (%d)",
					v, n, len(m.Seeds))
			}
		}
		fmt.Printf("  quality.csv: val_ppl = exp(val_loss) to %.1e, cache/token "+
			"reproduced from the shape in train-meta.json for all %d rows\n",
			worstPpl, len(q.rows))
	}

	// --- curves.csv ---------------------------------------------------------
	c, ok := parsed["curves.csv"]
	if !ok {
		fail("curves.csv", "missing")
	} else {
		evals := 0
		for s := 0; s < m.Steps; s++ {
			if s%m.EvalEvery == 0 || s == m.Steps-1 {
				evals++
			}
		}
		runs := map[string]int{}
		for _, row := range c.rows {
			runs[row[col(c.header, "variant")]+" seed "+row[col(c.header, "seed")]]++
		}
		want := 0
		if q, ok := parsed["quality.csv"]; ok {
			want = len(q.rows)
		}
		if len(runs) != want {
			fail("curves.csv", "%d runs, quality.csv has %d", len(runs), want)
		}
		for r, n := range runs {
			if n != evals {
				fail("curves.csv", "%s has %d points, %d steps every %d gives %d",
					r, n, m.Steps, m.EvalEvery, evals)
			}
		}
		fmt.Printf("  curves.csv: %d runs of %d evaluations, which is what %d steps "+
			"every %d gives\n", len(runs), evals, m.Steps, m.EvalEvery)
	}

	// --- the agreement the README quotes ------------------------------------
	gold, err := os.ReadFile(filepath.Join(*root, "verify", "golden", "mla_golden.txt"))
	if err != nil {
		fail("mla_golden.txt", "unreadable: %v", err)
	} else {
		re := regexp.MustCompile(`scalar torch_grid_max_abs_diff ([0-9.eE+-]+)`)
		mm := re.FindSubmatch(gold)
		if mm == nil {
			fail("mla_golden.txt", "no torch_grid_max_abs_diff scalar")
		} else {
			v, _ := strconv.ParseFloat(string(mm[1]), 64)
			quoted := fmt.Sprintf("%.1e", v) // e.g. 8.3e-07
			readme, err := os.ReadFile(filepath.Join(*root, "README.md"))
			if err != nil {
				fail("README.md", "unreadable: %v", err)
			} else if !strings.Contains(string(readme), quoted) {
				fail("README.md", "quotes no agreement of %s for absorbed against naive",
					quoted)
			} else {
				fmt.Printf("  README quotes %s for absorbed against naive, which is "+
					"what the golden file holds\n", quoted)
			}
		}
	}

	if len(problems) > 0 {
		fmt.Printf("\n%d problems:\n", len(problems))
		for _, p := range problems {
			fmt.Printf("  %s: %s\n", p.where, p.what)
		}
		os.Exit(1)
	}
	fmt.Println("ok: results files are well formed and agree with each other")
}
