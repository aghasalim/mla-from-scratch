/* Reimplementation of the MLA kernel in C, checked against golden vectors.
 *
 * verify/golden/mla_golden.txt holds the weights, the input, and the two output
 * tensors PyTorch produced (mla/naive.py and mla/absorbed.py). This file reads
 * the weights and the input only, recomputes both forward passes from the
 * arithmetic in the paper, and requires the results to match the golden
 * outputs. Nothing here shares a line of code with the Python, so a wrong
 * einsum index order or a transposed weight in mla/ would show up as a
 * disagreement rather than as plausible looking attention.
 *
 * Tensors are looked up BY NAME, so reordering the golden file cannot silently
 * feed the wrong matrix into a slot.
 *
 *   cc -std=c99 -O2 -Wall -Wextra -Wpedantic -Werror -o mla_kernel verify/mla_kernel.c -lm
 *   ./mla_kernel <repo root>
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_TENSORS 32
#define TOL 1e-5

typedef struct {
    char name[64];
    int ndim, dim[4];
    long n;
    double *v;
} Tensor;

static Tensor tensors[MAX_TENSORS];
static int n_tensors = 0;
static struct { char name[64]; long v; } cfgs[32];
static int n_cfgs = 0;
static struct { char name[64]; double v; } scalars[32];
static int n_scalars = 0;

static void die(const char *msg, const char *what) {
    fprintf(stderr, "mla_kernel: %s %s\n", msg, what ? what : "");
    exit(1);
}

static long cfg(const char *name) {
    for (int i = 0; i < n_cfgs; i++)
        if (!strcmp(cfgs[i].name, name)) return cfgs[i].v;
    die("missing config", name);
    return 0;
}

static double scal(const char *name) {
    for (int i = 0; i < n_scalars; i++)
        if (!strcmp(scalars[i].name, name)) return scalars[i].v;
    die("missing scalar", name);
    return 0;
}

static Tensor *tens(const char *name, long expect) {
    for (int i = 0; i < n_tensors; i++)
        if (!strcmp(tensors[i].name, name)) {
            if (tensors[i].n != expect) die("wrong element count for tensor", name);
            return &tensors[i];
        }
    die("missing tensor", name);
    return NULL;
}

static void load(const char *path) {
    FILE *fh = fopen(path, "r");
    if (!fh) die("cannot open", path);
    char line[512], kind[32], name[64];
    while (fgets(line, sizeof line, fh)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        if (sscanf(line, "%31s", kind) != 1) continue;
        if (!strcmp(kind, "config")) {
            long v;
            if (sscanf(line, "%*s %63s %ld", name, &v) != 2) die("bad config line", line);
            strcpy(cfgs[n_cfgs].name, name); cfgs[n_cfgs++].v = v;
        } else if (!strcmp(kind, "scalar")) {
            double v;
            if (sscanf(line, "%*s %63s %lf", name, &v) != 2) die("bad scalar line", line);
            strcpy(scalars[n_scalars].name, name); scalars[n_scalars++].v = v;
        } else if (!strcmp(kind, "tensor")) {
            if (n_tensors == MAX_TENSORS) die("too many tensors", NULL);
            Tensor *t = &tensors[n_tensors++];
            int off = 0;
            if (sscanf(line, "%*s %63s %d%n", t->name, &t->ndim, &off) != 2)
                die("bad tensor line", line);
            if (t->ndim < 1 || t->ndim > 4) die("bad tensor rank", t->name);
            t->n = 1;
            for (int d = 0; d < t->ndim; d++) {
                int adv = 0;
                if (sscanf(line + off, "%d%n", &t->dim[d], &adv) != 1) die("bad dims", t->name);
                off += adv; t->n *= t->dim[d];
            }
            t->v = malloc((size_t)t->n * sizeof(double));
            if (!t->v) die("out of memory", t->name);
            for (long i = 0; i < t->n; i++)
                if (fscanf(fh, "%lf", &t->v[i]) != 1) die("short tensor", t->name);
            if (fgets(line, sizeof line, fh) == NULL && !feof(fh)) die("read error", t->name);
        }
    }
    fclose(fh);
}

/* y = W x, W is (out, in) row major, as PyTorch stores nn.Linear weights. */
static void matvec(const double *W, const double *x, double *y, int out, int in) {
    for (int o = 0; o < out; o++) {
        double s = 0.0;
        for (int i = 0; i < in; i++) s += W[(long)o * in + i] * x[i];
        y[o] = s;
    }
}

/* RoPE on one vector of width d at position pos, pairs (2i, 2i+1). */
static void rope(double *v, int d, int pos, double theta) {
    for (int i = 0; i < d / 2; i++) {
        double f = 1.0 / pow(theta, (double)(2 * i) / (double)d);
        double a = (double)pos * f, cs = cos(a), sn = sin(a);
        double re = v[2 * i], im = v[2 * i + 1];
        v[2 * i] = re * cs - im * sn;
        v[2 * i + 1] = re * sn + im * cs;
    }
}

static void softmax_row(double *p, int n) {
    double m = -INFINITY, s = 0.0;
    for (int i = 0; i < n; i++) if (p[i] > m) m = p[i];
    for (int i = 0; i < n; i++) { p[i] = exp(p[i] - m); s += p[i]; }
    for (int i = 0; i < n; i++) p[i] /= s;
}

int main(int argc, char **argv) {
    char path[1024];
    snprintf(path, sizeof path, "%s/verify/golden/mla_golden.txt",
             argc > 1 ? argv[1] : ".");
    load(path);

    const int dm = (int)cfg("d_model"), nh = (int)cfg("n_heads");
    const int dh = (int)cfg("d_head"), dc = (int)cfg("d_c");
    const int dr = (int)cfg("d_rope"), S = (int)cfg("seq"), B = (int)cfg("batch");
    const double theta = scal("rope_theta");
    const double scale = sqrt((double)(dh + dr));

    Tensor *x = tens("x", (long)B * S * dm);
    Tensor *w_dkv = tens("w_dkv", (long)dc * dm);
    Tensor *w_uk = tens("w_uk", (long)nh * dh * dc);
    Tensor *w_uv = tens("w_uv", (long)nh * dh * dc);
    Tensor *wq = tens("wq", (long)nh * dh * dm);
    Tensor *wq_rope = tens("wq_rope", (long)nh * dr * dm);
    Tensor *wk_rope = tens("wk_rope", (long)dr * dm);
    Tensor *wo = tens("wo", (long)dm * nh * dh);
    Tensor *gold_naive = tens("out_naive", (long)B * S * dm);
    Tensor *gold_abs = tens("out_absorbed", (long)B * S * dm);

    double *c = malloc((size_t)S * dc * sizeof(double));
    double *k = malloc((size_t)S * nh * dh * sizeof(double));
    double *v = malloc((size_t)S * nh * dh * sizeof(double));
    double *q = malloc((size_t)S * nh * dh * sizeof(double));
    double *qr = malloc((size_t)S * nh * dr * sizeof(double));
    double *kr = malloc((size_t)S * dr * sizeof(double));
    double *att = malloc((size_t)S * sizeof(double));
    double *head_out = malloc((size_t)nh * dh * sizeof(double));
    double *ctx = malloc((size_t)nh * dc * sizeof(double));
    double *qlat = malloc((size_t)nh * dc * sizeof(double));
    double *out_n = malloc((size_t)B * S * dm * sizeof(double));
    double *out_a = malloc((size_t)B * S * dm * sizeof(double));
    /* absorbed weights, folded once: wq_abs (nh, dc, dm), wo_abs (dm, nh, dc) */
    double *wq_abs = calloc((size_t)nh * dc * dm, sizeof(double));
    double *wo_abs = calloc((size_t)dm * nh * dc, sizeof(double));
    if (!c || !k || !v || !q || !qr || !kr || !att || !head_out || !ctx || !qlat ||
        !out_n || !out_a || !wq_abs || !wo_abs) die("out of memory", NULL);

    for (int h = 0; h < nh; h++)
        for (int cc = 0; cc < dc; cc++)
            for (int m = 0; m < dm; m++) {
                double s = 0.0;
                for (int d = 0; d < dh; d++)
                    s += w_uk->v[(long)(h * dh + d) * dc + cc] * wq->v[(long)(h * dh + d) * dm + m];
                wq_abs[((long)h * dc + cc) * dm + m] = s;
            }
    for (int m = 0; m < dm; m++)
        for (int h = 0; h < nh; h++)
            for (int cc = 0; cc < dc; cc++) {
                double s = 0.0;
                for (int d = 0; d < dh; d++)
                    s += wo->v[(long)m * nh * dh + h * dh + d] *
                         w_uv->v[(long)(h * dh + d) * dc + cc];
                wo_abs[((long)m * nh + h) * dc + cc] = s;
            }

    for (int b = 0; b < B; b++) {
        const double *xb = x->v + (long)b * S * dm;
        for (int t = 0; t < S; t++) {
            matvec(w_dkv->v, xb + (long)t * dm, c + (long)t * dc, dc, dm);
            matvec(w_uk->v, c + (long)t * dc, k + (long)t * nh * dh, nh * dh, dc);
            matvec(w_uv->v, c + (long)t * dc, v + (long)t * nh * dh, nh * dh, dc);
            matvec(wq->v, xb + (long)t * dm, q + (long)t * nh * dh, nh * dh, dm);
            matvec(wq_rope->v, xb + (long)t * dm, qr + (long)t * nh * dr, nh * dr, dm);
            matvec(wk_rope->v, xb + (long)t * dm, kr + (long)t * dr, dr, dm);
            for (int h = 0; h < nh; h++) rope(qr + ((long)t * nh + h) * dr, dr, t, theta);
            rope(kr + (long)t * dr, dr, t, theta);
        }

        /* naive: keys and values materialised */
        for (int s = 0; s < S; s++) {
            for (int h = 0; h < nh; h++) {
                for (int t = 0; t <= s; t++) {
                    double sc = 0.0;
                    for (int d = 0; d < dh; d++)
                        sc += q[((long)s * nh + h) * dh + d] * k[((long)t * nh + h) * dh + d];
                    for (int d = 0; d < dr; d++)
                        sc += qr[((long)s * nh + h) * dr + d] * kr[(long)t * dr + d];
                    att[t] = sc / scale;
                }
                softmax_row(att, s + 1);
                for (int d = 0; d < dh; d++) {
                    double o = 0.0;
                    for (int t = 0; t <= s; t++) o += att[t] * v[((long)t * nh + h) * dh + d];
                    head_out[h * dh + d] = o;
                }
            }
            matvec(wo->v, head_out, out_n + ((long)b * S + s) * dm, dm, nh * dh);
        }

        /* absorbed: query folded into the latent space, values folded into W^O */
        for (int s = 0; s < S; s++) {
            for (int h = 0; h < nh; h++)
                matvec(wq_abs + (long)h * dc * dm, xb + (long)s * dm, qlat + (long)h * dc, dc, dm);
            for (int h = 0; h < nh; h++) {
                for (int t = 0; t <= s; t++) {
                    double sc = 0.0;
                    for (int cc = 0; cc < dc; cc++)
                        sc += qlat[(long)h * dc + cc] * c[(long)t * dc + cc];
                    for (int d = 0; d < dr; d++)
                        sc += qr[((long)s * nh + h) * dr + d] * kr[(long)t * dr + d];
                    att[t] = sc / scale;
                }
                softmax_row(att, s + 1);
                for (int cc = 0; cc < dc; cc++) {
                    double o = 0.0;
                    for (int t = 0; t <= s; t++) o += att[t] * c[(long)t * dc + cc];
                    ctx[(long)h * dc + cc] = o;
                }
            }
            for (int m = 0; m < dm; m++) {
                double o = 0.0;
                for (int h = 0; h < nh; h++)
                    for (int cc = 0; cc < dc; cc++)
                        o += ctx[(long)h * dc + cc] * wo_abs[((long)m * nh + h) * dc + cc];
                out_a[((long)b * S + s) * dm + m] = o;
            }
        }
    }

    double dn = 0.0, da = 0.0, self = 0.0;
    for (long i = 0; i < (long)B * S * dm; i++) {
        double a = fabs(out_n[i] - gold_naive->v[i]);
        double bdif = fabs(out_a[i] - gold_abs->v[i]);
        double sf = fabs(out_n[i] - out_a[i]);
        if (a > dn) dn = a;
        if (bdif > da) da = bdif;
        if (sf > self) self = sf;
    }

    printf("C, %d values per output tensor, batch %d seq %d d_model %d\n",
           B * S * dm, B, S, dm);
    printf("  C naive     vs golden naive     max abs diff %.3e\n", dn);
    printf("  C absorbed  vs golden absorbed  max abs diff %.3e\n", da);
    printf("  C naive     vs C absorbed       max abs diff %.3e\n", self);
    printf("  torch naive vs torch absorbed   max abs diff %.3e (from the golden file)\n",
           scal("torch_naive_vs_absorbed"));
    int bad = (dn > TOL) || (da > TOL) || (self > TOL);
    printf("%s: tolerance %.0e\n", bad ? "FAIL" : "ok", TOL);
    return bad;
}
