# Methods and detail

Long form detail moved out of the README.


## Quality at matched budget, and why this part is weak


Six variants, same depth, width, data, batch, steps and seed, only the attention
module changing. Character level Shakespeare, 4 layers, 192 wide, 1500 steps,
3 seeds each. 32 minutes on a laptop CPU.

| variant | cache/token | vs MHA | val perplexity | range over seeds |
|---|---:|---:|---:|---|
| MHA | 1536 | 1.0x | 4.566 | 4.504 to 4.577 |
| GQA(g=2) | 768 | 2.0x | 4.500 | 4.500 to 4.556 |
| GQA(g=3) | 512 | 3.0x | 4.523 | 4.522 to 4.556 |
| MQA | 256 | 6.0x | 4.538 | 4.521 to 4.547 |
| MLA(d_c=48) | 256 | 6.0x | 4.563 | 4.520 to 4.600 |
| MLA(d_c=96) | 448 | 3.4x | 4.581 | 4.514 to 4.591 |

![quality against cache budget](../results/quality.png)

**Nothing is separated.** The spread between variant medians is 0.081. The mean
spread between seeds of the same variant is 0.058. The ratio is 1.41, which is
not enough to call anything. Every one of the 18 runs lands between 4.500 and
4.600.

At matched budget, MQA scores 4.538 and MLA 4.563, a difference of 0.025 while
MLA's own seeds span 0.080. So the comparison the whole repo exists to make
comes out as a shrug.

![validation curves](../results/curves.png)

I am not going to dress this up. Reading it as "cache design does not matter"
would be wrong in the other direction: the model is 1.7M parameters on 1M
characters of one play, and the differences between attention variants are known
to appear with scale and long context, neither of which this has. MHA has six
times the cache of MQA here and cannot beat it either, which is the clearest
sign that the experiment, not the method, is the limiting factor.

What it would take to make this comparison real: a model large enough that the
KV bottleneck binds, a context long enough that retrieval over distance matters,
and enough steps that 0.02 perplexity is outside seed noise. That is a GPU
project, and this machine does not have one.


## What I got wrong


**I assumed the quality experiment would show something.** I built the whole
matched budget comparison before checking whether the setup had the resolution
to detect the effect. The right order was to estimate seed noise first, then
decide how large the model had to be for the expected difference to clear it.
Instead I have 32 minutes of CPU time and a table where the error bars swallow
the signal. The cache accounting, which cost about ten minutes and is exact, is
the part worth reading.

**I nearly shipped absorption without the equality test.** `AbsorbedMLA` produced
sensible looking output from the start. It was only when I asserted it against
the naive path that I had any reason to believe the einsum index order was
right, and the first version was not.
