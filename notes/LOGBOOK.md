# Logbook

## 2026-08-26, absorbed inference agrees with the naive form to 6e-07
**Tried:** implemented MLA twice. `naive.py` reconstructs K and V from the latent so the maths is readable; `absorbed.py` folds W_UK into W_Q and W_UV into W_O so neither is ever materialised. Then asserted they agree across d_c in {48, 96, 192} and sequence lengths {1, 7, 48}.
**Measured:** max absolute difference 5.96e-07, relative 8.77e-07.
**Concluded:** the folding is right. Worth noting how weak the alternative evidence was: the absorbed version produced sensible looking attention output from the first run, and the first einsum index order I wrote was wrong. Nothing except the equality assertion could have told me. Also added a hard refusal: constructing AbsorbedMLA from a model without decoupled RoPE raises, because a position dependent rotation between the query and W_UK cannot be folded into a fixed matrix and silently absorbing it would produce a plausible wrong answer.

## 2026-08-26, the cache accounting is the strong result, and it is free
**Tried:** exact element counts for MHA, GQA, MQA and MLA at DeepSeek V2 shape (n_h=32, d_h=128, d_c=512, d_rope=64, 60 layers).
**Measured:** per token per layer, MHA 8192 and MLA 576, a 14.22x reduction. At 128k context in fp16 that is 120.00 GB against 8.44 GB. Decoupled RoPE costs 64 of MLA's 576 elements, so 0.94 GB of the 8.44.
**Concluded:** arithmetic, no error bars, took about ten minutes to write. It is also the most useful thing in the repo, which is worth remembering next time I plan an experiment: MQA is smaller still at 256 elements and 3.75 GB, so MLA only earns its complexity if it holds quality better at a matched budget. That is what the training run was meant to settle.

## 2026-08-26, the quality comparison is a null and the experiment is why
**Tried:** six variants, identical depth width data batch steps and seed, only the attention module changing. Char level Shakespeare, 4 layers, 192 wide, 1500 steps, 3 seeds each. 1904 s on an M4 CPU.
**Measured:** every one of the 18 runs landed between 4.500 and 4.600 validation perplexity. Spread between variant medians 0.081; mean spread between seeds of the same variant 0.058; ratio 1.41. At matched budget (256 elements/token) MQA scored 4.538 and MLA 4.563, a gap of 0.025 while MLA's own seeds spanned 0.080.
**Concluded:** nothing is separated and I should have seen this coming. The tell is that MHA has six times the cache of MQA and cannot beat it either, which points at the experiment rather than the methods. A 1.7M parameter model on 1M characters at 128 context has no KV bottleneck to relieve, so there is nothing for cache design to trade against. The mistake was building the comparison before estimating seed noise and asking how big the model needed to be for the expected effect to clear it. Doing that first would have cost ten minutes and saved thirty two. Reporting it as a null rather than picking the seed where MLA wins.
