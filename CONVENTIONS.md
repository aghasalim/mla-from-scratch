# CONVENTIONS.md — rules every agent follows, in every repo

Each task spec in `prompts/` assumes these. Don't restate them in prompts; do enforce them in review.

## Execution

1. **One task spec per agent.** Never paste two. The specs are sized to be a complete unit of work with a testable end state.
2. **Parallel agents get separate git worktrees.** Same checkout means agents overwrite each other:
   ```bash
   git worktree add ../repo-task-03 -b task/03
   ```
   Merge to `main` in wave order. You resolve conflicts, not the agent — it wasn't scoped for the merge.
3. **Respect `OWNS` / `READS`.** Every spec declares which files it may edit and which it may only read. Agents in the same wave have disjoint `OWNS` sets; that's the entire reason concurrency is safe.

## Correctness

4. **A reference implementation exists before the optimized one.** Slow, obvious, high-precision, no tricks. Everything is measured against it. Write it first even when it feels like a detour.
5. **Never loosen a tolerance to make a test pass.** If the fast path disagrees with the reference beyond the bar, the fast path is wrong. Fix it.
6. **Relative tolerance beats absolute.** The bar is "no worse than the naive implementation in the same precision," not a magic epsilon. Absolute thresholds either pass broken code or fail correct code.
7. **Test the boundaries.** Non-power-of-two sizes, prime lengths, size 1, empty inputs, and the largest thing that fits. Bugs live at the edges and nowhere else.

## Measurement

8. **No number that didn't come from a measurement.** Not from the paper, not estimated from the algorithm, not "roughly." Measured or absent. This rule is the whole reason the projects are credible.
9. **Report medians, with p10/p90.** One preemption spike ruins a mean.
10. **Warm up, then synchronize, then time.** GPU calls are async. Every benchmark that looks impossibly fast is measuring kernel launches.
11. **Fix seeds and record them.** Any result you can't reproduce is a result you don't have.
12. **Report variance, not just the point estimate.** Three seeds minimum for anything involving training. A single-seed improvement of 2% is noise and reviewers know it.

## Recording

13. **Every experiment that moves a number gets a `notes/LOGBOOK.md` entry.** Three lines: Tried / Measured / Concluded. Write the Concluded line even when it's "no idea why" — the unexplained results are the ones worth returning to.
14. **Negative results stay in.** "I tried X, it was 8% worse, here's the profiler evidence" is signal. An ablation table containing only wins is not believable.
15. **Say "not measured on this hardware"** rather than extrapolating. Nobody is fooled by a number for a GPU you don't own.

## Scope

16. **Don't add a feature the spec didn't ask for.** The waves are sequenced so that each task can be verified in isolation. An agent that helpfully implements the next task too has made both untestable.
17. **If a spec is wrong, say so and stop.** Better a paused task and a question than a confidently-built wrong thing.
