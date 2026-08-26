# Task NN — <title>

**Wave:** N (serial | parallel with NN, NN)
**OWNS:** paths this task may create or edit
**READS:** paths it may read but must not edit

## Context

Why this task exists, what it depends on, what the reader should have understood before starting. Two or three sentences. If a previous task produced a finding this one relies on, name the file.

## Task

Numbered deliverables, each a concrete file. For each: what it contains, the interface, the specific technique. Include the actual math where math is involved — an agent given a formula produces better code than an agent given a description of a formula.

## Acceptance criteria

Checkable statements. "Matches the fp64 reference within the relative bar," not "works correctly." "Under 2× the reference implementation's latency," not "is fast." If you can't check it, it isn't an acceptance criterion.

## Gotchas

The bugs you'd hit. This section is where domain knowledge lives and it's the highest-value part of the spec. Prefer specific failure symptoms — "if the error sits at 1e-2 and won't improve, check X" — over general warnings.

## Finish by

The LOGBOOK entry: which numbers this task must record for later tasks to use.
