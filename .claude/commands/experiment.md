---
description: Start a research experiment with a pre-registered entry
argument-hint: <hypothesis>
---
Before running anything, append to `docs/research/EXPERIMENTS.md` an entry with: id (EXP-NNN), hypothesis ($ARGUMENTS),
metric(s), baseline, what will be varied (one variable), seeds, expected effect size, and the stopping rule. Commit it,
then run the experiment, store raw results under `benchmarks/` or `docs/research/data/`, and append the outcome
(including a negative result) with commit hash and command line. Report effect sizes with spread, not single numbers.
