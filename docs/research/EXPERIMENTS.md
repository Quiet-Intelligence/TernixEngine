# Experiment log

Protocol (see CLAUDE.md): hypothesis, metric, baseline, single varied factor, seeds and stopping rule are written BEFORE the
run. Negative results stay in the log. Each entry records the commit and the raw-output path.

Machine for all entries unless stated: i7-14650HX, 24 GB single-channel DDR5-5600, RTX 5060 Laptop 8 GB, Windows 11, MSVC 19.5 /
CUDA 13.3, model BitNet b1.58 2B4T (revision 276681394656abdadb8e80e5b2c3db5e5d7fcaff).

---

## EXP-001  Numerical noise floor of W1.58A8 logits

Status: registered 2026-10-26, before running.

Question: how much do final logits move when the computation is perturbed at the level of float32 rounding only? This is the
yardstick for every "engine A vs engine B" logit comparison (our CPU vs GPU, ours vs PyTorch, bf16 vs fp32).

Hypothesis H1: per-token absmax int8 activation quantisation turns last-bit float differences into integer rounding flips that
propagate through 30 layers, so a perturbation of relative size 2^-24 on the embeddings produces a logit KL of the same order
(1e-4 to 1e-2) as the engine-vs-reference differences measured so far (KL 1e-4 .. 1e-2), i.e. those differences are
irreducible noise and not an engine bug. Falsified if KL(ref || ref+noise) < 1e-5 on every prompt.

Metric: KL(p_ref || p_perturbed) of last-token distributions, max |dlogit|, top-1 agreement.
Baseline: the float32 reference (`scripts/reference_bitnet.py`) with no perturbation.
Varied factor: relative noise amplitude added to the embedded input, {0, 2^-24, 2^-16 (bf16-level 2^-8 excluded)}, 3 noise seeds.
Prompts: the five prompts in `scripts/verify_logits.py`.
Stopping rule: fixed grid, no early stopping.

Outcome (2026-10-26, commit d264a0d + uncommitted harness, raw data `docs/research/data/exp001_noise_floor.json`):

| perturbation of the embedded input | median KL | min .. max KL | top-1 flips |
|---|---:|---|---:|
| relative 2^-24 (float32 rounding level), 3 seeds x 5 prompts | 9.6e-4 | 1e-7 .. 7.7e-3 | 0 / 15 |
| relative 2^-16, 3 seeds x 5 prompts | 2.4e-3 | 4.8e-5 .. 1.2e-2 | 0 / 15 |

| comparison on the same 5 prompts | median KL | max KL | top-1 equal |
|---|---:|---:|---:|
| CPU engine vs float32 reference | 5.3e-3 | 8.1e-3 | 5 / 5 |
| CUDA engine vs float32 reference | 3.3e-3 | 1.9e-2 | 5 / 5 |
| CPU engine vs CUDA engine | 2.3e-3 | 7.3e-3 | 5 / 5 |

H1 is supported: a float32-rounding-sized perturbation already produces logit KL of 1e-4 to 8e-3, because per-token int8 activation
quantisation turns last-bit differences into integer rounding flips that survive 30 layers. The engine-vs-reference differences
are inside that band (the worst, 1.9e-2, is 1.6x the largest perturbation KL, so a small extra contribution from the fp16 KV
cache and float summation order is not excluded). Consequence for the project: logit-level "bit exactness" against a float
reference is not a meaningful gate for W1.58A8 models. Correctness gates are (a) exact integer-kernel tests, (b) top-1 agreement
and KL below the measured noise band, (c) perplexity within the run-to-run noise on a fixed text. Not run: a separate fp32-KV
engine variant to isolate the KV contribution (candidate for a later entry).
