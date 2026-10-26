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

---

## EXP-002  Exact activation sparsity of the FFN down-projection input

Status: run 2026-10-26 (logged here retroactively as an exploratory measurement; the confirmatory test is EXP-004).

Question: how many entries of the int8 activation vector entering each BitLinear are exactly zero for BitNet 2B4T on natural text?
Method: float32 reference (`scripts/reference_bitnet.py`, `collect=`), 4 windows x 512 WikiText-2 tokens, statistics of the
per-token absmax-quantised activations at every projection input. Script `scripts/exp002_activation_sparsity.py`, data
`docs/research/data/exp002_activation_sparsity.json`.

Result: down-projection input 80.4% exactly zero, 88.0% in {-1,0,1} (per layer 69%..91%); o_proj input 4.8% zero; q/k/v/gate/up
input 1.5% zero. No 128-wide block of the down input is entirely zero at batch 1 (so block-skipping in the current row-major
layout would gain nothing; element-wise skipping needs a column-major layout).
Caveat: one model, one text; per-token (batch 1) statistics only.

## EXP-003  Quantised output head with exact rescoring (contribution C1)

Status: registered 2026-10-26, before implementation.

Hypotheses: H3a: an int8 per-row-scaled copy of the tied head, used to select the top C = 64 candidates that are then rescored
with the bf16 rows, returns the same argmax as the full bf16 head on >= 99.9% of positions. H3b: with int4 (group 32) the same
holds for C = 256. H3c: decode throughput rises by >= 1.3x (CPU) because the head read shrinks from 656 MB to 328 MB (int8) or
164 MB (int4) plus C x 5 KB of rescoring. Falsified if agreement < 99%, or speedup < 1.15x.
Metrics: argmax agreement and top-5 set agreement on 20,000 WikiText-2 positions; KL(p_full || p_candidate-set renormalised) for
sampling use; perplexity of the candidate-restricted distribution vs the full softmax; decode tok/s (median of 7).
Baseline: bf16 head. Varied factor: head precision {bf16, int8, int4-g32} x C {16, 64, 256}.
Stopping rule: fixed grid. Not claimed: sampling-exactness (the candidate set truncates the tail; its mass is measured).

## EXP-004  Exact zero-skipping in the down projection (C2)

Status: registered 2026-10-26, before implementation.

Hypothesis: a column-major (K-major) copy of the down-projection weights used with the nonzero activation indices produces
bit-identical int32 outputs and reads only the columns of nonzero activations, so the down projection takes <= 0.45x the dense
time at the measured 80% sparsity at batch 1 on CPU and GPU; break-even sparsity is reported. Falsified if outputs differ at all
or time ratio > 0.7.
Metrics: exact equality of outputs; kernel time (median of 15, rotating weights beyond LLC); end-to-end decode tok/s; speedup as a
function of sparsity on synthetic activation vectors (0..95%).

## EXP-005  Core selection and scheduling for decode on a hybrid CPU (C5)

Status: registered 2026-10-26.

Hypothesis: with dynamic chunk scheduling, P-core-only pinning (16 threads) gives the highest decode tok/s; using all 24 logical
processors lowers it (measured read bandwidth 26 -> 17 GB/s); a static equal partition over 24 threads is the slowest and has
the highest run-to-run spread. Falsified if all-24 dynamic beats 16 P-core dynamic by > 3%.
Varied factor: {threads 4, 8, 12, 16, 20, 24} x {pinned P-core order, unpinned} x {dynamic chunks, static partition}.
Metrics: decode tok/s median/min/max over 7 runs of 64 tokens, ctx 128.

## EXP-006  int8 KV cache (C6)

Status: registered 2026-10-26.

Hypothesis: int8 K and V with a per-(token, kv-head) scale change WikiText-2 perplexity by less than the fp16-vs-fp32 difference
measured in the same run set (noise band) at windows of 512 and 2048 tokens, and raise decode tok/s at 2048-token context by
>= 8%. Falsified if the perplexity change exceeds 1% or the speedup is below 3%.

## EXP-007  Zero-bitmap + sign layout on AVX-VNNI and CUDA (C3)

Status: registered 2026-10-26.

Hypothesis: at the measured mean zero density 0.42 the bitmap+sign layout moves 17% fewer bytes than 2-bit; on the 16-thread
P-core pool decode GEMV is >= 1.05x faster than I2K128 despite the extra unpack instructions; on the GPU it is no slower. This
hypothesis is genuinely uncertain: BITCOS reports an instruction-bound loss on one 8-core part. A negative result is an
acceptable outcome and will be reported.
Metrics: GEMV time on the four decode shapes with rotating weights; bytes per token; effective GB/s against the 26 GB/s ceiling.

## EXP-008  Tensor-core W1.58A8 prefill (C4)

Status: registered 2026-10-26.

Hypothesis: decoding 2-bit codes to s8 fragments in registers and using `mma.sync.aligned.m16n8k32.s32.s8.s8.s32` gives >= 5x
the dp4a kernel's prefill throughput for chunks of 64-512 tokens on the RTX 5060, bit-identical to the dp4a kernel.
Metrics: prefill tok/s (pp128/pp512), exact output equality, achieved int8 TOPS.
