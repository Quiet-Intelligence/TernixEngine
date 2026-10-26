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

---

# Outcomes (2026-10-26 / 27)

All CPU timings: 16 P-core threads pinned, BitNet 2B4T, 48-token decode after a 16-token prompt unless stated, interleaved repetitions
(`ternix abtest`), raw text in `docs/research/data/`. The memory read ceiling measured in the same session was 38.4 GB/s (16 threads);
earlier sessions measured 26 GB/s with background load, so ceilings are only comparable within a session.

## EXP-003 outcome: quantised output head with certified-exact rescoring

12,024 WikiText-2 positions (`exp003_headeval_12k.txt`, `ternix headeval`):

| head | bytes read per token | argmax identical to bf16 head | candidates rescored (mean / max) | KL(full || head logits) | notes |
|---|---:|---:|---:|---:|---|
| bf16 (reference) | 656.7 MB | - | - | - | |
| int8 per row, certified bound | 329.4 MB + 0.06 MB | 12024 / 12024 (guaranteed by the bound) | 10.8 / 829 | 8.5e-5 | exact for greedy |
| int4 group 32, 6.5-sigma bound | 185.2 MB + 0.10 MB | 12024 / 12024 (statistical) | 20.2 / 1939 | 6.0e-3 | worst error over 1.5e9 (row, position) pairs was 5.86 sigma |

H3a confirmed (and stronger: proved for int8). H3b: int4 matched on all 12,024 positions but its guarantee is statistical, not a proof;
the certified bound is useless for int4 (the worst-case bound keeps all 128k rows). The error ratio |exact - approx| / sigma follows
a Gaussian tail (max 5.86-6.16 over ~1.5e9 samples is what a Gaussian predicts), which is the evidence for the 6.5-sigma setting.
H3c (decode >= 1.3x): interleaved A/B, 21 rounds, `exp003_004_abtest_short.txt`:

| config | decode tok/s median (min..max) | paired ratio vs bf16 head (p10..p90) |
|---|---:|---:|
| bf16 head | 25.01 (24.25..25.89) | 1.000 |
| int8 head | 32.29 (30.26..33.56) | 1.295 (1.240..1.319) |
| int8 head + sparse down | 33.70 (32.11..34.46) | 1.341 (1.318..1.366) |
| int4 head + sparse down | 34.27 (32.08..35.39) | 1.381 (1.314..1.397) |

H3c holds for the int8 head with 1.295x (threshold 1.3x not reached by 0.005, reported as "just below"; the int4 head reaches 1.38x with
sparse down). The int8 result is within 3% of the bandwidth prediction (850 MB vs 1,177 MB per token = 1.38x upper bound). The same
head on CUDA: 157.7 -> 188.7 tok/s (1.20x) with the graph replay; greedy outputs identical (pytest `test_int8_head_greedy_is_identical`,
CPU and GPU, 3 prompts x 40 tokens). Caveat: single model, single text; the proof covers greedy decoding only, sampling sees exact
logits for candidates and approximate logits elsewhere.

## EXP-004 outcome: exact zero-skipping in the down projection

Microbenchmark (`exp004_bench_i2k.txt`, 2560x6912, weights rotating beyond the LLC, dense = 2-bit pool kernel at 35-38 GB/s):

| zeros in the int8 activation | nnz | dense us | sparse us | ratio |
|---:|---:|---:|---:|---:|
| 0% | 6912 | 144.8 | 240.2 | 1.66 |
| 50% | 3387 | 141.8 | 175.9 | 1.24 |
| 70% | 2076 | 146.9 | 120.9 | 0.82 |
| 80% (measured in the model) | 1358 | 128.4 | 102.2 | 0.80 |
| 90% | 673 | 153.6 | 70.6 | 0.46 |
| 95% | 337 | 131.8 | 46.9 | 0.36 |

Outputs were bit-identical in every row and in the engine (pytest `test_sparse_down_is_bit_identical`). H4 is falsified as stated
(time ratio 0.80 > 0.7 at 80% zeros; break-even near 72%). The kernel is instruction/latency bound (34 ns per nonzero column single-threaded,
about 60 us of fixed cost for chunking and reduction), not bandwidth bound, so skipping 80% of the bytes saves only 20% of the time.
End to end the effect is +4% decode (int8 head: 32.29 -> 33.70 tok/s, paired 1.295 -> 1.341); it is exact, so it stays as an option
(`--sparse-down`), default off. The measured nonzero fraction during decoding was 0.195 (EXP-002 predicted 0.196).

## EXP-005 outcome: threads, pinning and scheduling on the hybrid CPU

Grid in `exp005_sched.json` (median decode tok/s, int8 head, 7 repetitions):

| threads | pinned P-core, dynamic | pinned P-core, static | unpinned (OS) dynamic | pinned in logical-CPU order (incl. E-cores) dynamic / static |
|---:|---:|---:|---:|---:|
| 4 | 17.4 | 17.8 | 9.9 | 17.5 / 17.6 |
| 8 | 27.9 | 29.2 | 6.4 | 28.6 / 28.6 |
| 12 | 30.4 | 30.3 | 4.6 | 30.9 / 29.9 |
| 16 | 32.7 | 32.6 | 2.3 | 26.4 / 32.0 |
| 20 | 1.5 (wrapped pins) | 1.6 | 1.9 | 26.9 / 20.5 |
| 24 | 1.5 (wrapped pins) | 1.6 | 1.5 | 7.2 / 8.4 |

Findings: (1) the best configuration is 16 threads on the performance-core logical processors, 32.7 tok/s; 8 threads already reach 85% of it
(memory bound, ceiling 38 GB/s). (2) On equally fast cores dynamic and static scheduling are indistinguishable (32.7 vs 32.6). (3) Adding
E-cores never helps; with 20 threads dynamic claiming beats a static split by 31% (26.9 vs 20.5), which is the only place it matters.
(4) A spin-waiting pool without pinning collapses by up to 14x (16 threads unpinned: 2.3 tok/s) because workers migrate and share cores
with the main thread; the same happens when more workers than logical processors are pinned (the original code wrapped pins:
1.5 tok/s). The pool now spills extra workers onto the remaining logical processors instead of wrapping. H5 as registered
("static is slowest with the highest spread") is only partly supported: it holds above 16 threads, not on P cores. Practical consequence for any
comparison with other engines: thread count and affinity must be reported, and runs must be pinned or repeated.

## EXP-006 outcome: int8 KV cache and decode attention

Perplexity (WikiText-2, 8,200 tokens, `exp006_ppl.txt`) and decode speed at a 2,000-token context (`exp006_abtest_ctx2000_v3.txt`):

| window | KV fp16 perplexity | KV int8 perplexity | change |
|---:|---:|---:|---:|
| 512 | 21.3299 | 21.3383 | +0.04% |
| 2048 | 15.1864 | 15.2260 | +0.26% |

(CPU engine, bf16 head, 8,200 scored tokens; the fp16 cache itself differs from an fp32 cache by an amount of the same order as the noise
floor of EXP-001, so a 0.04-0.26% change is not distinguishable from rounding-level noise at this sample size.) Perplexity half of H6 holds
(< 1%); the speed half does not. Decode attention was rewritten (grouped query heads share each K/V row, split
into 256-position chunks, merged): 2,000-token-context decode went from 22.2 to 26.8 tok/s (+20%); the short-context decode is unchanged.
int8 K/V changed decode speed by +1.4% (paired median 1.015, p10..p90 0.979..1.067) at 2,000 tokens, below the 3% falsification threshold: the CPU
attention is compute/latency bound at this length, not bandwidth bound.

## EXP-007 outcome (negative): zero-bitmap + sign layout

Prototype `benchmarks/bench_bitmap.cpp` (BMI2 pdep, AVX2 mask expansion, vpdpbusd), zero density 0.42, bit-exact against the I2K128 kernel
(`exp007_bitmap.txt`):

| shape | bits/weight | pool, DRAM-rotating: I2K128 us (GB/s) | bitmap us (GB/s) | bitmap / I2K128 | 1-thread cache-resident GMAC/s: I2K128 / bitmap |
|---|---:|---:|---:|---:|---:|
| qo 2560x2560 | 1.631 | 48.0 (34.1) | 59.5 (22.5) | 1.24x slower | 104.4 / 16.8 |
| gate/up 6912x2560 | 1.631 | 114.5 (38.6) | 159.5 (22.6) | 1.39x slower | 132.0 / 16.9 |
| down 2560x6912 | 1.599 | 114.0 (38.8) | 149.1 (23.7) | 1.31x slower | 130.2 / 16.8 |

The hypothesis (>= 1.05x faster) is falsified. Without AVX-512 the decode of a bitmap plus compacted signs costs 6-8x the instructions of the
2-bit decode (16.8 vs 104-132 GMAC/s per thread), so the kernel is instruction bound at 22-24 GB/s while the 2-bit kernel reaches the 38 GB/s
memory ceiling. This agrees with the one instruction-bound platform in the BITCOS paper (S5, Lunar Lake) and indicates that its gains need either
AVX-512 `pdep`/mask registers or a wide memory system. Not tested: a GPU variant, AVX-512 hardware, or per-block adaptive fallback (moot
because every shape lost). Layout stays I2K128.

## EXP-008 outcome: tensor-core W1.58A8 prefill

`mma.sync.m16n8k32` kernel on a byte-permuted weight copy (`ternary_mma_kernel`), bit-identical to the CPU reference and to the dp4a kernel
(`CudaEngine.TensorCoreGemmMatchesCpuKernelExactly`, M in {9..64}, K up to 6912). GPU prefill, interleaved harness (`speed_ternix_phases.json`,
pp128, 7 rounds, median): baseline commit d264a0d (dp4a): 1226 tok/s; current: 3478 tok/s (bf16 head) / 3565 tok/s (int8 head), i.e. 2.8-2.9x. A
controlled toggle in the current build (`--no-tc`): pp64 1149 -> 3532 tok/s (3.1x), pp512 906 -> 2326 tok/s (2.6x; 2700 tok/s after
vectorising the prefill attention scores). H8 (>= 5x) is falsified end to end: after the GEMMs are accelerated the prefill is limited by the
attention kernel (44% of the time at pp512 before the vectorised score loop) and by normalisation/quantisation kernels, which were not tensor-core work. Not
measured: achieved int8 TOPS per kernel (nsys shows the GEMM share of the profile only).

## Phase summary: baseline engine vs improved engine (same session, interleaved rounds)

`docs/research/data/speed_ternix_phases.json` (pp128 / tg64, 7 rounds, median (min..max)); the harness crashed after the rounds while collecting
environment data and the table was reconstructed from its log (noted in the file):

| engine | prefill tok/s | decode tok/s | bytes/token (MB) |
|---|---:|---:|---:|
| ternix baseline CPU (d264a0d), bf16 head | 245.5 (202.4..280.9) | 24.94 (24.10..25.51) | 1177 |
| ternix CPU, bf16 head (current kernels) | 261.0 (157.5..337.2) | 24.40 (23.94..24.63) | 1177 |
| ternix CPU, certified int8 head | 447.0 (417.9..455.0) | 32.22 (29.77..33.48) | 850 |
| ternix CPU, int8 head + sparse down | 178.7 (161.8..179.7) | 31.99 (31.33..34.27) | ~803 |
| ternix CPU, int4 head + sparse down | 256.1 (217.2..328.6) | 33.22 (32.35..34.22) | ~658 |
| ternix baseline GPU (d264a0d), dp4a, no graph | 1226 (1140..1241) | 137.3 (129.5..148.2) | 1177 |
| ternix GPU, bf16 head | 3478 (3342..3503) | 211.5 (193.6..212.5) | 1177 |
| ternix GPU, certified int8 head | 3565 (3123..3582) | 257.4 (244.0..259.6) | 850 |

Reading: CPU decode was already at the memory ceiling in the baseline (24.9 tok/s x 1177 MB = 29 GB/s); the gains come from moving fewer bytes
(int8 head: 1.29x, int4 + sparse: 1.33x here, 1.38x in the paired in-process test). The CPU prefill column is noisy (a single 128-token pass is
0.4-0.7 s, min..max spans 1.7x) and the head mode cannot affect it, so the differences between its rows are noise, not effects. On the GPU the same
byte reduction compounds with CUDA-graph replay (launch overhead), tensor cores and split-KV attention: decode 1.88x, prefill 2.9x over the
baseline commit.
