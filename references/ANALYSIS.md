# Analysis: what the literature leaves open, and what TernixEngine does about it

Sources and verification levels are in `INDEX.md` (IDs such as S5 refer to it). Measurements marked **[measured]** were made
in this repository on the testbed (i7-14650HX, 24 GB single-channel DDR5-5600, RTX 5060 Laptop 8 GB); see
`docs/research/EXPERIMENTS.md` for protocols and raw data.

## 1. State of the field in one paragraph

Ternary (W1.58A8) LLMs are now a family, not one model: BitNet b1.58 2B4T (M1), Falcon-E (M4), TriLM/Spectra (M2, M3), ParetoQ
(M6), PTQ/QAT conversions of Qwen and Llama (M7-M13), and since 2026 commercially packaged ternary models up to 27B and a 20B MoE
(M15, M16). Kernel work has converged on three families: 2-bit multiply-add on integer dot-product instructions (bitnet.cpp
I2_S, Litespark, Intel PyTorch-TPP: S1, S3, S4), lookup-table kernels (T-MAC, TL1/TL2, Vec-LUT, T-SAR: S1, S2, S6, S7), and
denser-than-2-bit layouts (TQ1_0 at 1.69 bpw, Sherry at 1.25 bpw through retraining, BITCOS at 2-z bpw through a zero bitmap:
S15, M9, S5). Decode is memory-bandwidth bound everywhere; the open question is how few bytes per token an engine can move
without paying more in instructions than it saves in bandwidth.

## 2. Gaps, with evidence

**G1. The output head is the largest single tensor and is ignored.** BitNet 2B4T has 128256 x 2560 tied embeddings in bf16:
656 MB against 521 MB of ternary weights **[measured from the .ternix file: 626 MiB vs 497 MiB]**, i.e. 56% of the bytes
read per decoded token. The model card reports "non-embedding memory 0.4 GB" (M1/E2), bitnet.cpp does not describe head
handling (S1), BITCOS does not mention it (S5), and an analytical note puts the head at 48% of traffic for a 2-bit Qwen3-4B
(A3). TEAL notes its gains are limited by the unsparsified head (A1). No source benchmarks a quantised or approximated head for
ternary models. => Contribution C1.

**G2. Exact activation sparsity in ReLU^2 FFNs is not used.** BitNet 2B4T uses squared ReLU, so the down-projection input
contains exact zeros. After per-token absmax int8 quantisation **80.4% of the down-projection input is exactly 0 and 88.0% is in
{-1,0,1}** (2048 WikiText-2 tokens, layer range 69-91%, EXP-002); the other projections see 1.5-4.8% zeros. A4.8 and TEAL
induce sparsity by training or thresholding (M19, A1); here it is free and lossless in the model's own arithmetic, but only
exploitable at batch 1 with a column-major layout (skipping is per input element). The down projection is 26% of ternary
bytes. => Contribution C2.

**G3. Zero-aware layouts are decode-GEMV-only and ISA-restricted.** BITCOS (S5, the most relevant 2026 work) reaches 2-z
bits/weight but evaluates only GEMV, has no NVIDIA path, needs AVX-512 `pdep` or the AVX-VNNI-INT8 `vpdpbssd` (absent on
Raptor Lake and earlier; this CPU reports `avx_vnni_int8 = 0`), found no gain on an instruction-bound 8-core part, and
released no code. Sherry (M9) needs retraining. TQ1_0 is slower than 2-bit (S1, S5). The measured mean zero fraction of
2B4T is 42.3% **[measured, 210 matrices]**, in line with S5's 42.19%. => Contribution C3 (a layout that works with plain
AVX-VNNI `u8 x s8` and with dp4a, evaluated on this hardware).

**G4. Prefill and batched decode are under-served.** bitnet.cpp leaves prefill to future work (S1), BITCOS and the NVIDIA
kernel of the 2B4T release are GEMV (S5, S16), Vec-LUT addresses prefill with LUTs on edge CPUs (S6). On NVIDIA hardware no
open ternary engine uses int8 tensor cores for W1.58A8 prefill (S16 uses dp4a on CUDA cores; our baseline does too: 700 tok/s
prefill **[measured]**). => Contribution C4.

**G5. Hybrid cores and single-channel memory are not studied.** Published thread-scaling results (S1: peak at 4-5 threads;
S3: PARLOOPER hybrid scheduling) use 2-channel desktops/servers. On this laptop the sequential-read ceiling is 26 GB/s with 16
threads and falls to 17 GB/s when the 24th logical core is added **[measured, bench_membw]**; static partitions across
P and E cores would stall on the slowest core. => Contribution C5 (dynamic chunk scheduling on P cores, measured against a
static partition).

**G6. KV cache is the next bandwidth term once context grows.** fp16 KV at 4096 tokens for 2B4T is 315 MB per decoded token,
60% of the ternary weights; Bonsai-class models reach 262K context (M15). Ternary + KV-quantisation appears only in hardware
papers (S8, S13). => Contribution C6 (int8 KV, measured in perplexity and speed).

**G7. Benchmarks are snapshot-dependent and exclude the head.** Baselines drift (T-MAC's numbers tie to llama.cpp b2794;
updating llama.cpp alone moved both sides by 10-15%), formats differ in bits per weight, llama.cpp silently falls back for
shapes not divisible by 256, thermal throttling destabilises multi-thread runs, energy methods are unstated (E3). Quality
claims use one model and three tasks for "lossless" (S1) or no accuracy at all (S3). => Contribution C7 (TernixBench).

**G8. Numerical fragility of W1.58A8 is unquantified.** Per-token int8 quantisation turns last-bit differences into rounding
flips. **[measured, EXP-001]** A relative 2^-24 perturbation of the embeddings gives a logit KL of 1e-4..8e-3 (median 9.6e-4)
with no top-1 flip, the same band as engine-vs-reference differences. Consequences: logit-exactness against a float
reference is not a valid correctness gate; quality comparisons between engines need an explicit noise floor; "lossless"
claims (S1) are statements about perplexity equality on one model, not about logits. => Methodological contribution M1
(noise floor and KL/flips reporting, after A5).

**G9. Generality.** Every open ternary engine is tied to the 2B4T block (S4 says so explicitly) or to a vendor fork with
private GGUF types (S18). 2026 ternary models use group-128 FP16 scales, SiLU-GLU, QK-norm and, for Bonsai 2, a Hadamard
rotation applied to activations at run time (M15). The I2K128 layout (128-column blocks) already matches group size 128; a
per-block float scale costs one convert and one FMA per 128 weights per row. => stretch goal C8 (group scales and a
Qwen3-style block) if time permits.

## 3. Contributions planned (each gets a pre-registered experiment)

| ID | Contribution | Type | Success criterion (registered in EXPERIMENTS.md before running) |
|---|---|---|---|
| C1 | Quantised output head with exact top-C rescoring (int8, int4) | engine + analysis | greedy next-token identical to bf16 head on >= 99.9% of positions at C = 64; decode >= 1.3x on CPU and GPU; perplexity unchanged within the noise floor |
| C2 | Exact activation-sparse down projection (column-major layout, batch 1) | engine | bit-identical output; down-projection time <= 0.4x dense at 80% zeros; report break-even sparsity |
| C3 | Zero-bitmap + sign layout for AVX-VNNI and CUDA, block-adaptive fallback to 2-bit | format + kernels | decode speed >= 2-bit layout on the P-core pool and on the GPU, payload -17% at z = 0.42; if slower it is reported as a negative result |
| C4 | Tensor-core W1.58A8 prefill (decode trits to s8 in registers, `mma.sync.m16n8k32`), and a blocked AVX-VNNI GEMM | kernels | >= 5x prefill tokens/s over the dp4a baseline on GPU; >= 2x on CPU |
| C5 | Dynamic chunk scheduling on P cores | systems | beats static partition and "all 24 threads" on decode tok/s with lower variance |
| C6 | int8 KV cache with per-token-per-head scales | engine | perplexity within noise at 512-4096 ctx; decode speed gain grows with context |
| C7 | TernixBench: fair cross-engine and quality suite | tooling | pinned versions, bytes-per-token roofline column, head included, noise floor, KL/flips, thread sweeps, JSON out |
| C8 | Group-scale ternary + Qwen3-style block (Ternary-Bonsai-class models) | engine | loads and matches a PyTorch reference to the EXP-001 noise floor |

CUDA graphs (launch overhead of ~400 kernel launches per token) are an engineering item, not a contribution.

## 4. Evaluation design (TernixBench), distilled from the literature

* Speed: end-to-end tok/s for prefill (pp128, pp512) and decode (tg128) including the head and a realistic KV length; median of
  N >= 7 after warm-up, with min/max; fixed prompt token ids; thread sweep and P-core pinning stated; GPU clocks/power state
  recorded; competitor builds pinned by commit; bits-per-weight and file size reported next to every number; **bytes moved per
  token and fraction of the measured bandwidth ceiling** as the fairness column (a 2-bit and a 1.6-bit kernel are comparable
  only through this).
* Quality: WikiText-2 perplexity at fixed windows, KL(ref || engine) and top-1 flips against the float32 reference on a fixed
  prompt set with the EXP-001 noise floor next to them (A5), lm-evaluation-harness zero-shot tasks (ARC-e/c, HellaSwag,
  WinoGrande, PIQA) as in F2/M1, and a long-context probe (A6) when a long-context model is available.
* Energy: not measured unless a wall-power meter is available (RAPL is not reachable on this Windows host); NVML board power
  for the GPU is recorded and labelled as such.
* Pitfalls the suite guards against: shape fallbacks (assert the kernel actually used), baseline drift (pin and log), thermal
  throttling (log temperatures, randomise engine order), and formats that are not equal in bits per weight.
