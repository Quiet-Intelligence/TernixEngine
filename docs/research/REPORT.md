# Where the bytes go: a measured study of 1.58-bit LLM decoding on consumer hardware

Technical report, TernixEngine, 2026-10-27. Protocols, raw data and negative results: `EXPERIMENTS.md`, `data/`. Literature: `../../references/`.

## 1. Question and setting
Autoregressive decoding of a ternary model is memory-bandwidth bound. Published ternary kernels (bitnet.cpp, T-MAC, Litespark, BITCOS, Sherry) compete on
how densely the *ternary matrices* are packed and how cheaply they are decoded. We ask a prior question: for a real model on a real machine, which bytes
does a decoded token actually read, and how many of them can be removed without changing the output?

Testbed: i7-14650HX (8 P + 16 E cores, AVX2 + AVX-VNNI, no AVX-512), 24 GB single-channel DDR5-5600 (read ceiling 38.4 GB/s measured in the final session),
RTX 5060 Laptop 8 GB. Model: BitNet b1.58 2B4T (2.4 B parameters, W1.58A8, tied 128,256 x 2,560 embedding).

## 2. Findings
1. **The head is the largest tensor.** 626 MiB of bf16 embedding against 497 MiB of ternary weights: 55.8% of the bytes of a decoded token. Memory
   figures that exclude the embedding (the model card's "0.4 GB", the BITCOS and bitnet.cpp papers) understate decode traffic by 2.3x for this model.
2. **A quantised head can be made exact for greedy decoding.** Store int8 rows with their residual norm r_v. For a hidden vector x the bf16 logit lies in
   s_v +- ||x|| r_v (Cauchy-Schwarz); only rows whose upper bound reaches the best lower bound can be the argmax, and those few (mean 10.8, max 829 of
   128,256 over 12,024 WikiText-2 positions) are rescored from the bf16 rows. The argmax therefore equals the bf16 head's by construction, and the traffic
   falls from 657 MB to 329 MB. CPU decode 24.9 -> 32.2 tok/s (1.29x paired); CUDA 212 -> 257 tok/s. For int4 the worst-case bound is useless (it keeps all
   rows) but the approximation error follows a Gaussian tail with sigma = ||x|| r_v / sqrt(H) (max 5.86 sigma over 1.5e9 samples): a 6.5-sigma statistical bound
   keeps 20 candidates on average, matched all 12,024 positions and moves 185 MB.
3. **Exact zeros are free.** With squared-ReLU FFNs, 80.4% of the down-projection input is exactly zero after per-token absmax int8 quantisation. Skipping
   those columns (K-major copy) is bit-identical, but our kernel only reaches 0.80x of the dense time at 80% zeros (break-even 72%): the sparse access is
   instruction/latency bound, so the bandwidth saving does not convert. End-to-end +4%. Reported as a falsified efficiency hypothesis with a correct and
   cheap implementation.
4. **Denser packing is not automatically faster.** A zero-bitmap + sign layout (2 - z bits per weight, 1.60-1.63 bpw here) decodes 6-8x slower per weight with
   BMI2 `pdep` + AVX2 expansion and loses 1.24-1.39x against 2-bit on this CPU; the 2-bit kernel already sits at the memory ceiling. The gains reported for
   such layouts need wider vectors, mask registers or higher bandwidth per core.
5. **On a hybrid CPU the pool design matters more than the kernel.** Pinned 16-P-core dynamic scheduling gives 32.7 tok/s. The same spin-waiting pool
   unpinned runs at 2.3 tok/s (14x slower), and pinning more workers than logical processors gives 1.5 tok/s; E-cores never help (20 threads: 26.9, with
   dynamic claiming 31% better than a static split). Cross-engine comparisons that do not state affinity are not comparable.
6. **Numerical noise sets the correctness bar.** A relative 2^-24 perturbation of the input embeddings gives a logit KL of 1e-4..8e-3 (median 1e-3) with no
   top-1 flip, because per-token int8 quantisation turns last-bit differences into rounding flips. Engine-vs-reference differences (median 3-5e-3) live inside
   that band. Logit-exactness is the wrong gate for W1.58A8; exact integer-kernel tests plus top-1/KL within the noise band are the right ones.
7. **GPU decode is overhead bound until it is graph-captured.** 400+ kernel launches per token cost more than the bandwidth time. CUDA-graph replay (position and
   token read from device memory), split-KV attention and the device-side certified head take decode from 137 to 257 tok/s (ideal for 850 MB at 384 GB/s:
   ~450); int8 tensor-core prefill is bit-exact and 2.6-3.1x faster (the 5x hypothesis failed: attention and norms dominate afterwards).
8. **int8 KV cache** costs +0.04% / +0.26% perplexity (window 512 / 2048) and gives no speed at 2k context on the CPU (attention is compute bound there);
   grouped split-chunk attention (query heads of a group share each K/V row) is +20% at 2k context.

## 3. What this changes for the field
* Report decode memory *including* the embedding/head and the KV cache, and report bytes per token next to tokens per second; otherwise 2-bit, 1.6-bit and
  1.25-bit layouts cannot be compared and head-bound models look better than they are.
* Quantise or approximate the head with a proof or a measured tail bound; for the ternary models studied it is the cheapest 1.3x available and it needs no
  retraining.
* Treat exact activation sparsity of ReLU^2 models as a free resource, but check that the kernel is not instruction bound before expecting bandwidth gains.
* Specify affinity and pool behaviour in every CPU benchmark on hybrid processors.

## 4. Threats to validity
One model and one text for most measurements; one machine (single-channel memory raises the weight of bytes-moved effects relative to dual-channel desktops
and servers); CPU prefill timings are noisy (min..max up to 1.7x) and should not be read as effects; the int4 guarantee is statistical; competitor engines were not
measured (permission), so no ranking against llama.cpp, bitnet.cpp or T-MAC is claimed; the harness and exact commands are ready (`bench/COMPETITORS.md`).
Numbers from different sessions are not comparable (the read ceiling moved between 26 and 38 GB/s); all comparisons used here are interleaved within a session.

## 5. Future work
A GPU backend for the group-scaled architecture, further 2026 ternary families, AVX-512 VNNI and NEON kernels, a certified top-k head for sampling, speculative decoding with a
ternary draft, KV compression for long-context models, the competitor comparison, and energy measurement with a wall-power meter (package/board power only is measured here).
