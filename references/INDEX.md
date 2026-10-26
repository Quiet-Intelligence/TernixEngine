# References: 1.58-bit (ternary) LLMs and their inference, 2024 - October 2026

Compiled 2026-10-26 by web search plus reading of primary sources. Foundation papers (the original five PDFs) stay in
`foundation - why/` and `hardware execution - how/`. Everything else is catalogued here as links and notes; the PDFs of the
newer works are not redistributed in this repository (fetch them from the links; arXiv IDs given).

**Verification level** (column "V"): P = primary text read (method, results and stated limitations), A = abstract and search
snippets only, S = secondary source only (vendor page, news, blog). Numbers in the "Reported" column are the authors' own
claims and are not independently reproduced here unless an entry says so.

## 1. Foundations (in this repository)

| ID | Work | Date / venue | Core idea | V |
|---|---|---|---|---|
| F1 | BitNet: Scaling 1-bit Transformers for LLMs (Wang et al., MSR), arXiv 2310.11453 | 2023-10 | BitLinear, SubLN, binary weights, 8-bit activations | P |
| F2 | The Era of 1-bit LLMs: BitNet b1.58 (Ma et al.), arXiv 2402.17764 | 2024-02 | absmean ternary weights, per-token absmax int8 activations, LLaMA-like block | P |
| F3 | T-MAC: CPU Renaissance via Table Lookup (Wei et al.), arXiv 2407.00088, EuroSys'25 | 2024-07 | bit-serial LUT mpGEMM, g=4 nibble tables, weight interleaving, mirror consolidation, table quantisation | P |
| F4 | MARLIN (Frantar et al.), arXiv 2408.11743 | 2024-08 | cp.async pipeline, evict_first, XOR swizzle, striped partitioning for W4A16 on tensor cores | P |
| F5 | QuIP# (Tseng et al.), arXiv 2402.04396 | 2024-02 | Hadamard incoherence, E8P lattice codebook | P |

## 2. Models and training recipes (what exists to run)

| ID | Work | Date / venue | Core idea and reported result | Stated limits / future work | V |
|---|---|---|---|---|---|
| M1 | BitNet b1.58 2B4T technical report, arXiv 2504.12285; HF `microsoft/bitnet-b1.58-2B-4T{,-bf16,-gguf}` | 2025-04 | 2B params, 4T tokens, ReLU^2 GLU, SubLN, Llama-3 tokenizer, 4096 ctx; W1.58A8. Model card: 0.4 GB non-embedding memory, 29 ms CPU decode (bitnet.cpp) | embeddings excluded from the memory figure; needs bitnet.cpp for efficiency; larger models, longer context, multilingual, multimodal listed as future work | P |
| M2 | Spectra / TriLM (Nolano AI, Mila et al.), arXiv 2407.12327 | 2024-07 | open suite 99M-3.9B, 300B tokens; ternary beats FP16 of equal bit size above 1B | small scale only | A |
| M3 | Spectra 1.1: Scaling Laws and Efficient Inference for Ternary LMs, arXiv 2506.23025 (ACL 2025) | 2025-06 | ternary benefits more from tokens than parameters; 1.2T tokens; 2-bit and 1.6-bit packing; TriRun GPU kernel up to 5x over fp (8x in ICLR abstract) | release of TriRun not confirmed | A |
| M4 | Falcon-Edge / Falcon-E 1B, 3B (TII), HF blog 2025-05; `onebitllms` | 2025-05 | bf16 + native BitNet + pre-quantised variants from one run; Axolotl/Megatron-Core support | Falcon3-10B ternary experimental | S |
| M5 | Fine-tuning LLMs to 1.58bit (Hugging Face blog), HF1BitLLM Llama3-8B | 2024-09 | warmup quantisation (lambda scheduling) to convert Llama3-8B | accuracy gap remains | S |
| M6 | ParetoQ (Liu et al., Meta), arXiv 2502.02631, NeurIPS'25 | 2025-02 | 1-4 bit unified framework; ternary/2/3-bit on the same Pareto front; 600M ternary beats prior 3B ternary | scale limited; 2-bit kernels reported 4.14x over FP16 | A |
| M7 | BitNet Distillation / BitDistill (Wu et al., MSR), arXiv 2510.13998 | 2025-10 | SubLN + attention distillation + continued pretraining converts Qwen to 1.58-bit; 10x memory, 2.65x CPU | task-specific; gap grows with size without the recipe | A |
| M8 | Tequila: trapping-free ternary quantization (Huang et al., Tencent), arXiv 2509.23809, ICLR'26 | 2025-09 | reactivates deadzone-trapped weights as dynamic biases; >4% over SOTA ternary on ARC; code in AngelSlim | evaluation to 3B | A |
| M9 | Sherry: 1.25-bit ternary via 3:4 sparsity (Huang et al., Tencent), arXiv 2601.07892, ACL'26 | 2026-01 | 3:4 sparse ternary, 4 weights in 5 bits (4-bit LUT index + shared sign), Arenas annealed residual against weight trapping; i7-14700HX: 148 vs 132 (I2_S) vs 117 (TL2) tok/s at 0.7B | stops at 3B; weight-only (activations/KV bf16); no server-GPU kernels; Arenas adds QAT cost; needs retraining | P |
| M10 | CAT-Q: cost-efficient accurate ternary PTQ (Intel), arXiv 2606.26650, ICML'26 oral | 2026-06 | post-training ternarisation with 512 calibration samples, 1.7B-235B (8-60 h on 8xA100); beats BitNet 100B-token models | measured zero fraction 33-51% (see X1) | A |
| M11 | PTQTP: post-training trit-planes (Xiao et al.), arXiv 2509.16989 | 2025-09 | two ternary planes + scales per row, 2x1.58 bit, ~1 h quantisation | two planes double the weight traffic | A |
| M12 | PT^2-LLM: post-training ternarization (Yan et al.), arXiv 2510.03267, ICLR'26 | 2025-10 | asymmetric ternary quantiser, iterative ternary fitting, activation-aware grid alignment | PTQ accuracy below QAT | A |
| M13 | ExTernD: expanded-rank ternary decomposition (single author), arXiv 2607.13511 | 2026-07 | factor W as two ternary matrices with inner rank above full; ~5.2-5.5 effective bits matches Q4_K | not 1.58-bit in effect; unreviewed | A |
| M14 | TernaryLLM, arXiv 2406.07177 | 2024-06 | dual learnable ternarisation, outlier-friendly distillation; W1.58A16 | | A |
| M15 | Ternary Bonsai 1.7B / 4B / 8B / 27B (PrismML), 2026-04..09 and Bonsai 2 2B/4B/9B/27B (2026-09/10) | 2026 | Apache-2.0 ternary models on Qwen3 / Qwen3.5 bases, group-128 FP16 scales (1.71-1.8 bpw files), Bonsai 2 weights in a Hadamard-rotated basis (runtime must rotate activations); 262K ctx on 27B; vendor-reported 95-98% retention | vendor benchmarks only; custom GGUF types (PQ2_0, PTQ1_0) need the PrismML llama.cpp fork; upstream llama.cpp supports Q2_0 and Bonsai 2 (PR #29600, 2026-10), PTQ1_0 pending (#29672) | S |
| M16 | Maple-Preview 20B-A1B (DeepGrove), HF `deepgrove/maple-preview`, 2026-08, MIT | 2026-08 | natively trained ternary MoE (256 experts, 8 active, 131K ctx); 200+ tok/s on Mac mini M4 (vendor) | needs the stamsam llama.cpp fork; vendor numbers | S |
| M17 | BitNet embedding models (MSR), per microsoft/BitNet README news 2026-07 | 2026-07 | 1-bit/ternary text-embedding models with I2_S kernels | model cards not located in this survey | S |
| M18 | MatMul-free LM (Zhu et al.), arXiv 2406.02528 | 2024-06 | ternary accumulation, MLGRU token mixer, FPGA design | scale to 2.7B | A |
| M19 | BitNet a4.8 (Wang et al.), arXiv 2411.04965; BitNet v2 (native 4-bit activations + online Hadamard), arXiv 2504.18415 | 2024-11 / 2025-04 | 4-bit activations on Gaussian-like inputs, sparsified 8-bit elsewhere, 3-bit KV; H-BitLinear Hadamard before activation quantisation | needs training from scratch; v2 marked work in progress | A |

## 3. Inference systems and kernels (what we compete with)

| ID | Work | Date / venue | Design | Reported | Stated limits / gaps (verified where V = P) | V |
|---|---|---|---|---|---|---|
| S1 | bitnet.cpp (Wang et al., MSR), arXiv 2502.11880, ACL'25; github.com/microsoft/BitNet | 2025-02 | I2_S (2-bit, MAD, per-tensor int8 activations, K%128), TL1 (g=2, 2 bpw), TL2 (g=3, 1.67 bpw, sign + 4-bit index, mirror), pshufb/tbl LUTs | I2_S 6.25x over FP16; TL2 up to 2.32x over T-MAC; lossless on bitnet_b1_58-large | ternary-only; prefill left to future work; ELUT unproven; embeddings/lm_head handling not described; one model for the lossless claim; README: 2026-01 parallel kernels + embedding quantisation (1.15-2.1x extra) | P |
| S2 | T-MAC GitHub, microsoft/T-MAC | 2024-2025 | see F3; NPU extension (2025-05) | 2-4x over llama.cpp | baseline drift: numbers tied to llama.cpp b2794; T-MAC treats ternary as 2 one-bit planes | S |
| S3 | Pushing the Envelope of LLM Inference on AI-PC and Intel GPUs (Georganas et al., Intel), arXiv 2508.06753 | 2025-08 | VNNI4-interleaved int2 microkernel (1 shift, 2 AND, 4 vpshufb), BRGEMM TPP + PARLOOPER hybrid scheduling, XeTLA int2 x BF16 GPU kernel | up to 2.2x over bitnet.cpp end-to-end; BMG (Arc B580) 365 tok/s 2B vs A100 250 | no accuracy evaluation (dummy weights); OpenMP fork/join overhead on small GEMVs; 1-bit gives no gain on LNL | P |
| S4 | Litespark-Inference (Mindbeam), arXiv 2605.06485 | 2026-05 | ternary stored as int8 (556 MB), per-token int8 activations, zero-point correction with precomputed column sums, VNNI/SDOT, runtime dispatch | 39.9-41.2 tok/s on 2B4T on CPUs; 18-196x over PyTorch (not over bitnet.cpp) | weight memory 11% above 2-bit packing; threading not described; only BitNet architecture; no perplexity/task scores | P |
| S5 | BITCOS, "Breaking the 1.58-bit Barrier for Ternary LLMs", arXiv 2609.16338 | 2026-09 | presence bitmap + compacted sign vector, 2-z bits/weight (1.485 on sparsest), fp16 scale per 128; AVX-512 (pdep), AVX2 (5-instr expansion + vpdpbssd), Xe2 (256-entry SLM LUT) | GEMV 1.14-1.28x over 2-bit; decode 1.10-1.18x CPU, 1.02-1.27x Xe2 | GEMV only (no prefill/GEMM); no NVIDIA/CUDA; slower on an instruction-bound 8-core client part (LNL); needs vpdpbssd (AVX-VNNI-INT8) on AVX2 so it cannot run on 13th/14th-gen Intel; no code released; lm_head not discussed | P |
| S6 | Vec-LUT (Tsinghua AIR et al.), arXiv 2512.06443; vlut.cpp | 2025-12 | one LUT shared across parallel tokens, 1->N lookup, vector-LUT tensor layout | up to 4.2x for parallel (prefill) inference, 5 edge devices | edge CPUs; LUT cost < 1%; integrated in llama.cpp | A |
| S7 | T-SAR (Oh et al.), arXiv 2511.13676, DATE'26 | 2025-11 | LUTs generated inside the SIMD register file | 5.6-24.5x GEMM, 1.1-86.2x GEMV (simulation); LUT accesses >75% of memory requests in T-MAC/bitnet.cpp | simulation-based; prefill measured at N=128 only | A |
| S8 | Unified LUT inference with signed-digit K/V caches (Duan et al.), arXiv 2608.03229 | 2026-08 | K/V stored as multi-plane signed digits read by activation LUTs, shared datapath for projections and attention | see paper | hardware-oriented | A |
| S9 | LUT Tensor Core (MSR), arXiv 2408.06003 | 2024-08 | software-hardware co-design for LUT mpGEMM | >10x compute density/energy (simulation) | needs new hardware | A |
| S10 | TeLLMe / TeLLMe v2 (Qiao et al.), arXiv 2504.16266 / 2510.15926 | 2025 | FPGA table-lookup ternary matmul, prefill+decode, 5-7 W | 9 tok/s at 7 W (KV260) | FPGA only | A |
| S11 | TerEffic, arXiv 2502.16473 | 2025-02 | on-chip FPGA, 1.6 bpw weight encoding | 12.7-16.3k tok/s on 370M | model size limits | A |
| S12 | TENET, arXiv 2509.13765 | 2025-09 | sparsity-aware LUT-centric ternary accelerator (FPGA/ASIC) | 4.3x / 21.1x energy vs A100 | prototype | A |
| S13 | VitaLLM (2604.27396) and VitaLLM-BitNet (2605.00320) | 2026-04/05 | ternary/INT8 mixed accelerators; predictive sparse attention (54.9x less KV traffic) | | hardware | A |
| S14 | TOM: ternary read-only memory accelerator, arXiv 2602.20662 | 2026-02 | ROM-based ternary weight storage | | hardware | A |
| S15 | llama.cpp TQ1_0 (1.69 bpw, 5 trits/byte) and TQ2_0 (2.06 bpw), PR #8151 | 2024 | per-256 block scales; CUDA support not confirmed in this survey | TQ1_0 1.66x slower than Q2_0 on ARL (from S5) | blocks of 256 not lossless vs per-tensor training | A |
| S16 | BitNet GPU kernel, microsoft/BitNet gpu/ (2025-05) | 2025-05 | W2A8 GEMV, dp4a, CUDA cores, 4 trits per int8 | per 2B4T report | GEMV only, CUDA cores not tensor cores | S |
| S17 | Ladder (OSDI'23) / BitBLAS | 2023 | tensor-compiler data-type support incl. int2 | used for the original b1.58 latency numbers | | S |
| S18 | PrismML llama.cpp fork and upstream Q2_0 (group 64) | 2026 | PQ2_0 (type 142, group 128), PTQ1_0 (type 143, dense trits), Hadamard activation runtime for Bonsai 2 | CUDA, Metal, Vulkan, ROCm, SYCL | formats not yet upstream | S |
| S19 | Prism/Bonsai MLX 2-bit packs (mlx-vlm >= 0.7.8) | 2026 | Apple Silicon path | 310 tok/s 2B on M5 Max (vendor) | | S |

## 4. Adjacent techniques used or considered

| ID | Work | Relevance | V |
|---|---|---|---|
| A1 | TEAL training-free activation sparsity, arXiv 2408.14690 (ICLR'25) | 40-50% magnitude sparsity, 1.5-1.8x decode; gains limited because the LM head is not sparsified | A |
| A2 | SlimSpec: low-rank draft LM head, arXiv 2605.10453 | head cost O(Vd) -> O(rd+Vr) for drafters | A |
| A3 | "Your Quantized LLM Is Not Slow Because of the Quantization" (analysis) | head share of traffic 16% at FP16 -> 48% at 2-bit for Qwen3-4B (analytical) | S |
| A4 | Demystifying Small Language Models for Edge Deployment, ACL 2025 | vocabulary size disproportionately affects memory; decode memory-bound | A |
| A5 | Accuracy is Not All You Need (Dutta et al.), arXiv 2407.09141, NeurIPS'24 | KL divergence and "flips" as compression metrics | A |
| A6 | Does quantization affect long-context performance? arXiv 2505.20276 (EMNLP'25) | short-context evals under-report low-bit damage | A |
| A7 | BitDecoding (arXiv 2503.18773), RaBitQCache, D-Quant (2609.19880), KVQuant | low-bit KV caches for long context | A |
| A8 | Dovetail, TriSpec (2601.23180), spec-verify | speculative decoding; no ternary-draft study found | A |
| A9 | Marlin-style and BitDecoding tensor-core kernels | reference designs for decode-to-int8 then `mma` | A |

## 5. Evaluation and benchmarking practice (summary; details in ANALYSIS.md)

| ID | Source | Takeaway |
|---|---|---|
| E1 | lm-evaluation-harness (EleutherAI): ARC-e/c, HellaSwag, WinoGrande, PIQA, BoolQ, OpenBookQA, WikiText-2/C4 perplexity | used by F2, M1, M9 |
| E2 | 2B4T report suite: ARC-C/E, MMLU, HumanEval+, GSM8K, MATH-500, IFEval, MT-bench, memory (non-embedding) and CPU decode latency | head excluded from memory; latency at one config |
| E3 | T-MAC GitHub notes (baseline drift, thermal instability, equal-throughput framing); bitnet.cpp lossless table (1 model, 3 tasks) | engine comparisons are snapshot- and format-dependent |
| E4 | LLM-Inference-Bench (arXiv 2411.00136) | accelerator-level benchmarking, no ternary focus |
| E5 | A5 and A6 above | add KL/flips and long-context checks to accuracy |
