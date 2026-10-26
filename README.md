# TernixEngine
**A 1.58-bit (ternary) LLM inference engine: AVX-VNNI/AVX2 CPU and CUDA backends, a bit-exact verification harness, and a research log of what actually makes ternary decoding fast.**
*Distributed under the Elastic License 2.0 (see [License](#license)).*

![C++20](https://img.shields.io/badge/C%2B%2B-20-blue) ![CUDA](https://img.shields.io/badge/CUDA-sm__80%2B-green) ![AVX-VNNI](https://img.shields.io/badge/SIMD-AVX2%20%7C%20AVX--VNNI-orange) ![Python](https://img.shields.io/badge/Python-PyBind11-yellow)

## Abstract
TernixEngine runs BitNet b1.58-style models (weights in {-1, 0, +1}, per-token int8 activations) end to end: a converter from the
Hugging Face checkpoint, a memory-mapped model format, a Llama-3 BPE tokenizer, CPU and GPU forward passes, sampling, a CLI, a Python
binding and an OpenAI-compatible server. Its kernels never materialise dequantised weights: a ternary matrix is stored at 2 bits per
weight in a layout whose activation partners are contiguous loads (`docs/FORMAT.md`), and the dot product is an unsigned-byte x
signed-byte `vpdpbusd` / `dp4a` / `mma.sync` with a one-scalar correction.

Beyond a working engine, the project is a measured study of where the bytes go. For BitNet b1.58 2B4T the tied bf16 output head is
**56% of the bytes read per decoded token**, a fact the model card's "0.4 GB non-embedding memory" hides. The engine therefore adds a
**certified-exact quantised head** (the argmax equals the bf16 head's, with a Cauchy-Schwarz proof for int8), an exact activation-sparse
down projection (80% of that input is exactly zero), CUDA-graph decoding, int8 tensor-core prefill and grouped split-chunk attention, and
every claim is logged with a pre-registered hypothesis, raw data and the negative results (`docs/research/EXPERIMENTS.md`).

> [!NOTE]
> **Status.** Working: conversion, CPU and CUDA inference of BitNet b1.58 2B4T, tokenizer (exact parity with Hugging Face `tokenizers`),
> verification, perplexity and lm-eval harnesses, server. Not yet done: head-to-head runs against llama.cpp, bitnet.cpp and T-MAC
> (building and running third-party binaries needs an explicit permission decision, see [Competitors](#competitors)); other architectures
> (group-scaled and weight-only ternary models such as Ternary Bonsai); AVX-512 kernels.

## Results (testbed: i7-14650HX 8P+16E with AVX-VNNI, 24 GB single-channel DDR5-5600, RTX 5060 Laptop 8 GB, Windows 11)
BitNet b1.58 2B4T, pp128 / tg64, 7 interleaved rounds, median (min..max); bytes/token is what the engine reads per decoded token.

| engine | prefill tok/s | decode tok/s | bytes/token |
|---|---:|---:|---:|
| baseline CPU (first end-to-end commit), bf16 head | 245 (202..281) | 24.9 (24.1..25.5) | 1177 MB |
| CPU, certified int8 head | 447 (418..455)* | **32.2** (29.8..33.5) | 850 MB |
| CPU, int4 head (6.5-sigma) + sparse down | 256 (217..329)* | **33.2** (32.4..34.2) | ~658 MB |
| baseline GPU (dp4a GEMV, no graph) | 1226 (1140..1241) | 137 (129..148) | 1177 MB |
| GPU, bf16 head | 3478 (3342..3503) | 212 (194..212) | 1177 MB |
| GPU, certified int8 head | **3565** (3123..3582) | **257** (244..260) | 850 MB |

\* single 128-token CPU passes are noisy (min..max spans 1.7x) and the head cannot affect prefill: differences between those rows are noise.
CPU decode is memory bound: the measured read ceiling was 38.4 GB/s (26 GB/s in an earlier, loaded session, so ceilings are only comparable within a
session), and the baseline already ran at ~29 GB/s; every gain is a reduction in bytes. At 2,000 tokens of context decode is 26.8 tok/s.
Quality is unchanged: greedy output is identical with the int8 head on CPU and GPU (tested), perplexity with an int8 KV cache moves by
+0.04% (window 512) and +0.26% (window 2048), and engine logits agree with an independent float32 PyTorch reference to within the numerical
noise floor of W1.58A8 (EXP-001: a 2^-24 relative perturbation of the embeddings alone gives logit KL of 1e-4..8e-3).

Zero-shot task accuracy of the converted model through the engine (lm-evaluation-harness adapter `bench/lm_eval_ternix.py`, first 500 items per
task, CUDA engine, 95% half-width about +-2 points; raw `docs/research/data/lm_eval_ternix_gpu.json`):

| ARC-Easy acc / acc_norm | ARC-Challenge acc / acc_norm | HellaSwag acc / acc_norm | PIQA acc / acc_norm | WinoGrande acc |
|---:|---:|---:|---:|---:|
| 75.0 / 75.6 | 46.0 / 46.4 | 50.4 / 60.6 | 77.0 / 76.8 | 74.4 |

These are consistent with the published numbers for this checkpoint (e.g. ARC-Easy 74.8 on the model card); the lm-eval tasks score through
teacher-forced logits, so they do not exercise the quantised head (the head only changes how the next token is *selected*, which the greedy
identity tests cover).

### What the experiments found (full tables in `docs/research/EXPERIMENTS.md`)
| ID | Question | Result |
|---|---|---|
| EXP-001 | How noisy are W1.58A8 logits? | float32-rounding-level perturbations give KL 1e-4..8e-3: logit bit-exactness is not a valid gate |
| EXP-002 | Exact zeros in activations | 80.4% of the FFN down-projection input is exactly 0 (ReLU^2 plus int8 rounding) |
| EXP-003 | Quantised head | int8: argmax identical on 12,024/12,024 positions (proved), 329 MB vs 657 MB, decode 1.29x; int4 (statistical): 12,024/12,024, 185 MB |
| EXP-004 | Skip zero columns in the down projection | bit-identical; kernel only 0.80x of dense time at 80% zeros (hypothesis falsified), +4% decode end to end; opt-in |
| EXP-005 | Threads, pinning, scheduling on a hybrid CPU | best = 16 P-core threads pinned (32.7 tok/s); an unpinned spin pool collapses up to 14x; E-cores never help |
| EXP-006 | int8 KV cache | perplexity within 0.3%, but no speed gain at 2k context (attention is not bandwidth bound there) |
| EXP-007 | Zero-bitmap + sign layout (BITCOS-style) | **negative**: 1.24-1.39x slower than 2-bit on this AVX2/VNNI CPU (instruction bound, 17 vs 104-132 GMAC/s per thread) |
| EXP-008 | Tensor-core prefill | bit-exact, 2.6-3.1x prefill (hypothesis of 5x falsified: attention and norms dominate afterwards) |

## Build, convert, run
Requirements: C++20 compiler (MSVC from VS 2022/2026 or GCC 11+), CMake >= 3.20, Ninja, CUDA toolkit (optional), Python >= 3.10 with `uv`.
```bat
:: Windows (any shell): loads the MSVC environment, configures and builds with Ninja
scripts\build.bat build-dev
uv venv && uv pip install numpy safetensors huggingface_hub tokenizers pytest
uv run scripts/fetch_model.py microsoft/bitnet-b1.58-2B-4T-bf16          :: sha256-verified download
uv run scripts/quantize.py --input models/bitnet-b1.58-2B-4T-bf16 --output models/bitnet-2b4t.ternix

build-dev\ternix_cli.exe generate models\bitnet-2b4t.ternix --prompt "How many legs does a spider have?" --chat --n 64            :: CPU
build-dev\ternix_cli.exe generate models\bitnet-2b4t.ternix --prompt "..." --chat --gpu --head int8                              :: CUDA
build-dev\ternix_cli.exe chat models\bitnet-2b4t.ternix --head int8                  :: multi-turn with a shared KV cache
build-dev\ternix_cli.exe bench models\bitnet-2b4t.ternix --pp 128 --tg 64 --head int8
build-dev\ternix_cli.exe abtest models\bitnet-2b4t.ternix --configs "bf16,int8,int8+sparse,int4+sparse"   :: interleaved A/B decode
build-dev\ternix_cli.exe ppl models\bitnet-2b4t.ternix data\wikitext2_test.txt --ppl-ctx 512 --max-tokens 20000 --gpu
```
Linux/WSL: `cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build`.
CLI options: `--threads T` (default: one per performance core, pinned), `--isa scalar|avx2|avx_vnni`, `--head bf16|int8|int4`, `--sparse-down`,
`--kv-int8`, `--gpu`, `--no-graph`, `--no-tc`, sampling flags (`--temp --top-k --top-p --min-p --repeat-penalty --seed`).

Python (`PYTHONPATH=build-dev`):
```python
import ternix_engine as tx
e = tx.Engine("models/bitnet-2b4t.ternix", gpu=True, head="int8")
print(e.generate("The capital of France is", n=16))
logits = e.logits(e.encode("Hello", True, True))            # last-position logits
nll, argmax = e.nll(tokens, targets)                         # teacher-forced scoring (used by the lm-eval adapter)
```
Server: `python scripts/serve.py --model models/bitnet-2b4t.ternix --gpu` (OpenAI-style `/v1/chat/completions`, streaming, 127.0.0.1 only).

## Design (details in `docs/ARCHITECTURE.md`, `docs/FORMAT.md`)
* **Format.** `.ternix`: mmap-able container; ternary matrices in I2K128 (128-column blocks of 32 bytes, 2-bit codes w+1); tokenizer embedded.
* **CPU.** `vpdpbusd` (AVX-VNNI) / `vpmaddubsw` (AVX2) / scalar chosen at run time from cpuid; dynamic-chunk thread pool pinned to P cores;
  `OutputHead` with certified int8 and statistical int4 modes; exact activation-sparse down projection; int8 KV option; grouped split-chunk
  decode attention.
* **GPU.** dp4a GEMV for up to 8 tokens, int8 tensor-core GEMM (`mma.sync.m16n8k32`) above that, CUDA-graph decode step (position and token read
  from device memory), split-KV decode attention, device-side certified int8 head, device-side NLL for perplexity.
* **Verification.** integer kernels against an independent reference (exact), tokenizer against Hugging Face (exact), logits against an independent
  float32 PyTorch implementation and Hugging Face (KL within the measured noise floor), CPU-vs-GPU agreement, exactness tests for every
  optimisation that claims to be lossless.

## Repository layout
```text
include/ternix, src/      engine: format, tokenizer, ops, i2k kernels, head, model (CPU), cuda_engine.cu, thread pool, CLI
bindings/python/          PyBind11 module (Engine, kernels)
scripts/                  quantize.py, fetch_model.py, verify_logits.py, reference_bitnet.py (PyTorch), serve.py, exp00*.py
bench/                    ternixbench.py (fair cross-engine harness), lm_eval_ternix.py, hf_bench.py
benchmarks/               kernel micro-benchmarks (decode shapes, bandwidth ceiling, sparse, bitmap prototype)
tests/                    GoogleTest (CPU, CUDA), pytest (binding, tokenizer parity, end to end)
references/               foundation papers, INDEX.md (45+ works 2024-2026), ANALYSIS.md (gaps and plan)
docs/                     ARCHITECTURE.md, FORMAT.md, COLAB.md, research/EXPERIMENTS.md and data
CLAUDE.md, .claude/       working agreement and guard hook for autonomous work
```

## Testing and CI
`ctest --test-dir build-dev` (kernels, packing, norms, CUDA kernels), `pytest tests` (binding, tokenizer parity, end-to-end; model-dependent tests skip
without `models/bitnet-2b4t.ternix`). CI (`.github/workflows/ci.yml`) builds and tests on Linux and Windows, and compiles the CUDA code for sm_80..90;
hosted runners have no GPU, so GPU execution is verified locally.

## Competitors
The comparison against llama.cpp (TQ1_0/TQ2_0/Q2_0), bitnet.cpp (I2_S, TL1, TL2), T-MAC and Litespark has **not** been run: the assistant's
auto-mode permission classifier denied building third-party code and executing downloaded binaries, and that decision belongs to the repository
owner. The harness is ready (`bench/ternixbench.py`, `bench/COMPETITORS.md` lists the exact commands, official release hashes verified).
Until those numbers exist, no claim of being faster than any other engine is made. The official Hugging Face PyTorch path is measured as
a reference in the results above.

## Limitations and roadmap
1. One architecture (BitNet b1.58 block, per-tensor scale). 2026 ternary models (Ternary Bonsai, CAT-Q/Qwen3-class conversions) use group-128 scales,
   SiLU-GLU, QK-norm and in several cases weight-only (A16) inference, which this int8-activation path does not reproduce faithfully.
2. CPU kernels target AVX2/AVX-VNNI; AVX-512 VNNI and ARM NEON/SDOT are not implemented.
3. The int4 head's guarantee is statistical (6.5 sigma, 0 misses in 12,024 positions); only int8 is proved. Sampling sees exact logits for the
   candidates and approximate logits elsewhere.
4. Single-sequence decoding; no continuous batching, no paged KV, no speculative decoding.
5. Energy is not measured (no wall-power meter; RAPL is not reachable on this host).

## License
Distributed under the **Elastic License 2.0** (`LICENSE`): use, modify and share with attribution; not as a hosted/managed service; no
circumvention of license-key functionality. Provided as-is.

## Citation
```bibtex
@software{ternix_engine_2026,
  author = {Tripathi, Pundarikaksh N.},
  title  = {TernixEngine: a 1.58-bit LLM inference engine with certified-exact quantised heads},
  year   = {2026},
  url    = {https://github.com/Quiet-Intelligence/TernixEngine}
}
```
