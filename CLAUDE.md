# TernixEngine: working agreement for Claude Code

1.58-bit (ternary) LLM inference engine: C++20 + CUDA core, Python tooling. Goal: a correct, fast, fairly benchmarked
engine, plus honest research on top of it. Product facts live in `README.md`; research notes in `docs/` and `references/`.

## Commands
- Build (any shell, loads MSVC): `cmd //c scripts\build.bat build` (CUDA arch is auto-detected; GPU is an RTX 5060 Laptop, sm_120)
- Test: `ctest --test-dir build --output-on-failure` ; Python: `PYTHONPATH=build python -m pytest tests -q`
- Full pipeline incl. benchmarks and README tables: `run_all.bat` (from the VS x64 Native Tools prompt)
- Python env: `uv` only (never `python -m venv`; see the CA-incident rules below). Run scripts with `uv run` or `python -I`.

## Git workflow (enforced by `.claude/hooks/guard.py`)
`main` = releases only. `develop` = integration. Work on `feature/*`, `fix/*`, `perf/*`, `docs/*`, `chore/*`, `research/*`
branches cut from `develop`, merge back with `git merge --no-ff`. Conventional commits (`feat:`, `fix:`, `perf:`, `docs:`,
`test:`, `chore:`, `research:`), atomic, no emojis, no attribution trailers. No force pushes, no direct commits to
`main`/`develop`. Push branches freely; merging `develop` into `main` is a release decision for the user.

## Style
- C++20, 32-byte-aligned tensor storage, no heavy dependencies in core inference (GoogleTest, Google Benchmark, pybind11 only).
- Dry engineering prose in code, comments and docs: no emojis, no filler words ("robust", "crucial", "delve"). Comments
  explain why, not what. Match the surrounding code's density and idiom.
- Integer-only GEMM path: do not cast weights or activations to float inside the matmul loops.

## Verification rules (what "done" means)
1. Every kernel has a differential test against an independent reference (bit-exact for integer paths), ragged shapes included.
2. Model-level changes are checked against the PyTorch/HF reference logits (`scripts/verify_logits.py`) before any speed claim.
3. A performance number is only reported with: hardware, thread count and pinning, build flags, warm-up, N repetitions,
   median and spread, and the same model/prompt/threads for every competitor. Raw JSON goes under `benchmarks/`.
4. Do not hand-edit generated README tables; regenerate them.
5. Say so when something was not run or failed. No claim without a command that produced it.

## Research protocol (R&D work)
Write the hypothesis, metric, baseline and stopping rule in `docs/research/EXPERIMENTS.md` BEFORE running. Record every run
(including negative results) with seed, commit hash and raw output path. One variable at a time. Ablate each claimed gain.
Cite sources in `references/` (see `references/INDEX.md`); mark anything unverified as such.

## Compute limits
Local box: i7-14650HX (8P+16E, AVX2 + AVX-VNNI, no AVX-512), 24 GB RAM, RTX 5060 Laptop 8 GB VRAM, ~50 GB free disk.
Anything needing > 8 GB VRAM, > ~20 GB RAM or multi-hour training (e.g. QAT of models above ~300M parameters, 7B+ inference)
goes to Google Colab: stop, tell the user, and provide a Colab/VS Code notebook with exact steps.

## Security (CA-key incident, Oct 2026)
pip >= 26.2.2, certifi >= 2026.10.19, uv >= 0.13.4. Create environments with `uv venv`. Download models/binaries only with
a hash check; Python code that downloads should build its SSL context from certifi (`ssl.create_default_context(cafile=certifi.where())`).
Untrusted downloads go in their own empty directory and are processed with `python -I`.
