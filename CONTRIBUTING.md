# Contributing to TernixEngine

Contributions are licensed under the Elastic License 2.0 (see `LICENSE`). `CLAUDE.md` is the working agreement for humans and agents
alike; this file is the short version.

## Workflow
1. Branch from `develop`: `feature/*`, `fix/*`, `perf/*`, `docs/*`, `chore/*` or `research/*`. `main` is for releases only.
2. Conventional commits (`feat:`, `fix:`, `perf:`, `docs:`, `test:`, `chore:`, `research:`), atomic, no emojis, no attribution trailers.
3. Every new kernel or operator ships with a differential test (exact for integer paths, ragged shapes included) in `tests/`.
4. A performance claim needs: hardware, thread count and pinning, build flags, N interleaved repetitions with median and spread, and
   the head and KV cache included (see `bench/ternixbench.py`). A change in logits needs the noise floor of EXP-001 for context.
5. Research changes register the experiment in `docs/research/EXPERIMENTS.md` before running it, and record negative results.
6. Merge into `develop` with `git merge --no-ff`; CI (`.github/workflows/ci.yml`) must pass.

## Build and test
```bat
scripts\build.bat build-dev          :: MSVC + Ninja, CUDA auto-detected (Windows)
ctest --test-dir build-dev --output-on-failure
set PYTHONPATH=build-dev && python -m pytest tests -q      :: needs `uv pip install numpy pytest tokenizers`; model tests skip without models/
```
Python environments are created with `uv venv` (see the security notes in `CLAUDE.md`).

## Reporting bugs
Include OS, CPU and GPU model, the failing command and its output, and `build-dev/CMakeCache.txt`.
