# Running the competitor engines (needs your approval: third-party code is built and executed)

The assistant session that produced this repository was not allowed to build third-party sources or run downloaded binaries (permission
classifier decision). Everything below is therefore prepared but unrun. Run these yourself in a terminal (in Claude Code prefix with `!`),
or add a Bash/PowerShell permission rule for the competitor directory, then re-run `bench/ternixbench.py`.

Work in `C:\PROFESSIONAL\Development\AI-HPC\ternix-competitors\` (already created; llama.cpp `d854d98` and BitNet `0b341e5` were cloned there,
and the official llama.cpp release `b11889` zips (cpu-x64, cuda-12.4-x64, cudart-12.4) were downloaded to `dl\` with SHA-256 equal to the digest GitHub
reports: cpu `bf563dff...`, cuda-12.4 `ce07031c...`, cudart `8c79a9b2...`).

## 1. llama.cpp (official Windows binaries, no compilation)
```powershell
cd C:\PROFESSIONAL\Development\AI-HPC\ternix-competitors
Expand-Archive dl\llama-b11889-bin-win-cpu-x64.zip llama-b11889-cpu
Expand-Archive dl\llama-b11889-bin-win-cuda-12.4-x64.zip llama-b11889-cuda124
Expand-Archive dl\cudart-llama-bin-win-cuda-12.4-x64.zip llama-b11889-cuda124
```
Convert the model (needs llama.cpp's `convert_hf_to_gguf.py` from the cloned repo and the `gguf` Python package) and quantise to the ternary types:
```powershell
cd llama.cpp
python convert_hf_to_gguf.py ..\..\TernixEngine\models\bitnet-b1.58-2B-4T-bf16 --outfile ..\bitnet2b4t-f16.gguf --outtype f16
..\llama-b11889-cpu\llama-quantize.exe ..\bitnet2b4t-f16.gguf ..\bitnet2b4t-tq2_0.gguf TQ2_0
..\llama-b11889-cpu\llama-quantize.exe ..\bitnet2b4t-f16.gguf ..\bitnet2b4t-tq1_0.gguf TQ1_0
```
Check at load time that the log reports TQ2_0/TQ1_0 for the ternary matrices (llama.cpp silently falls back to another type for shapes not divisible by 256).
Benchmark (pin to P cores the same way: 16 threads):
```powershell
..\llama-b11889-cpu\llama-bench.exe -m ..\bitnet2b4t-tq2_0.gguf -t 16 -p 128 -n 64 -r 5 -o json
..\llama-b11889-cuda124\llama-bench.exe -m ..\bitnet2b4t-tq2_0.gguf -ngl 99 -p 128 -n 64 -r 5 -o json
```

## 2. bitnet.cpp (Microsoft; I2_S and TL kernels, needs clang from the VS installer)
```powershell
cd ..\BitNet
python -m pip install -r requirements.txt     # inside a uv-created venv, see security notes
python setup_env.py --hf-repo microsoft/BitNet-b1.58-2B-4T-gguf -q i2_s     # official GGUF (i2_s); use -q tl1 for the TL kernel on supported shapes
python utils\e2e_benchmark.py -m models\BitNet-b1.58-2B-4T\ggml-model-i2_s.gguf -p 128 -n 64 -t 16
```
(Open a *Developer PowerShell for VS* first. README: 2B4T is supported with `i2_s`.)

## 3. Add them to the harness
Create `bench/engines_competitors.json` in the schema of `bench/engines_ternix.json` (kind `llamabench` for the `llama-bench` JSON output), including
`"repo"` for the commit hash and `"bytes_per_token"` (file size of the GGUF is a good upper bound), then
```powershell
python bench/ternixbench.py speed --config bench/engines_all.json --rounds 9 --pp 128 --tg 64 --out docs/research/data/speed_all.json
python bench/ternixbench.py report docs/research/data/speed_all.json > docs/research/data/speed_all.md
```
Pitfalls to check and log for every engine (see `references/ANALYSIS.md` section 4): kernel actually used, bits per weight of the file, threads and
affinity, whether the head is quantised, llama.cpp version drift, thermal state. Report numbers next to the bytes-per-token column.

## GPU comparator for llama.cpp (source finding, 2026-10-26)

At the pinned commit (d854d98 / b11889) `ggml/src/ggml-cuda` contains no TQ2_0 or TQ1_0 code (Vulkan does), so `llama-bench -ngl 99` on a TQ2_0 file is not a measurement of
ternary inference on the GPU. `bench/run_competitors.ps1` therefore also quantises Q2_K and Q4_0 for the CUDA runs; report them with their bits per weight (not lossless for
ternary weights) and keep the TQ2_0 CUDA log (`llamacpp_cuda_tq2_0.log`) as evidence of what happened. For a ternary-aware GPU competitor use the bitnet.cpp GPU kernel
(`BitNet/gpu`, W2A8, GEMV only, built with nvcc); see `references/CODE.md`.

