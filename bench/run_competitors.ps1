# One-shot competitor run for BitNet b1.58 2B4T on this machine. Run it yourself:   ! powershell -File bench\run_competitors.ps1
# (Claude Code's auto-mode denied building/executing third-party code, so this is left to the repository owner.)
# Steps: verify the downloaded llama.cpp release hashes, extract, convert the model to GGUF (TQ2_0 / TQ1_0), run llama-bench on CPU and GPU,
# store raw JSON under docs\research\data\competitors\. bitnet.cpp is optional (see bench\COMPETITORS.md section 2).
$ErrorActionPreference = "Stop"
# native commands do not honour $ErrorActionPreference: stop at the first non-zero exit code instead of benchmarking a missing file
function Run { param([string]$exe, [string[]]$a) & $exe @a; if ($LASTEXITCODE -ne 0) { throw "failed (exit $LASTEXITCODE): $exe $($a -join ' ')" } }
$root = "C:\PROFESSIONAL\Development\AI-HPC"
$comp = "$root\ternix-competitors"
$out  = "$root\TernixEngine\docs\research\data\competitors"
New-Item -ItemType Directory -Force $out | Out-Null

# 1. hashes must equal the digests GitHub reports for release b11889 (taken from the API on 2026-10-26)
$expected = @{
  "llama-b11889-bin-win-cpu-x64.zip"         = "bf563dffb502a230ae06c9dc455b54cbc079e7e0bc270cd4d6191209b28d175d"
  "llama-b11889-bin-win-cuda-12.4-x64.zip"   = "ce07031c3e4ab2a98d8b6addf1937b9f60d743208a569d3aad116cd25d6a41bd"
  "cudart-llama-bin-win-cuda-12.4-x64.zip"   = "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6"
}
foreach ($k in $expected.Keys) {
  $h = (Get-FileHash "$comp\dl\$k" -Algorithm SHA256).Hash.ToLower()
  if ($h -ne $expected[$k]) { throw "hash mismatch for $k : $h" }
}
Expand-Archive "$comp\dl\llama-b11889-bin-win-cpu-x64.zip" "$comp\llama-b11889-cpu" -Force
Expand-Archive "$comp\dl\llama-b11889-bin-win-cuda-12.4-x64.zip" "$comp\llama-b11889-cuda124" -Force
Expand-Archive "$comp\dl\cudart-llama-bin-win-cuda-12.4-x64.zip" "$comp\llama-b11889-cuda124" -Force

# 2. convert the bf16 checkpoint with llama.cpp's own converter (run inside the project venv; installs only the converter's dependency)
$py = "$root\TernixEngine\.venv\Scripts\python.exe"
Run "uv" @("pip", "install", "--python", $py, "gguf", "sentencepiece")
# llama.cpp's BitNet converter loads the vocabulary with _set_vocab_sentencepiece(), which needs a tokenizer.model; the 2B4T checkpoint only has the Llama-3 BPE
# tokenizer.json (the converter stops with FileNotFoundError). The vocabulary does not influence llama-bench timings, so switch to the BPE path and record the diff.
$bn = "$comp\llama.cpp\conversionitnet.py"
(Get-Content $bn -Raw) -replace 'self\._set_vocab_sentencepiece\(\)', 'self._set_vocab_gpt2()' | Set-Content $bn -NoNewline
& git -C "$comp\llama.cpp" diff | Out-File -Encoding utf8 "$out\converter_patch.diff"
Run $py @("$comp\llama.cpp\convert_hf_to_gguf.py", "$root\TernixEngine\models\bitnet-b1.58-2B-4T-bf16", "--outfile", "$comp\bitnet2b4t-f16.gguf", "--outtype", "f16")
foreach ($t in "TQ2_0", "TQ1_0") {
  Run "$comp\llama-b11889-cpu\llama-quantize.exe" @("$comp\bitnet2b4t-f16.gguf", "$comp\bitnet2b4t-$($t.ToLower()).gguf", $t)
}

# 3. benchmarks: 7 repetitions each, 16 threads (P cores), pp128 / tg64; the log shows which quant type each tensor really got
foreach ($t in "tq2_0", "tq1_0") {
  $m = "$comp\bitnet2b4t-$t.gguf"
  & "$comp\llama-b11889-cpu\llama-bench.exe" -m $m -t 16 -p 128 -n 64 -r 7 -o json | Out-File -Encoding utf8 "$out\llamacpp_cpu_$t.json"; if ($LASTEXITCODE -ne 0) { throw "llama-bench (cpu, $t) failed" }
  & "$comp\llama-b11889-cuda124\llama-bench.exe" -m $m -ngl 99 -p 128 -n 64 -r 7 -o json | Out-File -Encoding utf8 "$out\llamacpp_cuda_$t.json"; if ($LASTEXITCODE -ne 0) { throw "llama-bench (cuda, $t) failed" }
  Get-Item $m | Select-Object Name, Length | Out-File -Append "$out\sizes.txt"
}
"b11889" | Out-File "$out\llamacpp_version.txt"
Write-Host "done: raw results in $out. Ask Claude to fold them into the harness, README and report."
