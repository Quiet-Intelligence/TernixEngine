"""Merge repeated bench/energy.py outputs into one summary (per-engine values of every pass) and print a markdown table."""
import json
import sys
from pathlib import Path

runs = [json.loads(Path(p).read_text()) for p in sys.argv[2:]]
out = {"idle_package_w": [r["idle_package_w"] for r in runs], "idle_gpu_w": [r["idle_gpu_w"] for r in runs],
       "background_cpu_pct": [r.get("background_cpu_pct") for r in runs], "engines": {}}
for r in runs:
    for e in r["runs"]:
        d = out["engines"].setdefault(e["name"], {"decode_tok_s": [], "package_w": [], "gpu_w": [], "j_per_token_total_above_idle": []})
        for k in d:
            d[k].append(e[k])
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
print("| engine | decode tok/s | package W | GPU W | J/token above idle (package + GPU) |\n|---|---:|---:|---:|---:|")
f = lambda v, n=1: " / ".join(f"{x:.{n}f}" for x in v)
for name, d in out["engines"].items():
    print(f"| {name} | {f(d['decode_tok_s'])} | {f(d['package_w'])} | {f(d['gpu_w'])} | {f(d['j_per_token_total_above_idle'], 2)} |")
