"""TernixBench: a fair cross-engine speed and quality harness for ternary LLM inference.

Design rules (see references/ANALYSIS.md section 4):
  * every number is end-to-end (head and KV cache included) and tagged with bits per weight, file size and bytes moved per token;
  * engines are run interleaved in shuffled order for N rounds, so drift of clocks/thermals hits all alike; median, min, max kept;
  * the memory-bandwidth ceiling is measured in the same session (bench_membw) and reported as roofline fraction;
  * competitor builds are pinned (commit hash logged); the kernel actually used is recorded where the engine reports it;
  * quality is reported as perplexity plus KL/flips against the float32 reference with the EXP-001 noise floor next to it.

    python bench/ternixbench.py env  --out results/env.json
    python bench/ternixbench.py speed --config bench/engines.json --rounds 9 --out results/speed.json
    python bench/ternixbench.py report results/speed.json > results/speed.md

engines.json: {"model_bytes_per_token": {...}, "engines": [{"name", "kind": "ternix|llamabench", "cmd": [...], "model": "...", "bpw": 2.0, ...}]}
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str], timeout: int = 1800) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed: {r.stderr[-400:]}")
    return r.stdout + r.stderr


def git_head(path: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip() or None
    except Exception:
        return None


def collect_env(membw_exe: str | None) -> dict:
    env = {"date": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "platform": platform.platform(), "python": sys.version.split()[0],
           "ternix_commit": git_head(ROOT)}
    try:
        env["cpu"] = run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"]).strip()
        env["ram"] = run(["powershell", "-NoProfile", "-Command",
                          "Get-CimInstance Win32_PhysicalMemory | ForEach-Object { '{0} GB @ {1} MT/s' -f ($_.Capacity/1GB), $_.ConfiguredClockSpeed }"]).split()
        env["gpu"] = run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,clocks.max.sm", "--format=csv,noheader"]).strip()
    except Exception as e:  # environment probes are best effort
        env["probe_error"] = str(e)
    if membw_exe:
        out = run([str(Path(membw_exe).resolve())])
        env["membw_gbps"] = {int(m[0]): float(m[1]) for m in re.findall(r"threads\s+(\d+):\s+([\d.]+) GB/s", out)}
    return env


def parse_ternix_bench(out: str) -> dict:
    pp = re.search(r"prefill (\d+) tok: median ([\d.]+) tok/s", out)
    tg = re.search(r"decode\s+(\d+) tok: median ([\d.]+) tok/s", out)
    if not (pp and tg):
        raise RuntimeError("cannot parse ternix bench output:\n" + out[-300:])
    return {"pp_tps": float(pp.group(2)), "tg_tps": float(tg.group(2))}


def parse_llama_bench(out: str, pp: int, tg: int) -> dict:
    """llama-bench -o json prints a JSON array with one entry per test (n_prompt/n_gen, avg_ts)."""
    start = out.index("[")
    rows = json.loads(out[start:out.rindex("]") + 1])
    res = {}
    for r in rows:
        if r.get("n_prompt") == pp and r.get("n_gen") == 0:
            res["pp_tps"] = r["avg_ts"]
        if r.get("n_gen") == tg and r.get("n_prompt") == 0:
            res["tg_tps"] = r["avg_ts"]
    return res


def one_run(e: dict, pp: int, tg: int) -> dict:
    if e["kind"] == "ternix":
        return parse_ternix_bench(run(e["cmd"] + ["--pp", str(pp), "--tg", str(tg), "--reps", "1"]))
    if e["kind"] == "llamabench":
        return parse_llama_bench(run(e["cmd"] + ["-p", str(pp), "-n", str(tg), "-r", "1", "-o", "json"]), pp, tg)
    raise ValueError(e["kind"])


def cmd_speed(args) -> None:
    cfg = json.loads(Path(args.config).read_text())
    engines = cfg["engines"]
    results = {e["name"]: {"pp_tps": [], "tg_tps": []} for e in engines}
    rng = random.Random(7)
    for r in range(-1 if args.warmup else 0, args.rounds):
        order = engines[:]
        rng.shuffle(order)
        for e in order:
            try:
                m = one_run(e, args.pp, args.tg)
            except Exception as ex:
                print(f"[{e['name']}] failed: {ex}", file=sys.stderr)
                continue
            if r >= 0:
                for k, v in m.items():
                    results[e["name"]][k].append(v)
            print(f"round {r} {e['name']}: {m}", file=sys.stderr, flush=True)
    out = {"env": collect_env(args.membw), "pp": args.pp, "tg": args.tg, "rounds": args.rounds, "engines": []}
    for e in engines:
        res = results[e["name"]]
        rec = {k: e[k] for k in ("name", "kind", "model", "bpw", "bytes_per_token", "build", "notes") if k in e}
        rec["commit"] = git_head(Path(e["repo"])) if e.get("repo") else None
        for k in ("pp_tps", "tg_tps"):
            v = res[k]
            if v:
                rec[k] = {"median": statistics.median(v), "min": min(v), "max": max(v), "n": len(v), "all": v}
        out["engines"].append(rec)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))


def cmd_report(args) -> None:
    d = json.loads(Path(args.json).read_text())
    ceil = max(d["env"].get("membw_gbps", {"0": 0}).values()) if d["env"].get("membw_gbps") else None
    print(f"Machine: {d['env'].get('cpu')} | {d['env'].get('gpu')} | measured read ceiling {ceil} GB/s | ternix {d['env'].get('ternix_commit')}\n")
    print(f"pp{d['pp']} / tg{d['tg']}, {d['rounds']} interleaved rounds (median, min..max)\n")
    print("| engine | bits/weight | bytes/token | prefill tok/s | decode tok/s | decode GB/s | % of ceiling |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for e in d["engines"]:
        pp, tg = e.get("pp_tps"), e.get("tg_tps")
        bpt = e.get("bytes_per_token")
        gbps = tg["median"] * bpt / 1e9 if (tg and bpt) else None
        print(f"| {e['name']} | {e.get('bpw', '')} | {bpt / 1e6:.0f} MB |" if bpt else f"| {e['name']} | {e.get('bpw', '')} | |", end="")
        print(f" {pp['median']:.1f} ({pp['min']:.1f}..{pp['max']:.1f}) |" if pp else " |", end="")
        print(f" {tg['median']:.2f} ({tg['min']:.2f}..{tg['max']:.2f}) |" if tg else " |", end="")
        print(f" {gbps:.1f} | {100 * gbps / ceil:.0f}% |" if (gbps and ceil) else " | |")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("env")
    e.add_argument("--membw", default=str(ROOT / "build-dev" / "bench_membw.exe"))
    e.add_argument("--out")
    s = sub.add_parser("speed")
    s.add_argument("--config", required=True)
    s.add_argument("--rounds", type=int, default=9)
    s.add_argument("--pp", type=int, default=128)
    s.add_argument("--tg", type=int, default=64)
    s.add_argument("--warmup", action="store_true", default=True)
    s.add_argument("--membw", default=str(ROOT / "build-dev" / "bench_membw.exe"))
    s.add_argument("--out", required=True)
    r = sub.add_parser("report")
    r.add_argument("json")
    a = ap.parse_args()
    if a.cmd == "env":
        env = collect_env(a.membw)
        print(json.dumps(env, indent=1))
        if a.out:
            Path(a.out).write_text(json.dumps(env, indent=1))
    elif a.cmd == "speed":
        cmd_speed(a)
    else:
        cmd_report(a)


if __name__ == "__main__":
    main()
