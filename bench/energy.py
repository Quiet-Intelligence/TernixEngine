"""Energy per decoded token from hardware counters on Windows.

CPU: Windows "Energy Meter" RAPL_Package0_PKG power (mW, 1 s resolution; the DRAM counter reads 0 on this machine, so memory power is NOT included).
GPU: nvidia-smi board power.draw (W, 0.2 s). Idle power is measured first and subtracted; energy per token = (mean active - mean idle) x decode time / tokens.
Whole-system wall power is not measured (no meter), so figures are package/board only and are labelled as such.

    python bench/energy.py --config bench/engines_ternix.json --seconds 40 --out docs/research/data/energy.json
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import statistics
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class Sampler:
    def __init__(self) -> None:
        self.cpu: list[tuple[float, float]] = []
        self.gpu: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._procs: list[subprocess.Popen] = []

    def start(self) -> None:
        tp = subprocess.Popen(["typeperf", r"\Energy Meter(RAPL_Package0_PKG)\Power", "-si", "1"], stdout=subprocess.PIPE, text=True)
        gp = subprocess.Popen(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits", "-lms", "200"], stdout=subprocess.PIPE, text=True)
        self._procs = [tp, gp]
        threading.Thread(target=self._read_cpu, args=(tp,), daemon=True).start()
        threading.Thread(target=self._read_gpu, args=(gp,), daemon=True).start()

    def _read_cpu(self, p) -> None:
        for line in p.stdout:
            m = re.match(r'"[^"]+","([\d.]+)"', line.strip())
            if m:
                self.cpu.append((time.time(), float(m.group(1)) / 1000.0))

    def _read_gpu(self, p) -> None:
        for line in p.stdout:
            try:
                self.gpu.append((time.time(), float(line.strip())))
            except ValueError:
                pass

    def stop(self) -> None:
        for p in self._procs:
            p.terminate()


def mean_between(samples: list[tuple[float, float]], t0: float, t1: float) -> float:
    v = [w for t, w in samples if t0 <= t <= t1]
    return statistics.mean(v) if v else float("nan")


def background_load(seconds: int = 6) -> float:
    """Mean total CPU utilisation (%) over a few seconds with nothing of ours running."""
    out = subprocess.run(["typeperf", r"\Processor(_Total)\% Processor Time", "-sc", str(seconds)], capture_output=True, text=True).stdout
    v = [float(m.group(1)) for m in re.finditer(r'"[^"]+","([\d.]+)"', out)]
    return statistics.mean(v) if v else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seconds", type=int, default=40, help="approximate steady decode time per engine")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-background-cpu", type=float, default=5.0, help="abort when other processes keep the CPU busier than this (%%)")
    a = ap.parse_args()
    load = background_load()
    print(f"background CPU load before the run: {load:.1f} %")
    if not load <= a.max_background_cpu:
        raise SystemExit(f"aborting: {load:.1f} % background CPU load would contaminate package power and decode rates (limit {a.max_background_cpu} %)")
    engines = [e for e in json.loads(Path(a.config).read_text())["engines"] if e["kind"] == "ternix"]
    s = Sampler()
    s.start()
    time.sleep(3)
    t0 = time.time()
    time.sleep(20)                                   # idle baseline
    t1 = time.time()
    idle_cpu, idle_gpu = mean_between(s.cpu, t0, t1), mean_between(s.gpu, t0, t1)
    print(f"idle: package {idle_cpu:.1f} W, GPU {idle_gpu:.1f} W")
    out = {"idle_package_w": idle_cpu, "idle_gpu_w": idle_gpu, "background_cpu_pct": load, "runs": []}
    for e in engines:
        is_gpu = "--gpu" in e["cmd"]
        tg = 256
        probe = subprocess.run(e["cmd"] + ["--pp", "16", "--tg", "64", "--reps", "1"], capture_output=True, text=True).stdout
        tps = float(re.search(r"decode\s+\d+ tok: median ([\d.]+)", probe).group(1))
        reps = max(2, int(a.seconds * tps / tg))
        t_start = time.time()
        r = subprocess.run(e["cmd"] + ["--pp", "16", "--tg", str(tg), "--reps", str(reps)], capture_output=True, text=True)
        t_end = time.time()
        med = float(re.search(r"decode\s+\d+ tok: median ([\d.]+)", r.stdout).group(1))
        # steady window: skip the first 8 s (model load, head build) and the last 1 s
        w0, w1 = t_start + 8, t_end - 1
        p_cpu, p_gpu = mean_between(s.cpu, w0, w1), mean_between(s.gpu, w0, w1)
        # time per token from the measured median rate; energy above idle
        j_cpu = (p_cpu - idle_cpu) / med
        j_gpu = (p_gpu - idle_gpu) / med
        rec = {"name": e["name"], "decode_tok_s": med, "package_w": p_cpu, "gpu_w": p_gpu, "j_per_token_package_above_idle": j_cpu,
               "j_per_token_gpu_above_idle": j_gpu, "j_per_token_total_above_idle": j_cpu + j_gpu, "window_s": w1 - w0, "cpu_samples": len([1 for t, _ in s.cpu if w0 <= t <= w1]), "gpu_samples": len([1 for t, _ in s.gpu if w0 <= t <= w1])}
        out["runs"].append(rec)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
        time.sleep(15)                                # cool-down back to idle
    s.stop()
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
