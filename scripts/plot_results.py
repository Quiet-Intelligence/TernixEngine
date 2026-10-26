"""Turn raw benchmark JSON into charts and README tables.

Inputs
  benchmarks/results.json       Google Benchmark output of bench_simd_math (CPU)
  benchmarks/results_cuda.json  written by `bench_cuda <path>` (GPU, optional)
Outputs
  benchmarks/benchmark_results.png   CPU: effective throughput per kernel and problem size
  benchmarks/benchmark_cuda.png      GPU: naive vs cp.async-tiled kernel (if the JSON exists)
  README.md                          tables between <!-- BENCH_CPU_START/END --> and <!-- BENCH_GPU_START/END -->
"""
import argparse
import json
import os
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CPU_IMPLS = [
    ("BM_NaiveTernary", "Naive (i-j-k, branching decode)"),
    ("BM_ScalarTernary", "Scalar (i-k-j, branch-free decode)"),
    ("BM_AVX2Ternary", "AVX2, one row at a time"),
    ("BM_AVX2BlockedTernary", "AVX2, 4-row register blocking"),
]
COLORS = ["#adb5bd", "#74c0fc", "#4c6ef5", "#1b3a8a"]


def load_cpu(path):
    """-> {impl_prefix: {size: median_cpu_ms}} (uses the median aggregate if repetitions were run)."""
    with open(path) as f:
        data = json.load(f)
    table = defaultdict(dict)
    for b in data["benchmarks"]:
        if b.get("run_type") == "aggregate" and b.get("aggregate_name") != "median":
            continue
        m = re.match(r"(BM_\w+?)(?:/(\d+))?(?:_median)?$", b["name"])
        if not m or m.group(2) is None:
            continue
        unit = {"ns": 1e-6, "us": 1e-3, "ms": 1.0, "s": 1e3}[b["time_unit"]]
        table[m.group(1)][int(m.group(2))] = b["cpu_time"] * unit
    return table, data.get("context", {})


def gops(size, ms):
    return 2.0 * size**3 / (ms * 1e-3) / 1e9


def plot_cpu(table, out_png):
    sizes = sorted({s for impl in table.values() for s in impl})
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.2
    for i, (key, label) in enumerate(CPU_IMPLS):
        vals = [gops(s, table[key][s]) if s in table.get(key, {}) else 0 for s in sizes]
        xs = [j + (i - 1.5) * width for j in range(len(sizes))]
        bars = ax.bar(xs, vals, width, label=label, color=COLORS[i])
        for bar, v in zip(bars, vals):
            if v:
                ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([f"{s}³" for s in sizes])
    ax.set_xlabel("Square problem size (M = K = N)")
    ax.set_ylabel("Effective throughput (G ternary-MAC ops/s, 2 ops per MAC)")
    ax.set_title("Ternary GEMM on one CPU thread (int8 activations, 2-bit weights)")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_gpu(gpu, out_png):
    rows = gpu["results"]
    ns = [r["n"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    xs = range(len(ns))
    ax.bar([x - 0.2 for x in xs], [r["naive_ms"] for r in rows], 0.4, label="naive (1 thread / output)", color="#adb5bd")
    ax.bar([x + 0.2 for x in xs], [r["tiled_ms"] for r in rows], 0.4, label="tiled, cp.async double buffer", color="#4c6ef5")
    for x, r in zip(xs, rows):
        ax.text(x + 0.2, r["tiled_ms"], f'{r["naive_ms"] / r["tiled_ms"]:.1f}×', ha="center", va="bottom", fontsize=9)
    ax.set_yscale("log")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{n}³" for n in ns])
    ax.set_xlabel("Square problem size (M = K = N)")
    ax.set_ylabel("Kernel time (ms, log scale)")
    ax.set_title(f"Ternary GEMM kernels on {gpu['device']}")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def cpu_markdown(table, ctx):
    sizes = sorted({s for impl in table.values() for s in impl})
    big = sizes[-1]
    base = table["BM_NaiveTernary"][big]
    lines = [
        f"CPU: {ctx.get('num_cpus', '?')} logical cores, one thread, release build. "
        "Seeded random int8 activations and random ternary weights (identical for every kernel).",
        "",
        "| Kernel | " + " | ".join(f"{s}³ ms" for s in sizes) + f" | G ops/s @ {big}³ | vs naive @ {big}³ |",
        "|---|" + "---:|" * (len(sizes) + 2),
    ]
    for key, label in CPU_IMPLS:
        if key not in table:
            continue
        t = table[key]
        cells = " | ".join(f"{t[s]:.2f}" if s in t else "-" for s in sizes)
        lines.append(f"| {label} | {cells} | {gops(big, t[big]):.2f} | {base / t[big]:.1f}× |")
    return "\n".join(lines)


def gpu_markdown(gpu):
    lines = [
        f"GPU: {gpu['device']} (sm_{gpu['sm']}). CUDA-event kernel time, data resident on device, median of 31 launches; "
        "every result is checked bit-for-bit against the CPU kernel.",
        "",
        "| Size | naive ms | tiled (cp.async) ms | speed-up | tiled effective TOPS |",
        "|---:|---:|---:|---:|---:|",
    ]
    for r in gpu["results"]:
        lines.append(f"| {r['n']}³ | {r['naive_ms']:.3f} | {r['tiled_ms']:.3f} | {r['naive_ms'] / r['tiled_ms']:.1f}× | {r['tiled_tops']:.2f} |")
    return "\n".join(lines)


def inject(readme_path, tag, body):
    start, end = f"<!-- {tag}_START -->", f"<!-- {tag}_END -->"
    with open(readme_path, encoding="utf-8") as f:
        text = f.read()
    if start not in text:
        print(f"marker {start} not found in {readme_path}; skipping")
        return
    text = re.sub(re.escape(start) + r".*?" + re.escape(end), lambda _: f"{start}\n{body}\n{end}", text, flags=re.S)
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="benchmarks/results.json")
    ap.add_argument("--cuda-input", default="benchmarks/results_cuda.json")
    ap.add_argument("--output", default="benchmarks")
    ap.add_argument("--readme", default="README.md")
    a = ap.parse_args()

    table, ctx = load_cpu(a.input)
    plot_cpu(table, os.path.join(a.output, "benchmark_results.png"))
    inject(a.readme, "BENCH_CPU", cpu_markdown(table, ctx))
    print("CPU chart and README table updated")

    if os.path.exists(a.cuda_input):
        with open(a.cuda_input) as f:
            gpu = json.load(f)
        plot_gpu(gpu, os.path.join(a.output, "benchmark_cuda.png"))
        inject(a.readme, "BENCH_GPU", gpu_markdown(gpu))
        print("GPU chart and README table updated")
