"""EXP-005: decode throughput vs thread count, core pinning and chunk scheduling on the hybrid CPU."""
import itertools, json, os, re, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
cli = str(ROOT / "build-dev" / "ternix_cli.exe")
model = str(ROOT / "models" / "bitnet-2b4t.ternix")
rows = []
for threads, pin, sched in itertools.product([4, 8, 12, 16, 20, 24], ["p", "none", "all"], ["dynamic", "static"]):
    env = dict(os.environ, TERNIX_PIN=pin, TERNIX_SCHED=sched)
    out = subprocess.run([cli, "bench", model, "--pp", "16", "--tg", "48", "--reps", "7", "--head", "int8", "--threads", str(threads)], env=env,
                         capture_output=True, text=True).stdout
    m = re.search(r"decode\s+\d+ tok: median ([\d.]+) tok/s \(min ([\d.]+) max ([\d.]+)\)", out)
    rec = {"threads": threads, "pin": pin, "sched": sched, "median": float(m.group(1)), "min": float(m.group(2)), "max": float(m.group(3))}
    rows.append(rec)
    print(rec, flush=True)
Path(ROOT / "docs/research/data/exp005_sched.json").write_text(json.dumps(rows, indent=1))
