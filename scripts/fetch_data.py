"""Fetch evaluation text into data/ with sha256 verification against the Hub-reported LFS hashes.

    uv run scripts/fetch_data.py wikitext2      # WikiText-2 raw test split -> data/wikitext2_test.txt
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def wikitext2(out: Path) -> None:
    import pyarrow.parquet as pq
    repo, fn = "Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet"
    info = HfApi().dataset_info(repo, files_metadata=True)
    sib = next(s for s in info.siblings if s.rfilename == fn)
    path = Path(hf_hub_download(repo, fn, repo_type="dataset", revision=info.sha, local_dir=out / "_hf"))
    if sib.lfs is not None and sha256_file(path) != sib.lfs.sha256:
        raise SystemExit("sha256 mismatch for " + fn)
    rows = pq.read_table(path).column("text").to_pylist()
    # standard "\n\n".join over the rows as in lm-eval / HF perplexity recipes
    (out / "wikitext2_test.txt").write_text("".join(rows), encoding="utf-8", newline="")
    print(f"wikitext2 test: {len(rows)} rows, revision {info.sha}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["wikitext2"])
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    Path(a.out).mkdir(exist_ok=True)
    wikitext2(Path(a.out))
