"""Download a Hugging Face model snapshot into models/<name>/ and verify every LFS file against the sha256 the Hub
reports. The SSL context in huggingface_hub is certifi-based; the hash check is the second, independent line of defence
(see CLAUDE.md, security section).

    uv run scripts/fetch_model.py microsoft/bitnet-b1.58-2B-4T-bf16 [--files model.safetensors tokenizer.json ...]
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("--files", nargs="*", help="subset of files; default: all")
    ap.add_argument("--out", default="models")
    args = ap.parse_args()

    info = HfApi().model_info(args.repo, files_metadata=True)
    out = Path(args.out) / args.repo.split("/")[-1]
    out.mkdir(parents=True, exist_ok=True)
    for sib in info.siblings:
        if args.files and sib.rfilename not in args.files:
            continue
        local = Path(hf_hub_download(args.repo, sib.rfilename, local_dir=out, revision=info.sha))
        if sib.lfs is not None:
            got = sha256_file(local)
            if got != sib.lfs.sha256:
                raise SystemExit(f"sha256 mismatch for {sib.rfilename}: {got} != {sib.lfs.sha256}")
            print(f"verified {sib.rfilename} sha256={got[:16]}...")
        else:
            print(f"downloaded {sib.rfilename} (not LFS, no hub hash)")
    print(f"revision {info.sha} -> {out}")


if __name__ == "__main__":
    main()
