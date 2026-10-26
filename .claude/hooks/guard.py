"""PreToolUse guard for Bash: enforces the git workflow and commit hygiene. Stdlib only.

Exit 2 blocks the call and sends stderr back to the model; exit 0 allows it.
"""
import json
import re
import subprocess
import sys

PROTECTED = {"main", "develop"}
EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿⬀-⯿️]")
ATTRIBUTION = re.compile(r"co-authored-by|generated with|claude\.ai|noreply@anthropic", re.I)


def current_branch(cwd: str) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd, capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def block(msg: str) -> None:
    print(f"guard: {msg}", file=sys.stderr)
    sys.exit(2)


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return
    if payload.get("tool_name") != "Bash":
        return
    cmd = payload.get("tool_input", {}).get("command", "")
    cwd = payload.get("cwd") or "."
    if not re.search(r"\bgit\b", cmd):
        return

    if re.search(r"\bgit\s+commit\b", cmd):
        if current_branch(cwd) in PROTECTED:
            block("direct commits to main/develop are not allowed; create a feature/, fix/, perf/, docs/ or chore/ branch "
                  "and merge into develop with --no-ff.")
        if EMOJI.search(cmd):
            block("no emojis in commit messages.")
        if ATTRIBUTION.search(cmd):
            block("no attribution lines in commit messages.")
    if re.search(r"\bgit\s+push\b", cmd):
        if re.search(r"(--force\b|--force-with-lease|\s-f\b)", cmd):
            block("force push is not allowed.")
        if re.search(r"\bgit\s+push\b[^|;&]*\b(main|master)\b", cmd) or re.search(r"HEAD:(refs/heads/)?main\b", cmd):
            block("main is for releases only; push develop or a feature branch, release by merging develop into main "
                  "deliberately.")
    if re.search(r"\bgit\s+(reset\s+--hard|clean\s+-[a-z]*f|checkout\s+--\s|branch\s+-D)\b", cmd):
        block("destructive git command; confirm with the user first.")


if __name__ == "__main__":
    main()
