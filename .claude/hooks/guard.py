"""PreToolUse guard for Bash: enforces the git workflow and commit hygiene. Stdlib only.

Exit 2 blocks the call and sends stderr back to the model; exit 0 allows it.
Each simple command of a compound line (split on && || ; | newline) is judged on its own, so flags such as `rm -f` or
`test -f` are never attributed to a neighbouring `git push`.
"""
import json
import re
import subprocess
import sys

PROTECTED = {"main", "develop"}
EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿⬀-⯿️]")
ATTRIBUTION = re.compile(r"co-authored-by|generated with|claude\.ai|noreply@anthropic", re.I)
SPLIT = re.compile(r"&&|\|\||;|\||\n")


def current_branch(cwd: str) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd, capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def block(msg: str) -> None:
    print(f"guard: {msg}", file=sys.stderr)
    sys.exit(2)


def check_segment(seg: str, cwd: str, later_checkout: bool) -> None:
    if re.match(r"git\s+commit\b", seg):
        if current_branch(cwd) in PROTECTED and not later_checkout:
            block("direct commits to main/develop are not allowed; create a feature/, fix/, perf/, docs/ or chore/ branch "
                  "and merge into develop with --no-ff.")
    if re.match(r"git\s+push\b", seg):
        if re.search(r"(--force\b|--force-with-lease|\s-f\b)", seg):
            block("force push is not allowed.")
        if re.search(r"\b(main|master)\b", seg) or re.search(r"HEAD:(refs/heads/)?main\b", seg):
            block("main is for releases only; push develop or a feature branch, release by merging develop into main "
                  "deliberately.")
    if re.match(r"git\s+(reset\s+--hard|clean\s+-[a-z]*f|checkout\s+--\s|branch\s+-D)\b", seg):
        block("destructive git command; confirm with the user first.")


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
    segs = [s.strip() for s in SPLIT.split(cmd) if s.strip()]
    if any(re.match(r"git\s+commit\b", s) for s in segs):   # messages may span lines: judge the whole command text
        if EMOJI.search(cmd):
            block("no emojis in commit messages.")
        if ATTRIBUTION.search(cmd):
            block("no attribution lines in commit messages.")
    for i, seg in enumerate(segs):
        # a commit that follows a branch switch/creation in the same line is judged on the new branch, which we cannot know yet
        later_checkout = any(re.match(r"git\s+(checkout|switch)\b.*\s-[bcBC]\b", p) for p in segs[:i])
        check_segment(seg, cwd, later_checkout)


if __name__ == "__main__":
    main()
