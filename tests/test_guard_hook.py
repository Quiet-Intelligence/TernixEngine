"""The Claude Code git guard (.claude/hooks/guard.py): blocks what the workflow forbids, allows what it does not."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parent.parent / ".claude" / "hooks" / "guard.py"


def verdict(command: str, cwd: str = ".") -> int:
    payload = json.dumps({"tool_name": "Bash", "tool_input": {"command": command}, "cwd": cwd})
    return subprocess.run([sys.executable, "-I", str(HOOK)], input=payload, text=True, capture_output=True).returncode


@pytest.mark.parametrize("cmd", [
    "git push --force origin feature/x",
    "git push -f origin feature/x",
    "git push origin main",
    "git push origin HEAD:main",
    "git reset --hard HEAD~1",
    "git branch -D old",
])
def test_blocked(cmd):
    assert verdict(cmd) == 2


@pytest.mark.parametrize("cmd", [
    "git push origin develop",
    "git push -u origin feature/x",
    "rm -f build/cache.txt && git push origin develop",     # -f belongs to rm, not to git push
    "test -f a.txt && git push origin perf/x",
    "git status",
    "ls -la",
    "git log --oneline | head -3",
])
def test_allowed(cmd):
    assert verdict(cmd) == 0


def test_commit_message_hygiene():
    assert verdict('git checkout -b feature/y && git commit -m "feat: ok \U0001F680"') == 2
    assert verdict('git checkout -b feature/y && git commit -m "feat: ok\n\nCo-Authored-By: someone"') == 2
    assert verdict('git checkout -b feature/y && git commit -m "feat: fine"') == 0
