---
description: Build and run all tests, report pass/fail faithfully
---
Run, in order, and report the actual output summary of each (do not claim success for a step that was not run):
1. `cmd //c scripts\build.bat build`
2. `ctest --test-dir build --output-on-failure`
3. `PYTHONPATH=build python -m pytest tests -q` (if the binding built for the active Python)
4. If model code changed: `python scripts/verify_logits.py` against the reference.
List any failing test names and the first error line.
