---
name: mrt-test
description: Run the Magenta RT Python test suite the project's way — full or targeted pytest, with coverage and the checkpoint / bit-level-parity quirks handled. Use when asked to run tests, check the suite, or verify Python changes in this repo.
---

# Magenta RT testing

Config lives in `pyproject.toml` (`testpaths = ["tests"]`, `python_files = ["test_*.py"]`).

## Default: run the suite with coverage

```bash
pytest tests/ --cov=magenta_rt --cov-report=term
```

## Targeted runs

```bash
pytest -s tests/test_musiccoca.py
pytest -s tests/test_gptq.py
```

Available test files: `test_musiccoca.py`, `test_gptq.py`, `test_bitlevel_parity.py`, `test_prefill_correctness.py`.

## Checkpoint-gated tests (expected to skip without checkpoints)

`tests/test_bitlevel_parity.py` and `tests/test_prefill_correctness.py` require
downloaded checkpoints (`mrt models download`). Without them they are **skipped**
— that is the expected state in a fresh checkout / CI, not a failure. Do not
report skips here as broken.

## Bit-level parity workflow

Regenerate references **before** running the parity test, or it will compare
against stale data:

```bash
python scripts/generate_test_reference.py
pytest -s tests/test_bitlevel_parity.py
```

## How to run this skill

If `$ARGUMENTS` names a test path or expression (e.g. `tests/test_gptq.py` or
`-k musiccoca`), run exactly that with `pytest -s`. Otherwise run the default
coverage command above, and summarize pass / fail / skip counts — calling out
that any checkpoint-gated skips are expected.
