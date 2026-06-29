# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Magenta RealTime 2 (MRT2): real-time music generation. Three components:

- `magenta_rt/` — Python inference library + the `mrt` CLI; has **JAX** and **MLX** backends.
- `core/` — C++20 streaming engine (`magentart::core`); public headers in `core/include/magentart/`.
- `examples/` — DAW plugins and apps (C++ hosts + TypeScript/React UI npm workspaces).

## Platform: this is a Linux checkout

The C++ build (`core/` and everything in `examples/`) links MLX/Metal/Apple
frameworks and is **macOS-only**; the **MLX** backend and real-time streaming
require **Apple Silicon**. On Linux only **offline JAX inference** is usable:

- Install the `jax` extra, not `mlx`: `uv pip install -e ".[jax,dev]"`.
- Use the `mrt jax ...` command group, not `mrt mlx ...`. `mrt models init` / `mrt models download` still apply.
- Do not `cmake`/build `core/` or `examples/` here — it will not configure off-macOS.

## Environment & build

- Package manager is **uv**, not pip/poetry: `uv venv --python 3.12 && uv pip install -e ".[jax,dev]"`.
- A git submodule is vendored at `magenta_rt/_vendor/sequence-layers`. After a plain clone, run `git submodule update --init --recursive` or imports fail.

## Code style

- Python is formatted with **pyink** (config in `pyproject.toml`): 80-column lines, **2-space** indentation, majority quotes — not black / PEP 8 4-space.
- Every new source file needs the Apache license header (`.py` `.cpp` `.cc` `.h` `.mm` `.m` `.sh` `.tsx`, and `CMakeLists.txt`). `pre-commit` inserts it via `insert-license`; run `pre-commit run --files <path>` (or `pre-commit install` once) so the header and whitespace fixes land.

## Testing

- `pytest tests/` (config in `pyproject.toml`); targeted, e.g. `pytest -s tests/test_musiccoca.py`.
- `tests/test_bitlevel_parity.py` and `tests/test_prefill_correctness.py` need downloaded checkpoints (`mrt models download`); without them they are skipped — expected, not a failure.
- Before `tests/test_bitlevel_parity.py`, regenerate references: `python scripts/generate_test_reference.py`.

## Commits

Short imperative summaries, optionally scoped (`docs:`, `[pd]`). Keep commits focused by subsystem; PRs target `main`.
