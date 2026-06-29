---
name: mrt-bench
description: Run or inspect Magenta RT latency benchmarks via scripts/bench_track.py and scripts/bench_show.py. Use when the user asks to track, run, or view latency/throughput benchmark results for MRT2.
disable-model-invocation: true
---

# Magenta RT benchmarks

Two scripts, one shared log at `outputs/bench_track/benchmark_log.jsonl`.

## Inspect existing results (works anywhere, including Linux)

`scripts/bench_show.py` is pure Python — it only reads the log:

```bash
python scripts/bench_show.py            # latency table
python scripts/bench_show.py --samples  # also show last-10 audio samples
python scripts/bench_show.py --last 5   # only the last 5 runs
```

If the log is missing, no benchmark has been recorded in this checkout yet
(the log is produced by `bench_track.py` on macOS — see below).

## Record a new benchmark (macOS / Apple Silicon only)

`scripts/bench_track.py` runs a full **export → CMake build → run** pipeline
against the C++ `./benchmark_build/benchmark_mlxfn` binary, then appends a row to
the log. That binary links MLX/Metal, so **this only works on macOS with Apple
Silicon** — it will not run on this Linux checkout. Confirm the platform before
attempting it.

```bash
python scripts/bench_track.py                       # defaults: 8-bit, 100 steps
python scripts/bench_track.py --bits 8 --steps 100  # explicit
python scripts/bench_track.py --note "new depth decoder" --chip "M4 Max"
python scripts/bench_track.py --model-dir <dir>     # reuse an exported model (skips export)
```

## How to run this skill

`$ARGUMENTS` is passed through:

- empty, or starts with `show` → run `bench_show.py` (forward flags like `--samples`, `--last N`).
- starts with `track` → run `bench_track.py` with the remaining flags, but first confirm we are on macOS/Apple Silicon; if on Linux, stop and explain it cannot run here.
