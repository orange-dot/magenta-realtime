# JAX GPU Live Producer Runbook

This note records the local Linux/JAX GPU setup and the first successful
chunked live-producer measurement for the PC4MS full-ADG dataset.

It is a lab runbook, not a generic upstream installation guide. The important
local constraint is that runtime commands must run outside the Codex sandbox:
the sandbox can keep a stale `/dev` view and report missing NVIDIA devices even
when the host has already created them.

## Dataset Contract

Use the existing PC4MS full-ADG bundle dataset:

```text
/home/dev/work-base-20260421/workspace/systems/pc4-microkit-studio/.pc4ms/session-store/workbench-session/generated-drum-midi-takes/
```

Only `*.adg-bundle/manifest.json` entries with the full ADG contract are valid
for this benchmark. Do not call PC4MS to generate new ADG for this test, and do
not use older MIDI-only takes that lack ADG artifacts.

The measured two-minute run used:

```text
drum-live-1781270678428.adg-bundle/manifest.json
```

That take is long enough to cut to exactly `max_frames=3000` model frames
(`120.0` seconds at 25 model frames per second).

## Host GPU Setup

Create the NVIDIA device nodes on the host when they are missing:

```bash
sudo nvidia-modprobe -u -c=0
ls -l /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm
```

Expected device-node check from the successful run:

```text
crw-rw-rw-. 1 root root 195,   0 Jun 16 17:46 /dev/nvidia0
crw-rw-rw-. 1 root root 195, 255 Jun 16 17:46 /dev/nvidiactl
crw-rw-rw-. 1 root root 507,   0 Jun 16 17:50 /dev/nvidia-uvm
```

Verify that JAX sees the GPU from outside the sandbox:

```bash
cd /home/dev/work-base-20260421/workspace/systems/magenta-realtime
.venv/bin/python -c "import jax, json; print(json.dumps({'backend': jax.default_backend(), 'devices': [str(d) for d in jax.devices()]}, indent=2))"
```

Successful output:

```json
{
  "backend": "gpu",
  "devices": [
    "cuda:0"
  ]
}
```

## Render Worker

Start the JAX/GPU render worker first, so the model is loaded before submitting
the measurement request:

```bash
cd /home/dev/work-base-20260421/workspace/systems/magenta-realtime
env \
  MAGENTA_RT_RENDER_BACKEND=jax-gpu \
  MAGENTA_RT_MODEL=mrt2_small \
  MAGENTA_RT_CHECKPOINT=mrt2_small.safetensors \
  MAGENTA_RT_HOST=127.0.0.1 \
  MAGENTA_RT_PORT=18082 \
  MAGENTA_RT_WORKSPACE_ROOT=/home/dev/work-base-20260421 \
  MAGENTA_RT_MAGENTA_ROOT=/home/dev/Documents/Magenta/magenta-rt-v2 \
  MAGENTA_RT_CHUNK_FRAMES=8 \
  .venv/bin/python -m magenta_rt.render_worker
```

The successful run compiled and loaded the model with these worker logs:

```text
Compilation time: 78.4s
loaded backend=jax-gpu model=mrt2_small checkpoint=mrt2_small.safetensors in 81.69s
serving Magenta render worker backend=jax-gpu model=mrt2_small on 127.0.0.1:18082
```

The CUDA allocator can print transient OOM/rematerialization warnings during
JIT compilation. In this run those warnings did not mean the worker had failed.

Check readiness:

```bash
curl -fsS http://127.0.0.1:18082/health
```

The successful health response reported:

```json
{
  "backend": "jax-gpu",
  "checkpoint": "mrt2_small.safetensors",
  "model": "mrt2_small",
  "ok": true,
  "ready": true
}
```

The device was an NVIDIA RTX 500 Ada Generation Laptop GPU on `cuda:0`.

## Two-Minute Measurement

Submit the render request to the already-loaded worker:

```bash
cd /home/dev/work-base-20260421/workspace/systems/magenta-realtime
.venv/bin/python - <<'PY'
from __future__ import annotations

import json
from pathlib import Path
from urllib import request as urlrequest

run_dir = Path("/home/dev/work-base-20260421/runs/magenta-jax-2min-20260617-1246")
run_dir.mkdir(parents=True, exist_ok=True)

manifest = Path(
    "/home/dev/work-base-20260421/workspace/systems/pc4-microkit-studio/"
    ".pc4ms/session-store/workbench-session/generated-drum-midi-takes/"
    "drum-live-1781270678428.adg-bundle/manifest.json"
)

payload = {
    "adg_bundle_manifest": str(manifest),
    "output": str(run_dir / "render.wav"),
    "manifest": str(run_dir / "render.manifest.json"),
    "conditioning": str(run_dir / "conditioning.npz"),
    "prompt": "tight acoustic funk drums",
    "tail_seconds": 0.0,
    "max_frames": 3000,
    "temperature": 1.1,
    "top_k": 40,
    "cfg_musiccoca": 3.0,
    "cfg_notes": 3.0,
    "cfg_drums": 4.0,
}

body = json.dumps(payload).encode("utf-8")
req = urlrequest.Request(
    "http://127.0.0.1:18082/render/aig-adg",
    data=body,
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urlrequest.urlopen(req, timeout=1200) as resp:
    print(resp.read().decode("utf-8"))
PY
```

The HTTP result was:

```json
{
  "backend": "jax-gpu",
  "duration_seconds": 120.0,
  "elapsed_seconds": 98.05884575843811,
  "frames": 3000,
  "model": "mrt2_small",
  "ok": true,
  "steps_per_second": 30.593874288407534
}
```

The output artifacts are:

```text
/home/dev/work-base-20260421/runs/magenta-jax-2min-20260617-1246/render.wav
/home/dev/work-base-20260421/runs/magenta-jax-2min-20260617-1246/render.manifest.json
/home/dev/work-base-20260421/runs/magenta-jax-2min-20260617-1246/conditioning.npz
```

## Result Summary

| Metric | Value |
| --- | ---: |
| Source kind | `pc4ms_full_adg_take` |
| Take ID | `drum-live-1781270678428` |
| Model frames | `3000` |
| Audio duration | `120.0 s` |
| Conditioning events | `2160` |
| Mapped events | `1249` |
| Producer mode | `chunked` |
| Chunk frames | `8` |
| Chunks | `375` |
| HTTP elapsed | `98.05884575843811 s` |
| HTTP steps/s | `30.593874288407534` |
| Producer total | `95.6106805519994 s` |
| Producer steps/s | `31.377247632584332` |
| Producer RTF | `0.7967556712666616` |
| Hot-loop prompt embedding | `false` |
| WAV size | `23040044 bytes` |

This clears the first live-preview measurement gate:

```text
steps/s >= 30
RTF <= 0.833
```

The run used chunked producer mode with `chunk_frames=8`. Ring writes were not
part of this HTTP render-worker measurement, so `ring_write_seconds=0.0` and
`low_water_frames=null` in the producer metrics.

## Follow-Up: Low-Latency Chunk Sweep

After the accepted `chunk_frames=8` worker gate, a lower-latency sweep tested
`chunk_frames=1`, `2`, `4`, and `8` with the null-host ring consumer on the
same full-ADG take. A follow-up two-minute run then confirmed the lower-latency
candidates `2` and `4`.

Short 60-second sweep:

| Mode | Chunk | Producer steps/s | RTF | Total seconds |
| --- | ---: | ---: | ---: | ---: |
| `chunked` | `1` | `24.61165155804132` | `1.0157790484333342` | `60.946742906000054` |
| `chunked` | `2` | `25.60348219330699` | `0.9764296829333338` | `58.585780976000024` |
| `chunked` | `4` | `26.415843943189216` | `0.946401714583332` | `56.78410287499992` |
| `chunked` | `8` | `26.75183165253764` | `0.9345154501833349` | `56.070927011000094` |
| `pipeline-1` | `1` | `24.625242789377968` | `1.0152184168833325` | `60.91310501299995` |

Two-minute confirmation:

| Mode | Chunk | Producer steps/s | RTF | Total seconds |
| --- | ---: | ---: | ---: | ---: |
| `chunked` | `2` | `25.750224080812607` | `0.9708653377749973` | `116.50384053299967` |
| `chunked` | `4` | `26.503488844425657` | `0.9432720403999989` | `113.19264484799987` |

The realtime floor is `25 steps/s` because each model frame is 40 ms. The live
preview safety gate is stricter: `>=30 steps/s` and `RTF <= 0.833`.

Conclusion: `chunk=1` and `pipeline-1` are not viable live choices on this
path. `chunk=2` and `chunk=4` stayed ahead of realtime in the null-host run,
but neither clears the safety gate. `chunk=4` is the best lower-latency
compromise from this sweep; `chunk=8` remains the safer default when the
priority is robust preview throughput.

The experiment artifact is:

```text
/home/dev/work-base-20260421/runs/magenta-jax-latency-sweep-20260617/
```

## Follow-Up: Chunk-Drain Lag Experiment

`magenta_rt/live_pipeline.py` suggested a useful hypothesis: dispatch the next
unit of work before draining the previous audio output. The frame-level version
already exists as producer mode `pipeline-1`; the default live-preview path uses
chunked materialization, so the comparable experiment was to delay
`device_get()` by one chunk:

```text
baseline chunked:      dispatch chunk i -> device_get chunk i
experimental chunked: dispatch chunk i -> device_get chunk i-1
```

That experiment was run on 2026-06-17 against the same already-loaded JAX/GPU
worker, same full ADG take, same `max_frames=3000`, same `chunk_frames=8`.

| Metric | Baseline | Chunk-drain lag |
| --- | ---: | ---: |
| HTTP elapsed | `98.05884575843811 s` | `98.50388717651367 s` |
| HTTP steps/s | `30.593874288407534` | `30.45565089856973` |
| Producer total | `95.6106805519994 s` | `96.05565737500001 s` |
| Producer steps/s | `31.377247632584332` | `31.231892862781002` |
| Producer RTF | `0.7967556712666616` | `0.8004638114583335` |
| Dispatch seconds | `49.09452562901788` | `49.32876185202986` |
| Copy seconds | `46.385874429997784` | `46.58473168101773` |

Conclusion: chunk-drain lag did not improve this hardware path. The prototype
patch was reverted, leaving the original default `chunked` producer in place.
The result supports the current interpretation that the main gain has already
come from precomputing conditioning and batching materialization by chunk; the
remaining wall time is dominated by the dependent JAX/GPU streaming step.

The experiment artifact is:

```text
/home/dev/work-base-20260421/runs/magenta-jax-2min-chunkpipe-20260617-1302/
```

## Follow-Up: Device-Native `scan-chunked`

`scan-chunked` is an experimental producer mode that builds one device token
matrix for the conditioning schedule and executes each chunk with a JAX
`lax.scan` over dependent streaming steps. The normal worker default remains
`chunked`; the scan mode must be selected explicitly with:

```bash
MAGENTA_RT_PRODUCER_MODE=scan-chunked
```

The short GPU parity check used the same fresh initial state for `chunked` and
`scan-chunked` on a 16-frame full-ADG schedule. Shapes matched, but exact int16
output did not:

```json
{
  "equal": false,
  "max_abs_diff": 1483,
  "shape_chunked": [30720, 2],
  "shape_scan": [30720, 2]
}
```

The full two-minute scan experiment used one loaded model and measured
baseline `chunked-8`, then `scan-chunked` with chunk sizes `4`, `8`, and `16`.
The GPU was visibly under-occupied during this run and the same-process
baseline was degraded compared with the earlier accepted worker measurement.

| Mode | Chunk | Producer steps/s | RTF | Total seconds |
| --- | ---: | ---: | ---: | ---: |
| `chunked` | `8` | `26.73408723984467` | `0.9351357230083333` | `112.21628676099999` |
| `scan-chunked` | `4` | `28.268280317279636` | `0.8843834757333354` | `106.12601708800025` |
| `scan-chunked` | `8` | `29.353957134172838` | `0.8516739288583305` | `102.20087146299966` |
| `scan-chunked` | `16` | `29.92112549198561` | `0.8355300674333345` | `100.26360809200014` |

Conclusion: `scan-chunked-16` improved over that degraded same-run baseline by
about `11.9%`, but it did not clear the live `>=30 steps/s` line, did not beat
the earlier accepted `31.377247632584332 steps/s` baseline, and did not preserve
exact int16 parity. Keep it as an explicit experiment only.

The experiment artifact is:

```text
/home/dev/work-base-20260421/runs/magenta-jax-scan-experiment-20260617-1410/
```

## Implementation Verification

The producer implementation was checked before the run with:

```bash
cd /home/dev/work-base-20260421/workspace/systems/magenta-realtime
.venv/bin/python -m py_compile \
  magenta_rt/adg_dataset.py \
  magenta_rt/aig_bridge.py \
  magenta_rt/live_ring.py \
  magenta_rt/jax/live_producer.py \
  magenta_rt/render_worker.py \
  scripts/aig_adg_jax_render.py \
  scripts/bench_jax_live_producer.py \
  tests/test_adg_dataset.py \
  tests/test_aig_bridge.py \
  tests/test_live_ring.py \
  tests/test_jax_live_producer.py \
  tests/test_render_worker.py

.venv/bin/pytest -q \
  tests/test_adg_dataset.py \
  tests/test_aig_bridge.py \
  tests/test_live_ring.py \
  tests/test_jax_live_producer.py \
  tests/test_render_worker.py
```

Observed pytest result:

```text
20 passed in 0.04s
```

Dataset discovery found 25 full ADG takes under the PC4MS session-store root.
The first failed two-minute attempt used the latest take
`drum-live-1781609361176`, but that take only produced 790 frames (`31.6 s`)
with `max_frames=3000`. The successful measurement therefore used the longer
full ADG take `drum-live-1781270678428`.
