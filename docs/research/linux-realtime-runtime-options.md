# Linux Real-Time Runtime Options

Status: discussion draft, not a decision.

Date: 2026-06-17

This note captures the current design discussion around building a Linux
real-time Magenta/AIG runtime, using the existing Magenta RealTime 2 code where
it helps and writing new Linux pieces where that fits better.

The short version: the existing macOS C++ `RealtimeRunner` is the best
architectural reference, but the current Linux machine has already proven a
fast JAX/CUDA path. The next useful work is to test whether the official
Magenta C++/MLX path can be ported to Linux/CUDA before committing to a custom
JAX-daemon architecture.

## Current Evidence

Local measurements with `mrt2_small`, model loaded once:

| Backend | Chunk | Audio duration | Render time | Speed |
|---|---:|---:|---:|---:|
| JAX GPU / CUDA | 50 frames | 2.0 s | 2.31 s | 21.6 frames/s |
| JAX GPU / CUDA | 125 frames | 5.0 s | 4.69 s | 26.7 frames/s |
| JAX GPU / CUDA | 1500 frames | 60.0 s | 53.8 s | 27.9 frames/s |
| MLX CPU / Linux raw checkpoint | 50 frames | 2.0 s | 212.4 s | 0.235 frames/s |

The model frame rate is 25 Hz. One frame is 1920 stereo samples at 48 kHz,
or 40 ms of audio.

That means:

- JAX/CUDA is already faster than real-time for longer chunks on this machine.
- MLX CPU is roughly two orders of magnitude too slow for real-time.
- The measured JAX path is still an offline/render-ahead path, not an
  audio-callback-safe runtime.

## What "Real-Time" Means Here

We should separate three claims:

1. **Audio real-time**
   The audio callback never blocks on Python, HTTP, disk, model compilation,
   GPU execution, locks, or allocation. It only reads already-produced samples
   from a real-time-safe buffer.

2. **Generation real-time**
   The model produces at least 25 frames/s on average, with enough headroom
   that buffer underruns are rare under normal load.

3. **Interactive real-time**
   User actions such as MIDI, ADG edits, transport changes, prompt changes, or
   CFG changes become audible quickly enough to feel playable.

The current JAX worker has shown generation real-time for long chunks. It does
not yet provide audio real-time or low-latency interactive real-time.

## Existing Magenta Real-Time Design

The existing Magenta C++ path is centered on:

- `magentart::core::MLXEngine`
- `magentart::core::RealtimeRunner`
- `magentart::core::RingBuffer`

Important properties from the local code:

- `MLXEngine::generate_frame()` produces one 1920-sample stereo frame.
- `RealtimeRunner` runs an inference thread at the 25 Hz model cadence.
- The audio callback calls `read_audio_stereo()`.
- Audio uses lock-free single-producer/single-consumer ring buffers.
- Sampling parameters, MIDI notes, drumless mode, and onset mode are set
  through thread-safe setters.
- The runner tracks dropped frames/underruns.
- State save/load and prefill are first-class concepts.
- The bundled macOS hosts use this runner shape.

This is the right architecture for a live instrument. The problem is platform:
the current repository's CMake build is explicitly macOS-only, because it links
MLX/Metal and Apple host frameworks. The local Linux checkout guidance also
treats C++/examples as not buildable on Linux today.

## New MLX/CUDA Possibility

MLX itself now has a Linux CUDA backend. The current MLX documentation mentions
CUDA packages such as `mlx[cuda12]` / `mlx[cuda13]`, CUDA availability APIs, and
source builds with CUDA enabled.

This makes a Linux/CUDA port of the Magenta C++ core worth testing. It does not
mean the Magenta C++ core will work unchanged. The Magenta repository currently
sets `MLX_BUILD_CUDA OFF` and fails CMake configure on non-Apple platforms.

The main unknown is whether the exported MRT2 `.mlxfn` model path used by
`MLXEngine` works correctly on Linux/CUDA.

Local assets already exist:

- `/home/dev/Documents/Magenta/magenta-rt-v2/models/mrt2_small/mrt2_small.mlxfn`
- `/home/dev/Documents/Magenta/magenta-rt-v2/models/mrt2_small/mrt2_small_state.safetensors`
- `/home/dev/Documents/Magenta/magenta-rt-v2/resources/spectrostream/spectrostream_encoder.mlxfn`
- `/home/dev/Documents/Magenta/magenta-rt-v2/checkpoints/mrt2_small.safetensors`

## AIG/ADG Authority Boundary

AIG/ADG should remain the musical authority. Magenta is a neural renderer or
continuation layer, not the owner of the groove truth.

The desired boundary:

```text
AIG / ADG timeline
  -> explicit frame projection / loss map
  -> Magenta conditioning frames
  -> neural audio frames
  -> audio host output
```

This matters because ADG carries richer semantics than MRT2 conditioning. The
current Magenta conditioning surface is roughly:

- MusicCoCa prompt tokens
- 128 note slots
- one drum conditioning slot
- CFG and sampling parameters
- model state / prefill state

ADG can carry roles, flams, chokes, timing intent, source/atom metadata, and
other higher-level semantics. Any ADG -> MRT2 projection must stay explicit
about what is preserved, collapsed, or lost.

The existing Python `magenta_rt/aig_bridge.py` is already a lossy bridge. A
Linux real-time runtime should not hide that loss.

## Option A: Python MLX/CUDA `.mlxfn` Smoke

Goal: prove or disprove that Linux MLX/CUDA can execute the exported
`mrt2_small.mlxfn` path before porting C++.

Shape:

```text
Linux + MLX CUDA Python
  -> mx.import_function(mrt2_small.mlxfn)
  -> load mrt2_small_state.safetensors
  -> build one conditioning frame
  -> run one frame
  -> benchmark 50 / 125 / 1500 frames
```

Pros:

- fastest way to test the `.mlxfn` + MLX/CUDA hypothesis
- avoids CMake and TFLite/SentencePiece C++ build work at first
- gives a clear yes/no before deeper porting

Risks:

- Python MLX/CUDA may work while C++ MLX/CUDA still needs port work
- `mlxfn` import/export may have backend-specific assumptions
- this is not yet audio-thread-safe

Decision value:

- If this fails, the C++ MLX/CUDA port becomes much less attractive.
- If this passes and is fast, porting `MLXEngine` becomes a strong candidate.

## Option B: Core-Only C++ Linux/CUDA Port

Goal: build the smallest Linux/CUDA C++ executable that uses
`magentart::core::MLXEngine` to render a WAV.

This means porting only:

- `core/include/magentart/mlx_engine.h`
- `core/src/mlx_engine.cpp`
- minimal dependencies for MusicCoCa/TFLite/SentencePiece
- a small CLI similar to `examples/hello_mrt2`

Do not port yet:

- AUv3
- standalone AppKit host
- React/WebKit UI
- CoreMIDI host integration
- notarization/codesign flows

Likely changes:

- split CMake into Apple host targets and portable/core targets
- replace top-level `if(NOT APPLE) FATAL_ERROR` with a target-level guard
- enable MLX CUDA: `MLX_BUILD_CUDA=ON`
- keep MLX CPU as optional or disabled for this target
- avoid Metal-only keepalive logic
- make `AutoreleasePool` a no-op on non-Apple platforms
- ensure TFLite/SentencePiece build and link cleanly on Linux

Pros:

- closest path to the existing Mac real-time implementation
- avoids Python in the inference loop
- can later reuse `RealtimeRunner` concepts directly

Risks:

- CMake/dependency work may be nontrivial
- `MLXEngine` may have hidden Metal or Apple assumptions
- C++ MLX CUDA may expose different behavior than Python MLX CUDA
- build may need substantial dependency patching

Decision value:

- If it can render `mrt2_small.mlxfn` at or above 25 frames/s, it is the best
  candidate for a native Linux real-time engine.

## Option C: Linux `RealtimeRunner` Port

Goal: after Option B succeeds, port the existing runner shape to Linux.

Keep:

- inference thread
- 25 Hz frame generation
- stereo ring buffers
- underrun metrics
- reset and state lifecycle
- prompt/sampling setter model

Replace:

- Apple host integration
- Metal keepalive
- CoreMIDI/AppKit/WebKit assumptions

Add:

- JACK or PipeWire audio callback
- Linux MIDI/control input
- real-time scheduling policy
- memory locking and xruns/underrun reporting

Pros:

- closest to "what works on Mac, but on Linux"
- clean one-process native runtime if MLX CUDA is stable
- likely simpler audio callback semantics than Python/JAX IPC

Risks:

- depends on successful MLX/CUDA C++ core
- real-time Linux scheduling and GPU jitter still need measurement
- direct ADG conditioning API is still missing

## Option D: JAX Daemon + Linux Audio Host

Goal: use the proven JAX/CUDA path and build a Linux real-time shell around it.

Possible process split:

```text
mrt-jaxd
  - owns JAX model and CUDA device
  - keeps transformer state
  - generates 40 ms audio frames
  - writes to shared-memory ring buffer
  - accepts control/ADG conditioning commands

mrt-audio-host
  - JACK/PipeWire callback
  - reads shared-memory ring buffer
  - never calls Python or waits for GPU
  - reports underruns and buffer fill

aig-mrt-bridge
  - projects AIG/ADG truth into 25 Hz Magenta conditioning
  - writes loss maps and manifests
```

Pros:

- uses the fastest proven local backend today
- no need to port MLX C++ before testing live audio behavior
- isolates Python/JAX from the audio callback
- can be built incrementally from the current Docker worker

Risks:

- inter-process ring buffer/control protocol complexity
- Python process lifecycle and crash handling
- JAX GPU jitter may still cause buffer underruns
- more custom infrastructure than reusing `RealtimeRunner`

Decision value:

- If MLX/CUDA C++ stalls, this is the most realistic Linux path.

## Option E: Native C++ CUDA Runtime From Scratch

Goal: implement a new CUDA/TensorRT/XLA/ONNX-style native runtime without MLX.

Pros:

- maximum control
- potentially best long-term performance if fully optimized

Risks:

- largest scope by far
- must reimplement/export transformer execution, sampling, state, codec, and
  conditioning semantics
- easy to spend months before having a better result than JAX

Decision value:

- Keep as a last resort, not a first move.

## Option F: Hybrid Runtime

Possible long-term shape:

- JAX/CUDA worker remains the reference/benchmark backend.
- MLX/CUDA C++ becomes the preferred low-latency live backend if port succeeds.
- MLX CPU remains a diagnostic fallback only.
- Deterministic AIG/audio-atom rendering remains the safe fallback bed.

This avoids betting everything on one runtime too early.

## Real-Time Product Shape

The first realistic Linux live product should probably not promise
sample-accurate neural drum hits.

Better first target:

- AIG/ADG controls groove truth and transport.
- Magenta renders ahead into a buffer.
- Audio host plays from the buffer.
- User changes prompt/CFG/ADG region.
- Changes take effect on a safe frame/window boundary.
- If Magenta misses deadline, deterministic AIG/audio-atom layer or silence
  handles the gap explicitly.

Target latency tiers:

| Tier | Buffer / audible response | Meaning |
|---|---:|---|
| v0 debug live | 1-2 s | stable render-ahead preview |
| v1 playable preview | 320-640 ms | usable as a companion layer |
| v2 tight companion | 120-240 ms | feels responsive but still buffered |
| sub-50 ms | not promised | requires evidence, not assumptions |

## Required Measurements

Average frames/s is not enough. We need streaming measurements:

- per-frame `generate_frame` time
- p50 / p95 / p99 / max frame time
- ring buffer fill over time
- underruns for 3, 5, 8, 16, 32 frame buffers
- CPU/GPU memory
- behavior while changing prompt/CFG/conditioning
- behavior under desktop/GPU load
- recovery after underrun
- startup/load/compile time

## Proposed Investigation Order

1. **MLX/CUDA Python `.mlxfn` smoke**
   Test whether Linux MLX/CUDA can import and execute the existing
   `mrt2_small.mlxfn`.

2. **MLX/CUDA Python frame benchmark**
   Measure 50, 125, and 1500 frames with the same conditioning shape used in
   the existing JAX worker.

3. **Core-only C++ Linux/CUDA configure spike**
   Create a narrow CMake route that builds only `magentart::core` and a small
   Linux CLI. Do not touch AUv3/standalone host targets.

4. **C++ `MLXEngine` render benchmark**
   Render WAV from `mrt2_small.mlxfn`; compare against JAX worker metrics.

5. **Streaming-session benchmark**
   No audio device yet. Simulate ring buffer deadlines and underruns for
   5-10 minutes of continuous generation.

6. **Linux audio host**
   JACK first, PipeWire later if needed. Audio callback only reads a buffer.

7. **ADG conditioning clock**
   Add explicit AIG/ADG -> 25 Hz Magenta conditioning projection with loss maps.

8. **Decision**
   Choose between:
   - native C++ MLX/CUDA runner
   - JAX daemon + audio host
   - hybrid

## Initial Decision Bias

Current bias:

1. Try MLX/CUDA `.mlxfn` first because it can unlock the cleanest port.
2. Keep JAX/CUDA daemon as the proven fallback and benchmark reference.
3. Do not optimize MLX CPU.
4. Do not write a native CUDA transformer from scratch unless MLX/CUDA and JAX
   both fail the live-runtime goals.

## Open Questions

- Can Linux MLX/CUDA import and run the downloaded `mrt2_small.mlxfn`?
- Is `.mlxfn` portable across Metal and CUDA backends for this model?
- Does `MLXEngine` compile cleanly against MLX CUDA C++ once Apple-only targets
  are isolated?
- Does the C++ path equal or beat JAX/CUDA on this RTX 500 Ada GPU?
- Can the C++ path expose direct per-frame conditioning, or do we need a new
  API beyond MIDI note state?
- What latency target is musically acceptable for the first AIG/Magenta live
  preview?
- Should Linux audio host be Rust, C++, or both at different layers?
- Should Docker remain part of the live runtime, or only the lab/render setup?

## Useful Local References

- `README.md`: project overview and hardware claims
- `docs/models.md`: model sizes, assets, and `.mlxfn` layout
- `docs/inference.md`: Python JAX/MLX inference notes
- `core/README.md`: C++ `magentart::core` architecture
- `core/include/magentart/mlx_engine.h`: frame size, generation API, state API
- `core/include/magentart/realtime_runner.h`: audio-thread-safe runner API
- `core/include/magentart/ring_buffer.h`: SPSC audio ring buffer
- `core/src/mlx_engine.cpp`: conditioning layout and generation internals
- `core/src/realtime_runner.cpp`: inference loop and ring-buffer behavior
- `scripts/compare_python_n_cpp.py`: `.mlxfn` conditioning layout reference
- `magenta_rt/render_worker.py`: current Docker JAX/MLX HTTP render worker
- `magenta_rt/aig_bridge.py`: current AIG/ADG -> MRT2 conditioning bridge

## External References

- MLX build/install documentation:
  https://ml-explore.github.io/mlx/build/html/install.html
- MLX repository:
  https://github.com/ml-explore/mlx

