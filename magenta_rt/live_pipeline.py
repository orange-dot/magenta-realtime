# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pipelined JAX streaming generator for the Option D live runtime.

Why this exists
---------------
The Gate 0 host benchmark
(``runs/magenta-live-option-d-gate0/host-jax-4500/metrics.json``) measured a
steady-state RTF of 0.879 (28.45 steps/s vs. the 25 Hz realtime line) and
attributed ~30 ms/step to ``device_get``. That 30 ms is *not* the device->host
copy: the per-step payload is one ``[1920, 2]`` frame (~8-15 KB), which crosses
PCIe in microseconds. JAX dispatches asynchronously, so the model forward pass
surfaces at the first blocking call -- ``device_get`` -- which is therefore a
*GPU-compute wait*, not a transfer cost. (``probe_compute_vs_copy`` below
confirms this in ~30 s: ``block_until_ready`` carries the ~30 ms; the trailing
copy of an already-ready array is sub-millisecond.)

What the serial loop wastes
---------------------------
``render_worker.py`` runs strictly serially:

    build_conditioning (host, ~3.7 ms)  -> GPU IDLE
    device_get          (blocks ~30 ms) -> GPU busy, host idle

so ~3.7 ms of GPU idle is paid *every* step (33.7 ms total -> 28.45 steps/s).

What this module does
---------------------
Two changes reclaim that idle without touching the model or the ring payload:

1. **Precompute conditioning** for the looped schedule once (``ConditioningPlan``)
   so the hot loop never rebuilds numpy/``sl.Sequence`` blocks on the critical
   path. ``constants`` (temperature/top_k) are frame-independent in the current
   model, so they are built once and reused.
2. **Lag the audio pull by one step** so ``device_get`` of frame ``i-1`` overlaps
   the GPU compute of frame ``i``. This is safe because the decoded audio is a
   *side-emit*: only the donated ``state`` (``donate_argnums=(3,)``) feeds the
   autoregressive recurrence, so deferring the audio fetch by one frame costs
   nothing structurally and adds exactly one 40 ms frame of buffer latency
   (irrelevant for a 341-700 ms buffered preview).

Expected effect: the loop approaches the GPU floor (~30-31 ms/step ->
~31-33 steps/s -> RTF ~0.76-0.79), clearing the >=30 steps/s gate with no
precision or model changes. The remaining floor is the GPU forward pass itself;
shrinking it (bf16/fp16, clocks, autotune) is a separate lever.

Scope / boundaries
------------------
This module owns *only* the producer loop. It writes frames through an
``on_frame(frame)`` sink and never imports the shared-memory ring, the control
socket, or the CLI -- those belong to the daemon. The emitted frame is
``int16`` ``[1920, 2]`` C-contiguous, matching the
``magenta_rt.live_ring`` ``s16_interleaved_stereo`` contract; ``make_ring_sink``
adapts it to ``Int16InterleavedStereoRing.write`` (or the cdylib wrapper).

Backpressure
------------
``run(..., backpressure=cb)`` calls ``cb()`` immediately before each ring write;
``cb`` blocks until the ring has room for one model frame. Because the producer
outruns realtime (RTF < 1), this paces it to the consumer's drain rate and
bounds overproduction to the one-frame pipeline depth. For *measuring* the
producer ceiling (Gate 0), pass ``backpressure=None`` so nothing throttles it.

Typical use::

    plan = build_conditioning_plan(mrt, prompt="tight funk drums",
                                   notes=schedule.notes, drums=schedule.drums)
    holder = PlanHolder(plan)            # holder.set(new_plan) swaps at a frame
    gen = PipelinedStreamGenerator(mrt)  # boundary, e.g. for set_prompt
    gen.reset(restart_schedule=True)
    gen.run(holder, make_ring_sink(ring),
            should_stop=stop_event.is_set,
            backpressure=make_poll_backpressure(lambda: ring.free_frames))
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from time import perf_counter
from typing import Any, Protocol

import numpy as np

# This module deliberately drives the same (intentionally "private") streaming
# internals that magenta_rt/render_worker.py uses for offline AIG/ADG renders.
# pylint: disable=protected-access

# Mirrors the magenta_rt.live_ring contract. live_ring.py is the single source
# of truth for the ring payload; keep these in lockstep with it.
MODEL_FRAME_SIZE = 1_920  # audio frames produced per streaming step
NUM_CHANNELS = 2  # interleaved stereo
SAMPLE_RATE = 48_000
FRAME_DURATION_MS = 1_000.0 * MODEL_FRAME_SIZE / SAMPLE_RATE  # 40.0 ms
REALTIME_STEPS_PER_SECOND = SAMPLE_RATE / MODEL_FRAME_SIZE  # 25.0 Hz
DEFAULT_WARMUP_FRAMES = 50

_INT16_MIN = -32_768.0
_INT16_MAX = 32_767.0

OnFrame = Callable[[np.ndarray], None]
Backpressure = Callable[[], None]
ShouldStop = Callable[[], bool]


class StreamingSystem(Protocol):
  """Minimal surface of ``MagentaRT2Jax`` the pipelined generator relies on.

  These are the streaming internals ``render_worker.JaxGpuRenderer`` already
  drives; documented here as the integration contract, not re-implemented.
  """

  _params: Any

  def _jit_init_state(self, params: Any, constants: Mapping[str, Any]) -> Any:
    ...

  def _jit_streaming_step(
      self, params: Any, block: Any, constants: Any, state: Any
  ) -> tuple[Any, Any, Any]:
    ...

  def _build_conditioning(
      self,
      style: Any,
      notes: Any,
      drums: Any,
      cfgs: Any,
      temperature: float,
      top_k: int,
  ) -> tuple[Any, Any]:
    ...

  def embed_style(self, text_or_audio: Any, use_mapper: bool = ...) -> Any:
    ...

  def tokenize_style(self, embedding: Any) -> Any:
    ...


@dataclasses.dataclass
class ConditioningPlan:
  """Precomputed per-frame conditioning for one loop of a schedule.

  ``blocks[i]`` is the conditioning for schedule frame ``i``; the generator
  indexes ``blocks[frame_index % period]`` so a finite schedule loops. A new
  prompt / sampling change is expressed as a *new* plan handed to the
  ``PlanHolder``, which the loop adopts at the next frame boundary.
  """

  blocks: list[Any]
  constants: Any
  period: int
  prompt: str
  summary: dict[str, Any] = dataclasses.field(default_factory=dict)


def build_conditioning_plan(
    mrt: StreamingSystem,
    *,
    prompt: str,
    notes: np.ndarray,
    drums: np.ndarray,
    cfg_musiccoca: float = 3.0,
    cfg_notes: float = 3.0,
    cfg_drums: float = 4.0,
    temperature: float = 1.1,
    top_k: int = 40,
    use_mapper: bool = True,
    summary: Mapping[str, Any] | None = None,
) -> ConditioningPlan:
  """Embeds the prompt once and precomputes one conditioning block per frame.

  ``notes`` and ``drums`` are the per-frame schedule arrays produced by
  ``aig_bridge.build_conditioning_schedule`` (``[frames, 128]`` and
  ``[frames, 1]``). This mirrors ``render_worker.JaxGpuRenderer.render_schedule``
  exactly, but hoists every block out of the hot loop.
  """
  from magenta_rt.jax.system import discretize_cfg

  notes = np.asarray(notes)
  drums = np.asarray(drums)
  if notes.ndim != 2 or drums.ndim != 2:
    raise ValueError("notes and drums must be 2-D [frames, channels] arrays")
  if notes.shape[0] != drums.shape[0]:
    raise ValueError("notes and drums must have the same frame count")
  period = int(notes.shape[0])
  if period <= 0:
    raise ValueError("schedule must contain at least one frame")

  embedding = mrt.embed_style(prompt, use_mapper=use_mapper)
  style_tokens = mrt.tokenize_style(embedding).tolist()
  cfgs = [
      discretize_cfg(cfg_musiccoca, 0.2, 40),
      discretize_cfg(cfg_notes, 0.2, 40),
      discretize_cfg(cfg_drums, 1.0, 8),
  ]

  blocks: list[Any] = []
  constants: Any = None
  for index in range(period):
    notes_row = notes[index].astype(np.int32).tolist()
    drums_row = drums[index].astype(np.int32).tolist()
    block, step_constants = mrt._build_conditioning(
        style_tokens, notes_row, drums_row, cfgs, temperature, top_k
    )
    blocks.append(block)
    if constants is None:
      # constants depend only on temperature/top_k, identical every frame.
      constants = step_constants

  return ConditioningPlan(
      blocks=blocks,
      constants=constants,
      period=period,
      prompt=prompt,
      summary=dict(summary or {}),
  )


class PlanHolder:
  """Thread-safe slot for the active ``ConditioningPlan``.

  The control thread swaps in a new plan (set_prompt / set_sampling); the
  generation loop reads the current plan at each frame boundary, so changes
  take effect on a model-frame boundary and never tear a step.
  """

  def __init__(self, plan: ConditioningPlan):
    self._plan = plan
    self._lock = threading.Lock()

  def get(self) -> ConditioningPlan:
    with self._lock:
      return self._plan

  def set(self, plan: ConditioningPlan) -> None:
    with self._lock:
      self._plan = plan


def _stats(values: Sequence[float]) -> dict[str, float]:
  if not values:
    return {}
  arr = np.asarray(values, dtype=np.float64)
  return {
      "mean_ms": float(arr.mean()),
      "p50_ms": float(np.percentile(arr, 50)),
      "p95_ms": float(np.percentile(arr, 95)),
      "p99_ms": float(np.percentile(arr, 99)),
      "max_ms": float(arr.max()),
      "min_ms": float(arr.min()),
  }


@dataclasses.dataclass
class GenerationMetrics:
  """Per-frame timing, comparable with the Gate 0 metrics schema.

  ``frame_interval`` is the wall-clock cadence per emitted frame (the
  equivalent of Gate 0's ``step_total``); ``dispatch`` is the async
  ``_jit_streaming_step`` return time; ``drain`` is the lagged ``device_get``.
  With ``backpressure=None`` the interval is the producer ceiling; when paced by
  a ring it reflects the (realtime) consumer drain rate instead.
  """

  frame_duration_ms: float = FRAME_DURATION_MS
  warmup_frames: int = 0
  interval_ms: list[float] = dataclasses.field(default_factory=list)
  dispatch_ms: list[float] = dataclasses.field(default_factory=list)
  drain_ms: list[float] = dataclasses.field(default_factory=list)

  def record(self, interval: float, dispatch: float, drain: float) -> None:
    self.interval_ms.append(interval)
    self.dispatch_ms.append(dispatch)
    self.drain_ms.append(drain)

  def summary(self) -> dict[str, Any]:
    interval = _stats(self.interval_ms)
    mean_interval = interval.get("mean_ms", 0.0)
    steps_per_second = 1_000.0 / mean_interval if mean_interval > 0.0 else 0.0
    rtf = mean_interval / self.frame_duration_ms if mean_interval > 0.0 else 0.0
    measured = len(self.interval_ms)
    return {
        "schema": "magenta_rt.option_d.gate0_pipelined.v1",
        "measured_frames": measured,
        "warmup_frames": self.warmup_frames,
        "audio_seconds": measured * self.frame_duration_ms / 1_000.0,
        "steps_per_second": steps_per_second,
        "rtf": rtf,
        "realtime_steps_per_second": REALTIME_STEPS_PER_SECOND,
        "frame_interval": interval,
        "dispatch": _stats(self.dispatch_ms),
        "drain": _stats(self.drain_ms),
    }


class PipelinedStreamGenerator:
  """Drives ``_jit_streaming_step`` with a one-step audio lag.

  Holds the donated streaming ``state`` and exactly one in-flight step output
  (``_pending``). ``state`` is reassigned each step (the old buffer is donated);
  ``_pending`` is a side-emit and is safe to carry across the next dispatch.
  """

  def __init__(
      self,
      mrt: StreamingSystem,
      *,
      frame_samples: int = MODEL_FRAME_SIZE,
      num_channels: int = NUM_CHANNELS,
      frame_duration_ms: float = FRAME_DURATION_MS,
      output_dtype: Any = np.int16,
      logger: logging.Logger | None = None,
  ):
    self._mrt = mrt
    self._frame_samples = int(frame_samples)
    self._channels = int(num_channels)
    self._frame_duration_ms = float(frame_duration_ms)
    self._output_dtype = np.dtype(output_dtype)
    if self._output_dtype not in (np.dtype(np.int16), np.dtype(np.float32)):
      raise ValueError("output_dtype must be int16 or float32")
    self._log = logger or logging.getLogger(__name__)
    self._state: Any = None
    self._pending: Any = None
    self._frame_index = 0

  @property
  def frame_index(self) -> int:
    return self._frame_index

  def reset(self, *, restart_schedule: bool = False) -> None:
    """Re-initializes streaming state and drops any pending frame.

    Maps to the daemon ``reset`` op. The donated state cannot be snapshotted, so
    reset simply re-runs ``_jit_init_state``. ``restart_schedule`` also rewinds
    the conditioning index to 0.
    """
    self._state = self._mrt._jit_init_state(self._mrt._params, {})
    self._pending = None
    if restart_schedule:
      self._frame_index = 0

  def run(
      self,
      holder: PlanHolder,
      on_frame: OnFrame,
      *,
      max_frames: int | None = None,
      warmup_frames: int = DEFAULT_WARMUP_FRAMES,
      should_stop: ShouldStop | None = None,
      backpressure: Backpressure | None = None,
  ) -> dict[str, Any]:
    """Runs warmup then the measured/live loop; returns a metrics summary.

    Provide ``max_frames`` (bounded run / benchmark) and/or ``should_stop``
    (live run). Warmup runs the same pipeline with audio discarded and is never
    paced or measured, so the GPU reaches steady-state clocks before the buffer
    is primed.
    """
    import jax  # local: keeps the module importable without a JAX device

    if self._state is None:
      self.reset()
    metrics = GenerationMetrics(
        frame_duration_ms=self._frame_duration_ms,
        warmup_frames=max(0, warmup_frames),
    )
    if warmup_frames and warmup_frames > 0:
      self._drive(holder, None, warmup_frames, should_stop, None, jax, None)
    self._drive(
        holder, on_frame, max_frames, should_stop, backpressure, jax, metrics
    )
    return metrics.summary()

  def flush(
      self, on_frame: OnFrame, *, backpressure: Backpressure | None = None
  ) -> None:
    """Emits the final in-flight frame (call once on stop for a clean tail)."""
    import jax

    if self._pending is None:
      return
    host = jax.device_get(self._pending.values[0])
    self._pending = None
    frame = self._finish_frame(host)
    if backpressure is not None:
      backpressure()
    if on_frame is not None:
      on_frame(frame)

  def _dispatch(self, block: Any, constants: Any) -> Any:
    # Async dispatch: returns a handle quickly; donates and replaces state.
    step_output, self._state, _ = self._mrt._jit_streaming_step(
        self._mrt._params, block, constants, self._state
    )
    return step_output

  def _drive(
      self,
      holder: PlanHolder,
      sink: OnFrame | None,
      num_frames: int | None,
      should_stop: ShouldStop | None,
      backpressure: Backpressure | None,
      jax: Any,
      metrics: GenerationMetrics | None,
  ) -> None:
    if num_frames is None and should_stop is None:
      raise ValueError("provide max_frames or should_stop to bound the loop")

    produced = 0
    while True:
      if should_stop is not None and should_stop():
        break
      if num_frames is not None and produced >= num_frames:
        break

      plan = holder.get()
      if plan.period <= 0:
        raise ValueError("conditioning plan has no frames")
      block = plan.blocks[self._frame_index % plan.period]

      iter_t0 = perf_counter()
      t0 = perf_counter()
      out = self._dispatch(block, plan.constants)  # dispatch frame i (async)
      dispatch_ms = (perf_counter() - t0) * 1_000.0

      drain_ms = 0.0
      emitted = False
      if self._pending is not None:
        # Drain frame i-1 while the GPU computes frame i (overlap).
        t0 = perf_counter()
        host = jax.device_get(self._pending.values[0])
        drain_ms = (perf_counter() - t0) * 1_000.0
        frame = self._finish_frame(host)
        if backpressure is not None:
          backpressure()  # block until the ring has room for one frame
        if sink is not None:
          sink(frame)
        emitted = True

      self._pending = out
      self._frame_index += 1
      iter_ms = (perf_counter() - iter_t0) * 1_000.0

      if emitted:
        produced += 1
        if metrics is not None:
          metrics.record(iter_ms, dispatch_ms, drain_ms)

  def _finish_frame(self, host: Any) -> np.ndarray:
    arr = np.asarray(host).reshape(self._frame_samples, self._channels)
    if self._output_dtype == np.dtype(np.int16):
      # Match render_worker's int16 quantization; clip guards against wrap.
      arr = np.clip(arr, _INT16_MIN, _INT16_MAX)
      return np.ascontiguousarray(arr.astype(np.int16))
    arr = np.clip(np.asarray(arr, dtype=np.float32) / 32_768.0, -1.0, 1.0)
    return np.ascontiguousarray(arr.astype(np.float32))


def make_ring_sink(
    ring: Any, on_write: Callable[[dict], None] | None = None
) -> OnFrame:
  """Adapts a ``live_ring.Int16InterleavedStereoRing`` (or cdylib wrapper).

  The generator emits ``int16 [MODEL_FRAME_SIZE, 2]`` -- exactly what
  ``Int16InterleavedStereoRing.write`` expects. ``on_write`` (optional) receives
  the per-write stats dict (written/dropped/free frames).
  """

  def _sink(frame: np.ndarray) -> None:
    stats = ring.write(frame)
    if on_write is not None and stats is not None:
      on_write(stats)

  return _sink


def make_poll_backpressure(
    free_frames: Callable[[], int],
    *,
    need_frames: int = MODEL_FRAME_SIZE,
    sleep_s: float = 0.001,
    should_stop: ShouldStop | None = None,
) -> Backpressure:
  """Blocks until the ring reports room for one model frame.

  Wire ``free_frames`` to ``ring.free_frames`` (in-process) or the cdylib's
  free-frame query. Returns promptly if ``should_stop`` fires so a stop request
  is never swallowed by a full ring.
  """

  def _wait() -> None:
    while free_frames() < need_frames:
      if should_stop is not None and should_stop():
        return
      time.sleep(sleep_s)

  return _wait


def probe_compute_vs_copy(
    mrt: StreamingSystem,
    plan: ConditioningPlan,
    *,
    frames: int = 200,
    warmup: int = DEFAULT_WARMUP_FRAMES,
) -> dict[str, Any]:
  """Confirms the Gate 0 diagnosis: device_get is compute-wait, not transfer.

  Times ``block_until_ready`` (pure GPU-compute completion) separately from a
  trailing ``device_get`` of the now-ready array (pure host copy). Expect
  ``compute_blocking`` ~= the ~30 ms step and ``host_copy`` sub-millisecond --
  proof that shrinking the ring payload buys nothing.
  """
  import jax

  state = mrt._jit_init_state(mrt._params, {})
  index = 0
  for _ in range(max(0, warmup)):
    out, state, _ = mrt._jit_streaming_step(
        mrt._params, plan.blocks[index % plan.period], plan.constants, state
    )
    jax.block_until_ready(out.values)
    index += 1

  compute_ms: list[float] = []
  copy_ms: list[float] = []
  for _ in range(frames):
    out, state, _ = mrt._jit_streaming_step(
        mrt._params, plan.blocks[index % plan.period], plan.constants, state
    )
    t0 = perf_counter()
    jax.block_until_ready(out.values)  # wait for the forward pass
    compute_ms.append((perf_counter() - t0) * 1_000.0)
    t0 = perf_counter()
    jax.device_get(out.values[0])  # copy an already-ready array
    copy_ms.append((perf_counter() - t0) * 1_000.0)
    index += 1

  return {
      "schema": "magenta_rt.option_d.compute_vs_copy.v1",
      "frames": frames,
      "warmup_frames": max(0, warmup),
      "compute_blocking": _stats(compute_ms),
      "host_copy": _stats(copy_ms),
  }


def measure_pipelined(
    mrt: StreamingSystem,
    *,
    prompt: str,
    notes: np.ndarray,
    drums: np.ndarray,
    measured_frames: int,
    warmup_frames: int = DEFAULT_WARMUP_FRAMES,
    cfg_musiccoca: float = 3.0,
    cfg_notes: float = 3.0,
    cfg_drums: float = 4.0,
    temperature: float = 1.1,
    top_k: int = 40,
    on_frame: OnFrame | None = None,
    output_dtype: Any = np.int16,
    summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
  """Re-runs Gate 0 with the pipelined loop (unpaced -> producer ceiling).

  No backpressure, so the returned ``steps_per_second`` / ``rtf`` measure the
  raw producer throughput, directly comparable to the serial Gate 0 numbers.
  Pass an ``on_frame`` to also capture/verify audio; the default discards it.
  """
  plan = build_conditioning_plan(
      mrt,
      prompt=prompt,
      notes=notes,
      drums=drums,
      cfg_musiccoca=cfg_musiccoca,
      cfg_notes=cfg_notes,
      cfg_drums=cfg_drums,
      temperature=temperature,
      top_k=top_k,
      summary=summary,
  )
  holder = PlanHolder(plan)
  generator = PipelinedStreamGenerator(mrt, output_dtype=output_dtype)
  generator.reset(restart_schedule=True)
  sink: OnFrame = on_frame if on_frame is not None else (lambda frame: None)
  result = generator.run(
      holder,
      sink,
      max_frames=measured_frames,
      warmup_frames=warmup_frames,
      backpressure=None,
  )
  generator.flush(sink)
  result["prompt"] = prompt
  result["period_frames"] = plan.period
  return result
