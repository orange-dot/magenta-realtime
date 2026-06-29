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

"""Chunked JAX producer for live Magenta RT preview."""

from __future__ import annotations

from collections.abc import Callable, Iterable
import dataclasses
import functools
import time
from typing import Any, Literal

import numpy as np

from magenta_rt import aig_bridge
from magenta_rt.live_ring import DEFAULT_CHUNK_FRAMES
from magenta_rt.live_ring import DEFAULT_CHANNELS
from magenta_rt.live_ring import DEFAULT_MODEL_FRAME_SIZE
from magenta_rt.live_ring import DEFAULT_SAMPLE_RATE


ProducerMode = Literal[
    "serial", "sync-diagnosis", "pipeline-1", "chunked", "scan-chunked"
]
RingWriter = Callable[[np.ndarray], Any]


@dataclasses.dataclass(frozen=True)
class PromptTokenUpdate:
  """A prepared prompt/style token update requested for a model frame."""

  requested_frame: int
  style_tokens: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class PreparedFrame:
  """One model-frame worth of already-built conditioning."""

  frame_index: int
  block: Any
  constants: dict[str, Any]
  style_generation: int


@dataclasses.dataclass(frozen=True)
class PreparedSchedule:
  """Prepared frame loop for the JAX hot path."""

  frames: tuple[PreparedFrame, ...]
  token_values: Any | None = None
  constants: dict[str, Any] | None = None
  chunk_frames: int = DEFAULT_CHUNK_FRAMES
  frame_audio_frames: int = DEFAULT_MODEL_FRAME_SIZE
  sample_rate: int = DEFAULT_SAMPLE_RATE
  channels: int = DEFAULT_CHANNELS
  prompt_embedded_in_hot_loop: bool = False


@dataclasses.dataclass(frozen=True)
class ProducerMetrics:
  """Timing and throughput metrics for a producer run."""

  mode: ProducerMode
  frames: int
  chunks: int
  chunk_frames: int
  audio_frames: int
  dispatch_seconds: float
  sync_seconds: float
  copy_seconds: float
  ring_write_seconds: float
  total_seconds: float
  steps_per_second: float
  rtf: float
  low_water_frames: int | None
  prompt_embedded_in_hot_loop: bool

  def to_dict(self) -> dict[str, Any]:
    return dataclasses.asdict(self)


class JaxLiveProducer:
  """Reusable MRT2 JAX producer that syncs/materializes per chunk."""

  def __init__(
      self,
      mrt: Any,
      *,
      jax_module: Any | None = None,
      sequence_layers_module: Any | None = None,
  ):
    if jax_module is None:
      import jax as jax_module  # pylint: disable=reimported
    if sequence_layers_module is None:
      import sequence_layers.jax as sequence_layers_module

    self._mrt = mrt
    self._jax = jax_module
    self._sl = sequence_layers_module
    self._scan_chunk_fn = None
    self._compiled_scan_chunks: dict[int, Any] = {}

  def init_state(self) -> Any:
    return self._mrt._jit_init_state(  # pylint: disable=protected-access
        self._mrt._params, {}  # pylint: disable=protected-access
    )

  def prepare_style_tokens(
      self, prompt: str, *, use_mapper: bool = True
  ) -> list[int]:
    """Embeds and tokenizes a prompt outside the streaming hot loop."""

    embedding = self._mrt.embed_style(prompt, use_mapper=use_mapper)
    return self._mrt.tokenize_style(embedding).tolist()

  def prepare_schedule(
      self,
      schedule: aig_bridge.ConditioningSchedule,
      *,
      style_tokens: Iterable[int],
      cfgs: list[int],
      temperature: float,
      top_k: int,
      chunk_frames: int = DEFAULT_CHUNK_FRAMES,
      prompt_updates: Iterable[PromptTokenUpdate] = (),
  ) -> PreparedSchedule:
    """Precomputes JAX conditioning blocks/constants for every model frame."""

    style_tokens_tuple = tuple(int(token) for token in style_tokens)
    updates = normalize_prompt_updates(prompt_updates, chunk_frames)
    prepared = []
    token_rows = []
    frames = int(schedule.summary["frames"])
    shared_constants = None
    for index in range(frames):
      active_style, style_generation = style_tokens_for_frame(
          style_tokens_tuple, updates, index
      )
      notes = schedule.notes[index].astype(np.int32).tolist()
      drums = schedule.drums[index].astype(np.int32).tolist()
      block, constants = self._mrt._build_conditioning(  # pylint: disable=protected-access
          list(active_style),
          notes,
          drums,
          cfgs,
          temperature,
          top_k,
      )
      token_rows.append(np.asarray(block.values).reshape(-1).astype(np.int32))
      if shared_constants is None:
        shared_constants = constants
      prepared.append(
          PreparedFrame(
              frame_index=index,
              block=block,
              constants=constants,
              style_generation=style_generation,
          )
      )
    token_values = self._jax.device_put(np.stack(token_rows, axis=0))
    return PreparedSchedule(
        frames=tuple(prepared),
        token_values=token_values,
        constants=shared_constants,
        chunk_frames=chunk_frames,
    )

  def generate_int16(
      self,
      prepared: PreparedSchedule,
      *,
      state: Any | None = None,
      mode: ProducerMode = "chunked",
      ring_writer: RingWriter | None = None,
  ) -> tuple[np.ndarray, Any, ProducerMetrics]:
    """Runs the prepared schedule and returns interleaved int16 stereo audio."""

    if state is None:
      state = self.init_state()
    if mode == "serial":
      return self._generate_serial(prepared, state, ring_writer)
    if mode == "sync-diagnosis":
      return self._generate_sync_diagnosis(prepared, state, ring_writer)
    if mode == "pipeline-1":
      return self._generate_pipeline_1(prepared, state, ring_writer)
    if mode == "chunked":
      return self._generate_chunked(prepared, state, ring_writer)
    if mode == "scan-chunked":
      return self._generate_scan_chunked(prepared, state, ring_writer)
    raise ValueError(f"unsupported producer mode: {mode}")

  def precompile_scan_chunks(
      self, prepared: PreparedSchedule, *, state: Any | None = None
  ) -> None:
    """AOT-compiles the scan chunk function for all chunk lengths in a run."""

    if prepared.token_values is None or prepared.constants is None:
      raise ValueError("scan-chunked requires prepared token values")
    if state is None:
      state = self.init_state()
    for start, end in chunk_frame_ranges(
        len(prepared.frames), prepared.chunk_frames
    ):
      chunk_len = end - start
      self._compile_scan_chunk(
          chunk_len,
          prepared.token_values[start:end],
          prepared.constants,
          state,
      )

  def _step(self, frame: PreparedFrame, state: Any) -> tuple[Any, Any]:
    step_output, state, _ = self._mrt._jit_streaming_step(  # pylint: disable=protected-access
        self._mrt._params,  # pylint: disable=protected-access
        frame.block,
        frame.constants,
        state,
    )
    return step_output, state

  def _generate_serial(
      self,
      prepared: PreparedSchedule,
      state: Any,
      ring_writer: RingWriter | None,
  ) -> tuple[np.ndarray, Any, ProducerMetrics]:
    outputs = []
    dispatch_seconds = 0.0
    copy_seconds = 0.0
    ring_write_seconds = 0.0
    low_water_frames: int | None = None
    started = time.perf_counter()
    for frame in prepared.frames:
      t0 = time.perf_counter()
      step_output, state = self._step(frame, state)
      dispatch_seconds += time.perf_counter() - t0
      chunk, copy_elapsed = self._device_get_int16(step_output.values[0])
      copy_seconds += copy_elapsed
      outputs.append(chunk)
      ring_write_seconds, low_water_frames = self._write_ring(
          ring_writer, chunk, ring_write_seconds, low_water_frames
      )
    return self._finish(
        "serial",
        prepared,
        outputs,
        state,
        dispatch_seconds,
        0.0,
        copy_seconds,
        ring_write_seconds,
        low_water_frames,
        started,
    )

  def _generate_sync_diagnosis(
      self,
      prepared: PreparedSchedule,
      state: Any,
      ring_writer: RingWriter | None,
  ) -> tuple[np.ndarray, Any, ProducerMetrics]:
    outputs = []
    dispatch_seconds = 0.0
    sync_seconds = 0.0
    copy_seconds = 0.0
    ring_write_seconds = 0.0
    low_water_frames: int | None = None
    started = time.perf_counter()
    for frame in prepared.frames:
      t0 = time.perf_counter()
      step_output, state = self._step(frame, state)
      dispatch_seconds += time.perf_counter() - t0
      t1 = time.perf_counter()
      self._jax.block_until_ready(step_output.values[0])
      sync_seconds += time.perf_counter() - t1
      chunk, copy_elapsed = self._device_get_int16(step_output.values[0])
      copy_seconds += copy_elapsed
      outputs.append(chunk)
      ring_write_seconds, low_water_frames = self._write_ring(
          ring_writer, chunk, ring_write_seconds, low_water_frames
      )
    return self._finish(
        "sync-diagnosis",
        prepared,
        outputs,
        state,
        dispatch_seconds,
        sync_seconds,
        copy_seconds,
        ring_write_seconds,
        low_water_frames,
        started,
    )

  def _generate_pipeline_1(
      self,
      prepared: PreparedSchedule,
      state: Any,
      ring_writer: RingWriter | None,
  ) -> tuple[np.ndarray, Any, ProducerMetrics]:
    outputs = []
    dispatch_seconds = 0.0
    copy_seconds = 0.0
    ring_write_seconds = 0.0
    low_water_frames: int | None = None
    pending_output = None
    started = time.perf_counter()
    for frame in prepared.frames:
      t0 = time.perf_counter()
      step_output, state = self._step(frame, state)
      dispatch_seconds += time.perf_counter() - t0
      if pending_output is not None:
        chunk, copy_elapsed = self._device_get_int16(pending_output.values[0])
        copy_seconds += copy_elapsed
        outputs.append(chunk)
        ring_write_seconds, low_water_frames = self._write_ring(
            ring_writer, chunk, ring_write_seconds, low_water_frames
        )
      pending_output = step_output
    if pending_output is not None:
      chunk, copy_elapsed = self._device_get_int16(pending_output.values[0])
      copy_seconds += copy_elapsed
      outputs.append(chunk)
      ring_write_seconds, low_water_frames = self._write_ring(
          ring_writer, chunk, ring_write_seconds, low_water_frames
      )
    return self._finish(
        "pipeline-1",
        prepared,
        outputs,
        state,
        dispatch_seconds,
        0.0,
        copy_seconds,
        ring_write_seconds,
        low_water_frames,
        started,
    )

  def _generate_chunked(
      self,
      prepared: PreparedSchedule,
      state: Any,
      ring_writer: RingWriter | None,
  ) -> tuple[np.ndarray, Any, ProducerMetrics]:
    outputs = []
    dispatch_seconds = 0.0
    copy_seconds = 0.0
    ring_write_seconds = 0.0
    low_water_frames: int | None = None
    started = time.perf_counter()
    for start, end in chunk_frame_ranges(
        len(prepared.frames), prepared.chunk_frames
    ):
      step_outputs = []
      t0 = time.perf_counter()
      for frame in prepared.frames[start:end]:
        step_output, state = self._step(frame, state)
        step_outputs.append(step_output)
      chunk_sequence = self._sl.Sequence.concatenate_sequences(step_outputs)
      dispatch_seconds += time.perf_counter() - t0
      chunk, copy_elapsed = self._device_get_int16(chunk_sequence.values[0])
      copy_seconds += copy_elapsed
      outputs.append(chunk)
      ring_write_seconds, low_water_frames = self._write_ring(
          ring_writer, chunk, ring_write_seconds, low_water_frames
      )
    return self._finish(
        "chunked",
        prepared,
        outputs,
        state,
        dispatch_seconds,
        0.0,
        copy_seconds,
        ring_write_seconds,
        low_water_frames,
        started,
    )

  def _generate_scan_chunked(
      self,
      prepared: PreparedSchedule,
      state: Any,
      ring_writer: RingWriter | None,
  ) -> tuple[np.ndarray, Any, ProducerMetrics]:
    if prepared.token_values is None or prepared.constants is None:
      raise ValueError("scan-chunked requires prepared token values")
    outputs = []
    dispatch_seconds = 0.0
    copy_seconds = 0.0
    ring_write_seconds = 0.0
    low_water_frames: int | None = None
    started = time.perf_counter()
    for start, end in chunk_frame_ranges(
        len(prepared.frames), prepared.chunk_frames
    ):
      token_chunk = prepared.token_values[start:end]
      t0 = time.perf_counter()
      chunk_value, state = self._scan_chunk(
          token_chunk,
          prepared.constants,
          state,
      )
      dispatch_seconds += time.perf_counter() - t0
      chunk, copy_elapsed = self._device_get_int16(chunk_value)
      copy_seconds += copy_elapsed
      outputs.append(chunk)
      ring_write_seconds, low_water_frames = self._write_ring(
          ring_writer, chunk, ring_write_seconds, low_water_frames
      )
    return self._finish(
        "scan-chunked",
        prepared,
        outputs,
        state,
        dispatch_seconds,
        0.0,
        copy_seconds,
        ring_write_seconds,
        low_water_frames,
        started,
    )

  def _scan_chunk(
      self,
      token_chunk: Any,
      constants: dict[str, Any],
      state: Any,
  ) -> tuple[Any, Any]:
    chunk_len = int(token_chunk.shape[0])
    self._compile_scan_chunk(chunk_len, token_chunk, constants, state)
    compiled = self._compiled_scan_chunks[chunk_len]
    return compiled(
        self._mrt._params,  # pylint: disable=protected-access
        token_chunk,
        constants,
        state,
    )

  def _compile_scan_chunk(
      self,
      chunk_len: int,
      token_chunk: Any,
      constants: dict[str, Any],
      state: Any,
  ) -> None:
    if chunk_len in self._compiled_scan_chunks:
      return
    scan_chunk_fn = self._get_scan_chunk_fn()
    self._compiled_scan_chunks[chunk_len] = scan_chunk_fn.lower(
        self._mrt._params,  # pylint: disable=protected-access
        token_chunk,
        constants,
        state,
    ).compile()

  def _get_scan_chunk_fn(self):
    if self._scan_chunk_fn is not None:
      return self._scan_chunk_fn

    rngs = {
        "params": self._jax.random.PRNGKey(42),
        "random": self._jax.random.PRNGKey(0),
    }
    sampler = self._mrt._sampler  # pylint: disable=protected-access
    sl = self._sl
    jax = self._jax

    @functools.partial(jax.jit, donate_argnums=(3,))
    def _scan_chunk(params, token_chunk, constants, state):
      def _body(carry, token_row):
        block = sl.Sequence.from_values(token_row.reshape(1, 1, -1))
        step_output, next_state, _ = sampler.apply(
            params,
            x=block,
            state=carry,
            constants=constants,
            training=False,
            rngs=rngs,
            method=sampler.step_with_emits,
        )
        return next_state, step_output.values[0]

      state, audio = jax.lax.scan(_body, state, token_chunk)
      return audio.reshape((-1, DEFAULT_CHANNELS)), state

    self._scan_chunk_fn = _scan_chunk
    return self._scan_chunk_fn

  def _device_get_int16(self, value: Any) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    chunk = self._jax.device_get(value).astype(np.int16, copy=False)
    elapsed = time.perf_counter() - started
    _validate_chunk(chunk)
    return chunk, elapsed

  def _write_ring(
      self,
      ring_writer: RingWriter | None,
      chunk: np.ndarray,
      ring_write_seconds: float,
      low_water_frames: int | None,
  ) -> tuple[float, int | None]:
    if ring_writer is None:
      return ring_write_seconds, low_water_frames
    started = time.perf_counter()
    result = ring_writer(chunk)
    ring_write_seconds += time.perf_counter() - started
    available = _available_frames_from_writer_result(result, ring_writer)
    if available is not None:
      low_water_frames = (
          available
          if low_water_frames is None
          else min(low_water_frames, available)
      )
    return ring_write_seconds, low_water_frames

  def _finish(
      self,
      mode: ProducerMode,
      prepared: PreparedSchedule,
      outputs: list[np.ndarray],
      state: Any,
      dispatch_seconds: float,
      sync_seconds: float,
      copy_seconds: float,
      ring_write_seconds: float,
      low_water_frames: int | None,
      started: float,
  ) -> tuple[np.ndarray, Any, ProducerMetrics]:
    total_seconds = time.perf_counter() - started
    samples = (
        np.concatenate(outputs, axis=0)
        if outputs
        else np.zeros((0, DEFAULT_CHANNELS), dtype=np.int16)
    )
    frames = len(prepared.frames)
    audio_frames = int(samples.shape[0])
    target_seconds = audio_frames / prepared.sample_rate
    steps_per_second = frames / total_seconds if total_seconds > 0 else 0.0
    rtf = total_seconds / target_seconds if target_seconds > 0 else 0.0
    metrics = ProducerMetrics(
        mode=mode,
        frames=frames,
        chunks=len(outputs),
        chunk_frames=prepared.chunk_frames,
        audio_frames=audio_frames,
        dispatch_seconds=dispatch_seconds,
        sync_seconds=sync_seconds,
        copy_seconds=copy_seconds,
        ring_write_seconds=ring_write_seconds,
        total_seconds=total_seconds,
        steps_per_second=steps_per_second,
        rtf=rtf,
        low_water_frames=low_water_frames,
        prompt_embedded_in_hot_loop=prepared.prompt_embedded_in_hot_loop,
    )
    return samples, state, metrics


def chunk_frame_ranges(total_frames: int, chunk_frames: int) -> list[tuple[int, int]]:
  if total_frames < 0:
    raise ValueError("total_frames must be non-negative")
  if chunk_frames <= 0:
    raise ValueError("chunk_frames must be positive")
  return [
      (start, min(total_frames, start + chunk_frames))
      for start in range(0, total_frames, chunk_frames)
  ]


def normalize_prompt_updates(
    updates: Iterable[PromptTokenUpdate], chunk_frames: int
) -> tuple[tuple[int, tuple[int, ...]], ...]:
  if chunk_frames <= 0:
    raise ValueError("chunk_frames must be positive")
  normalized = []
  for update in updates:
    if update.requested_frame < 0:
      raise ValueError("prompt update frame must be non-negative")
    boundary = align_prompt_update_frame(update.requested_frame, chunk_frames)
    normalized.append((boundary, update.style_tokens))
  return tuple(sorted(normalized, key=lambda item: item[0]))


def align_prompt_update_frame(requested_frame: int, chunk_frames: int) -> int:
  if requested_frame < 0:
    raise ValueError("requested_frame must be non-negative")
  if chunk_frames <= 0:
    raise ValueError("chunk_frames must be positive")
  return ((requested_frame + chunk_frames - 1) // chunk_frames) * chunk_frames


def style_tokens_for_frame(
    base_tokens: tuple[int, ...],
    updates: tuple[tuple[int, tuple[int, ...]], ...],
    frame_index: int,
) -> tuple[tuple[int, ...], int]:
  active = base_tokens
  generation = 0
  for boundary, tokens in updates:
    if frame_index < boundary:
      break
    active = tokens
    generation += 1
  return active, generation


def _validate_chunk(chunk: np.ndarray) -> None:
  if chunk.dtype != np.int16:
    raise TypeError(f"producer output must be int16, got {chunk.dtype}")
  if chunk.ndim != 2 or chunk.shape[1] != DEFAULT_CHANNELS:
    raise ValueError(
        "producer output must have shape [audio_frames, 2] for stereo"
    )


def _available_frames_from_writer_result(
    result: Any, ring_writer: RingWriter
) -> int | None:
  if isinstance(result, dict) and "available_frames" in result:
    return int(result["available_frames"])
  owner = getattr(ring_writer, "__self__", None)
  available = getattr(owner, "available_frames", None)
  if available is None:
    return None
  return int(available)
