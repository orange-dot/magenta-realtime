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

"""Live-preview int16 interleaved stereo ring contract."""

from __future__ import annotations

import dataclasses
import mmap
from pathlib import Path
import struct
import time
from typing import Callable

import numpy as np


RING_FORMAT_S16_INTERLEAVED_STEREO = "s16_interleaved_stereo"
RING_FORMAT_ID_S16_INTERLEAVED_STEREO = 1
RING_MAGIC = b"MRT2RNG1"
RING_VERSION = 1
RING_HEADER_SIZE = 128
_RING_HEADER_STRUCT = struct.Struct("<8s7I6Q44x")
DEFAULT_SAMPLE_RATE = 48_000
DEFAULT_CHANNELS = 2
DEFAULT_MODEL_FRAME_SIZE = 1_920
DEFAULT_CHUNK_FRAMES = 8
ShouldStop = Callable[[], bool]


@dataclasses.dataclass
class RingHeader:
  """Shared-memory ring v1 metadata.

  Cursor units are audio frames, not scalar samples. With stereo int16 data,
  one cursor unit is one interleaved ``[left, right]`` row.
  """

  format: str = RING_FORMAT_S16_INTERLEAVED_STEREO
  sample_rate: int = DEFAULT_SAMPLE_RATE
  channels: int = DEFAULT_CHANNELS
  model_frame_size: int = DEFAULT_MODEL_FRAME_SIZE
  chunk_frames: int = DEFAULT_CHUNK_FRAMES
  capacity_frames: int = DEFAULT_MODEL_FRAME_SIZE * DEFAULT_CHUNK_FRAMES * 4
  write_cursor: int = 0
  read_cursor: int = 0
  underrun_frames: int = 0
  overrun_frames: int = 0
  low_water_frames: int | None = None

  def validate_consumer(
      self,
      *,
      sample_rate: int = DEFAULT_SAMPLE_RATE,
      channels: int = DEFAULT_CHANNELS,
      format: str = RING_FORMAT_S16_INTERLEAVED_STEREO,
  ) -> None:
    if self.format != format:
      raise ValueError(f"ring format mismatch: {self.format} != {format}")
    if self.sample_rate != sample_rate:
      raise ValueError(
          f"ring sample-rate mismatch: {self.sample_rate} != {sample_rate}"
      )
    if self.channels != channels:
      raise ValueError(f"ring channel mismatch: {self.channels} != {channels}")
    if self.model_frame_size != DEFAULT_MODEL_FRAME_SIZE:
      raise ValueError(
          "ring model-frame mismatch: "
          f"{self.model_frame_size} != {DEFAULT_MODEL_FRAME_SIZE}"
      )

  @property
  def normalized_low_water_frames(self) -> int:
    if self.low_water_frames is None:
      return self.capacity_frames
    return int(self.low_water_frames)

  def to_bytes(self) -> bytes:
    """Serializes the cross-process ring header used by the Rust host."""

    return _RING_HEADER_STRUCT.pack(
        RING_MAGIC,
        RING_VERSION,
        RING_HEADER_SIZE,
        RING_FORMAT_ID_S16_INTERLEAVED_STEREO,
        int(self.sample_rate),
        int(self.channels),
        int(self.model_frame_size),
        int(self.chunk_frames),
        int(self.capacity_frames),
        int(self.write_cursor),
        int(self.read_cursor),
        int(self.underrun_frames),
        int(self.overrun_frames),
        self.normalized_low_water_frames,
    )

  @classmethod
  def from_bytes(cls, data: bytes | bytearray | memoryview) -> "RingHeader":
    """Parses a cross-process ring header."""

    if len(data) < RING_HEADER_SIZE:
      raise ValueError("ring header data is too short")
    (
        magic,
        version,
        header_size,
        format_id,
        sample_rate,
        channels,
        model_frame_size,
        chunk_frames,
        capacity_frames,
        write_cursor,
        read_cursor,
        underrun_frames,
        overrun_frames,
        low_water_frames,
    ) = _RING_HEADER_STRUCT.unpack_from(data)
    if magic != RING_MAGIC:
      raise ValueError(f"ring magic mismatch: {magic!r}")
    if version != RING_VERSION:
      raise ValueError(f"ring version mismatch: {version} != {RING_VERSION}")
    if header_size != RING_HEADER_SIZE:
      raise ValueError(
          f"ring header-size mismatch: {header_size} != {RING_HEADER_SIZE}"
      )
    if format_id != RING_FORMAT_ID_S16_INTERLEAVED_STEREO:
      raise ValueError(f"ring format id mismatch: {format_id}")
    return cls(
        sample_rate=int(sample_rate),
        channels=int(channels),
        model_frame_size=int(model_frame_size),
        chunk_frames=int(chunk_frames),
        capacity_frames=int(capacity_frames),
        write_cursor=int(write_cursor),
        read_cursor=int(read_cursor),
        underrun_frames=int(underrun_frames),
        overrun_frames=int(overrun_frames),
        low_water_frames=int(low_water_frames),
    )


class Int16InterleavedStereoRing:
  """Single-producer/single-consumer in-process model of the ring v1 payload."""

  def __init__(
      self,
      *,
      capacity_frames: int,
      chunk_frames: int = DEFAULT_CHUNK_FRAMES,
      sample_rate: int = DEFAULT_SAMPLE_RATE,
      model_frame_size: int = DEFAULT_MODEL_FRAME_SIZE,
  ):
    if capacity_frames <= 0:
      raise ValueError("capacity_frames must be positive")
    if chunk_frames <= 0:
      raise ValueError("chunk_frames must be positive")
    self.header = RingHeader(
        sample_rate=sample_rate,
        model_frame_size=model_frame_size,
        chunk_frames=chunk_frames,
        capacity_frames=capacity_frames,
    )
    self._data = np.zeros(
        (capacity_frames, DEFAULT_CHANNELS), dtype=np.int16
    )

  @property
  def capacity_frames(self) -> int:
    return self.header.capacity_frames

  @property
  def available_frames(self) -> int:
    return self.header.write_cursor - self.header.read_cursor

  @property
  def free_frames(self) -> int:
    return self.capacity_frames - self.available_frames

  def reset(self) -> None:
    self.header.write_cursor = 0
    self.header.read_cursor = 0
    self.header.underrun_frames = 0
    self.header.overrun_frames = 0
    self.header.low_water_frames = self.capacity_frames
    self._data.fill(0)

  def write(self, chunk: np.ndarray) -> dict[str, int]:
    """Writes interleaved int16 stereo rows, overwriting oldest data if needed."""

    chunk = _validate_s16_interleaved_stereo(chunk)
    frames = int(chunk.shape[0])
    dropped = 0
    if frames > self.capacity_frames:
      dropped += frames - self.capacity_frames
      chunk = chunk[-self.capacity_frames :]
      frames = self.capacity_frames

    overflow = max(0, self.available_frames + frames - self.capacity_frames)
    if overflow:
      self.header.read_cursor += overflow
      self.header.overrun_frames += overflow
      dropped += overflow

    self._write_rows(chunk)
    self.header.write_cursor += frames
    self._record_low_water()
    return {
        "written_frames": frames,
        "dropped_frames": dropped,
        "available_frames": self.available_frames,
        "free_frames": self.free_frames,
        "low_water_frames": self.header.normalized_low_water_frames,
    }

  def read(self, frame_count: int) -> np.ndarray:
    """Reads ``frame_count`` frames, zero-filling underruns."""

    if frame_count < 0:
      raise ValueError("frame_count must be non-negative")
    output = np.zeros((frame_count, DEFAULT_CHANNELS), dtype=np.int16)
    readable = min(frame_count, self.available_frames)
    if readable:
      output[:readable] = self._read_rows(readable)
      self.header.read_cursor += readable
    missing = frame_count - readable
    if missing:
      self.header.underrun_frames += missing
    self._record_low_water()
    return output

  def _write_rows(self, rows: np.ndarray) -> None:
    start = self.header.write_cursor % self.capacity_frames
    first = min(rows.shape[0], self.capacity_frames - start)
    self._data[start : start + first] = rows[:first]
    remaining = rows.shape[0] - first
    if remaining:
      self._data[:remaining] = rows[first:]

  def _read_rows(self, frame_count: int) -> np.ndarray:
    start = self.header.read_cursor % self.capacity_frames
    first = min(frame_count, self.capacity_frames - start)
    if first == frame_count:
      return self._data[start : start + first].copy()
    return np.concatenate(
        [self._data[start : start + first], self._data[: frame_count - first]],
        axis=0,
    ).copy()

  def _record_low_water(self) -> None:
    self.header.low_water_frames = min(
        self.header.normalized_low_water_frames,
        max(0, self.available_frames),
    )


class SharedMemoryInt16InterleavedStereoRing:
  """File-backed shared-memory variant of the ring v1 payload.

  The path is an ordinary mmap-able file so the producer and ALSA host can be
  launched by hand without a named-shm dependency. Cursor units are audio
  frames, matching ``Int16InterleavedStereoRing`` and the Rust host.
  """

  def __init__(
      self,
      path: str | Path,
      *,
      capacity_frames: int,
      chunk_frames: int = DEFAULT_CHUNK_FRAMES,
      sample_rate: int = DEFAULT_SAMPLE_RATE,
      model_frame_size: int = DEFAULT_MODEL_FRAME_SIZE,
      create: bool = True,
  ):
    if capacity_frames <= 0:
      raise ValueError("capacity_frames must be positive")
    if chunk_frames <= 0:
      raise ValueError("chunk_frames must be positive")
    self.path = Path(path)
    self.path.parent.mkdir(parents=True, exist_ok=True)
    self._size = _ring_file_size(capacity_frames)
    mode = "w+b" if create else "r+b"
    self._file = self.path.open(mode)
    if create:
      self._file.truncate(self._size)
    elif self.path.stat().st_size < self._size:
      raise ValueError(f"ring file is too small: {self.path}")
    self._mmap = mmap.mmap(self._file.fileno(), self._size)
    if create:
      header = RingHeader(
          sample_rate=sample_rate,
          model_frame_size=model_frame_size,
          chunk_frames=chunk_frames,
          capacity_frames=capacity_frames,
          low_water_frames=capacity_frames,
      )
      self._write_header(header)
      self._data.fill(0)
    else:
      self.header.validate_consumer(
          sample_rate=sample_rate,
          channels=DEFAULT_CHANNELS,
      )

  @property
  def header(self) -> RingHeader:
    return RingHeader.from_bytes(self._mmap[:RING_HEADER_SIZE])

  @property
  def capacity_frames(self) -> int:
    return self.header.capacity_frames

  @property
  def available_frames(self) -> int:
    header = self.header
    return int(header.write_cursor - header.read_cursor)

  @property
  def free_frames(self) -> int:
    return self.capacity_frames - self.available_frames

  @property
  def _data(self) -> np.ndarray:
    return np.ndarray(
        (self.capacity_frames, DEFAULT_CHANNELS),
        dtype=np.int16,
        buffer=self._mmap,
        offset=RING_HEADER_SIZE,
    )

  def reset(self) -> None:
    header = self.header
    header.write_cursor = 0
    header.read_cursor = 0
    header.underrun_frames = 0
    header.overrun_frames = 0
    header.low_water_frames = header.capacity_frames
    self._write_header(header)
    self._data.fill(0)

  def close(self) -> None:
    self._mmap.flush()
    self._mmap.close()
    self._file.close()

  def write(self, chunk: np.ndarray) -> dict[str, int]:
    chunk = _validate_s16_interleaved_stereo(chunk)
    header = self.header
    frames = int(chunk.shape[0])
    dropped = 0
    if frames > header.capacity_frames:
      dropped += frames - header.capacity_frames
      chunk = chunk[-header.capacity_frames :]
      frames = header.capacity_frames

    available = int(header.write_cursor - header.read_cursor)
    overflow = max(0, available + frames - header.capacity_frames)
    if overflow:
      header.read_cursor += overflow
      header.overrun_frames += overflow
      dropped += overflow

    self._write_rows(chunk, header)
    header.write_cursor += frames
    header.low_water_frames = min(
        header.normalized_low_water_frames,
        max(0, int(header.write_cursor - header.read_cursor)),
    )
    self._write_header(header)
    return {
        "written_frames": frames,
        "dropped_frames": dropped,
        "available_frames": int(header.write_cursor - header.read_cursor),
        "free_frames": int(
            header.capacity_frames - (header.write_cursor - header.read_cursor)
        ),
        "low_water_frames": header.normalized_low_water_frames,
    }

  def _write_header(self, header: RingHeader) -> None:
    self._mmap[:RING_HEADER_SIZE] = header.to_bytes()

  def _write_rows(self, rows: np.ndarray, header: RingHeader) -> None:
    start = header.write_cursor % header.capacity_frames
    first = min(rows.shape[0], header.capacity_frames - start)
    data = self._data
    data[start : start + first] = rows[:first]
    remaining = rows.shape[0] - first
    if remaining:
      data[:remaining] = rows[first:]
    self._mmap.flush()


class PacingRingWriter:
  """Waits for ring space before delegating to a ring's normal ``write``.

  ``write`` intentionally keeps overwrite semantics for diagnostics and replay.
  This wrapper is for live audio paths, where a fast producer should pace to the
  consumer instead of overwriting samples that the audio host has not played yet.
  If a timeout is configured, the wrapper records the timeout and lets the
  underlying ring write decide what to drop.
  """

  def __init__(
      self,
      ring,
      *,
      poll_seconds: float = 0.001,
      timeout_seconds: float | None = None,
      should_stop: ShouldStop | None = None,
  ):
    if poll_seconds <= 0:
      raise ValueError("poll_seconds must be positive")
    if timeout_seconds is not None and timeout_seconds < 0:
      raise ValueError("timeout_seconds must be non-negative")
    self.ring = ring
    self.poll_seconds = float(poll_seconds)
    self.timeout_seconds = timeout_seconds
    self.should_stop = should_stop
    self.wait_seconds = 0.0
    self.wait_count = 0
    self.timeout_count = 0
    self.stopped_count = 0

  @property
  def available_frames(self) -> int:
    return int(self.ring.available_frames)

  @property
  def free_frames(self) -> int:
    return int(self.ring.free_frames)

  def __call__(self, chunk: np.ndarray) -> dict[str, int | float | str]:
    return self.write(chunk)

  def write(self, chunk: np.ndarray) -> dict[str, int | float | str]:
    chunk = _validate_s16_interleaved_stereo(chunk)
    needed = min(int(chunk.shape[0]), int(self.ring.capacity_frames))
    wait_seconds, status, waited = self._wait_for_free_frames(needed)
    self.wait_seconds += wait_seconds
    if waited:
      self.wait_count += 1
    if status == "stopped":
      self.stopped_count += 1
      return self._skipped_result(wait_seconds, status)

    result = dict(self.ring.write(chunk))
    result["backpressure_wait_seconds"] = wait_seconds
    result["backpressure_status"] = status
    result["backpressure_wait_count"] = self.wait_count
    result["backpressure_timeout_count"] = self.timeout_count
    result["backpressure_stopped_count"] = self.stopped_count
    return result

  def _wait_for_free_frames(self, needed: int) -> tuple[float, str, bool]:
    started = time.perf_counter()
    waited = False
    deadline = (
        None
        if self.timeout_seconds is None
        else started + self.timeout_seconds
    )
    while self.free_frames < needed:
      waited = True
      if self.should_stop is not None and self.should_stop():
        return time.perf_counter() - started, "stopped", waited
      if deadline is not None:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
          self.timeout_count += 1
          return time.perf_counter() - started, "timeout", waited
        time.sleep(min(self.poll_seconds, remaining))
      else:
        time.sleep(self.poll_seconds)
    return time.perf_counter() - started, "ready", waited

  def _skipped_result(
      self, wait_seconds: float, status: str
  ) -> dict[str, int | float | str]:
    header = self.ring.header
    available = int(header.write_cursor - header.read_cursor)
    return {
        "written_frames": 0,
        "dropped_frames": 0,
        "available_frames": available,
        "free_frames": int(header.capacity_frames - available),
        "low_water_frames": header.normalized_low_water_frames,
        "backpressure_wait_seconds": wait_seconds,
        "backpressure_status": status,
        "backpressure_wait_count": self.wait_count,
        "backpressure_timeout_count": self.timeout_count,
        "backpressure_stopped_count": self.stopped_count,
    }


def _validate_s16_interleaved_stereo(chunk: np.ndarray) -> np.ndarray:
  if not isinstance(chunk, np.ndarray):
    raise TypeError("ring payload must be a numpy array")
  if chunk.dtype != np.int16:
    raise TypeError(f"ring payload must be int16, got {chunk.dtype}")
  if chunk.ndim != 2 or chunk.shape[1] != DEFAULT_CHANNELS:
    raise ValueError(
        "ring payload must have shape [audio_frames, 2] for stereo"
    )
  return chunk


def _ring_file_size(capacity_frames: int) -> int:
  return RING_HEADER_SIZE + capacity_frames * DEFAULT_CHANNELS * 2
