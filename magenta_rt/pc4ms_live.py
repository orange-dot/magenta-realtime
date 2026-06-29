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

"""PC4MS live ADG chunk receiver for the Magenta RT ALSA preview path."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path
import signal
import socket
import time
from typing import Any

import numpy as np

from magenta_rt import aig_bridge
from magenta_rt.jax.live_producer import JaxLiveProducer
from magenta_rt.live_ring import DEFAULT_MODEL_FRAME_SIZE
from magenta_rt.live_ring import PacingRingWriter
from magenta_rt.live_ring import SharedMemoryInt16InterleavedStereoRing


PC4MS_LIVE_CHUNK_SCHEMA = "pc4ms.magenta_live_chunk.v1"
DAEMON_METRICS_SCHEMA = "magenta_rt.pc4ms_live_daemon_metrics.v1"
DEFAULT_SOCKET_PATH = Path("/tmp/pc4ms-magenta-live.sock")
DEFAULT_RING_PATH = Path("/tmp/mrt2-pc4ms-live.ring")


@dataclasses.dataclass(frozen=True)
class Pc4msLiveChunk:
  """Validated live chunk message from the PC4MS runtime."""

  take_id: str
  chunk_index: int
  tempo_bpm: float
  ppqn: int
  output_start_tick: int
  output_end_tick: int
  adg_events: list[dict[str, Any]]
  metrics: dict[str, Any]
  raw: dict[str, Any]


def parse_live_chunk_message(data: bytes | str) -> Pc4msLiveChunk:
  """Parses the `pc4ms.magenta_live_chunk.v1` JSON socket payload."""

  if isinstance(data, bytes):
    payload = json.loads(data.decode("utf-8"))
  else:
    payload = json.loads(data)
  if not isinstance(payload, dict):
    raise ValueError("live chunk payload must be a JSON object")
  if payload.get("schema") != PC4MS_LIVE_CHUNK_SCHEMA:
    raise ValueError(
        "live chunk schema mismatch: "
        f"{payload.get('schema')!r} != {PC4MS_LIVE_CHUNK_SCHEMA!r}"
    )
  timing = payload.get("timing")
  if not isinstance(timing, dict):
    raise ValueError("live chunk payload requires a timing object")
  adg_events = payload.get("adg_events")
  if not isinstance(adg_events, list):
    raise ValueError("live chunk payload requires an adg_events list")
  for index, event in enumerate(adg_events):
    if not isinstance(event, dict):
      raise ValueError(f"adg_events[{index}] must be an object")
  metrics = payload.get("metrics")
  if not isinstance(metrics, dict):
    metrics = {}
  tempo_bpm = float(payload.get("tempo_bpm", 0.0))
  ppqn = int(payload.get("ppqn", 0))
  if tempo_bpm <= 0:
    raise ValueError("live chunk tempo_bpm must be positive")
  if ppqn <= 0:
    raise ValueError("live chunk ppqn must be positive")
  return Pc4msLiveChunk(
      take_id=str(payload.get("take_id", "")),
      chunk_index=int(payload.get("chunk_index", 0)),
      tempo_bpm=tempo_bpm,
      ppqn=ppqn,
      output_start_tick=int(timing.get("output_start_tick", 0)),
      output_end_tick=int(timing.get("output_end_tick", 0)),
      adg_events=adg_events,
      metrics=metrics,
      raw=payload,
  )


def conditioning_schedule_from_live_chunk(
    chunk: Pc4msLiveChunk,
    *,
    tail_seconds: float = 0.0,
    max_frames: int | None = None,
) -> aig_bridge.ConditioningSchedule:
  """Builds MRT2 conditioning from the already-authoritative PC4MS ADG."""

  schedule = aig_bridge.build_conditioning_schedule_from_pc4ms_adg_events(
      _chunk_local_adg_events(chunk),
      tempo_bpm=chunk.tempo_bpm,
      ppqn=chunk.ppqn,
      tail_seconds=tail_seconds,
      max_frames=max_frames,
  )
  min_frames = _chunk_window_frames(chunk)
  if max_frames is not None:
    min_frames = min(min_frames, max_frames)
  return _pad_schedule_to_min_frames(schedule, min_frames)


def live_payload_from_debug_chunk(
    *,
    take_id: str,
    tempo_bpm: float,
    ppqn: int,
    debug_chunk: dict[str, Any],
    profile_id: str = "",
) -> dict[str, Any]:
  """Adapts a saved PC4MS debug live chunk to the socket contract."""

  run = debug_chunk.get("run")
  if not isinstance(run, dict):
    run = {}
  adg_events = run.get("adg_events")
  if not isinstance(adg_events, list):
    adg_events = []
  chunk_index = int(debug_chunk.get("chunk_index", 0))
  return {
      "schema": PC4MS_LIVE_CHUNK_SCHEMA,
      "take_id": str(take_id),
      "chunk_index": chunk_index,
      "tempo_bpm": float(tempo_bpm),
      "ppqn": int(ppqn),
      "profile_id": str(profile_id),
      "timing": {
          "source_start_tick": int(debug_chunk.get("source_start_tick", 0)),
          "source_end_tick": int(debug_chunk.get("source_end_tick", 0)),
          "output_start_tick": int(debug_chunk.get("output_start_tick", 0)),
          "output_end_tick": int(debug_chunk.get("output_end_tick", 0)),
      },
      "metrics": {
          "generation_micros": int(debug_chunk.get("generation_micros", 0)),
          "intake_event_count": int(debug_chunk.get("intake_event_count", 0)),
          "intake_fallback_used": bool(
              debug_chunk.get("intake_fallback_used", False)
          ),
          "replay": True,
      },
      "adg_events": adg_events,
  }


def send_live_payloads(
    *,
    socket_path: str | Path,
    payloads: list[dict[str, Any]],
    pace_realtime: bool,
    prime_chunks: int,
    timeout_seconds: float,
) -> dict[str, Any]:
  """Sends prepared live payloads to a Unix datagram daemon."""

  socket_path = Path(socket_path)
  deadline = time.monotonic() + timeout_seconds
  while not socket_path.exists():
    if time.monotonic() >= deadline:
      raise TimeoutError(f"socket did not appear: {socket_path}")
    time.sleep(0.05)

  sent = 0
  sent_audio_seconds = 0.0
  started = time.perf_counter()
  sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
  try:
    for payload in payloads:
      data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
      sock.sendto(data, str(socket_path))
      sent += 1
      duration = _payload_chunk_seconds(payload)
      sent_audio_seconds += duration
      if pace_realtime and sent > prime_chunks:
        time.sleep(duration)
  finally:
    sock.close()
  elapsed = time.perf_counter() - started
  return {
      "schema": "magenta_rt.pc4ms_live_replay_metrics.v1",
      "socket": str(socket_path),
      "chunks_sent": sent,
      "audio_seconds_sent": sent_audio_seconds,
      "elapsed_seconds": elapsed,
      "pace_realtime": bool(pace_realtime),
      "prime_chunks": int(prime_chunks),
  }


def _chunk_local_adg_events(chunk: Pc4msLiveChunk) -> list[dict[str, Any]]:
  base_tick = int(chunk.output_start_tick)
  localized = []
  for event in chunk.adg_events:
    local = dict(event)
    local["tick"] = int(local.get("tick", 0)) - base_tick
    localized.append(local)
  return localized


def _chunk_window_frames(chunk: Pc4msLiveChunk) -> int:
  tick_span = max(0, int(chunk.output_end_tick) - int(chunk.output_start_tick))
  if tick_span == 0:
    return 1
  seconds = tick_span * 60.0 / (chunk.tempo_bpm * chunk.ppqn)
  return max(1, math.ceil(seconds * aig_bridge.MAGENTA_FRAME_RATE))


def _pad_schedule_to_min_frames(
    schedule: aig_bridge.ConditioningSchedule, min_frames: int
) -> aig_bridge.ConditioningSchedule:
  current_frames = int(schedule.summary["frames"])
  if current_frames >= min_frames:
    return schedule
  extra = min_frames - current_frames
  notes = np.pad(schedule.notes, ((0, extra), (0, 0)), mode="constant")
  drums = np.pad(schedule.drums, ((0, extra), (0, 0)), mode="constant")
  frame_rate = int(schedule.summary["frame_rate"])
  summary = {
      **schedule.summary,
      "frames": int(min_frames),
      "duration_seconds": min_frames / frame_rate,
      "chunk_window_min_frames": int(min_frames),
  }
  return aig_bridge.ConditioningSchedule(
      notes=notes,
      drums=drums,
      summary=summary,
      packet_reports=schedule.packet_reports,
  )


def _payload_chunk_seconds(payload: dict[str, Any]) -> float:
  timing = payload.get("timing")
  if not isinstance(timing, dict):
    return 0.0
  tempo_bpm = float(payload.get("tempo_bpm", 0.0))
  ppqn = int(payload.get("ppqn", 0))
  if tempo_bpm <= 0 or ppqn <= 0:
    return 0.0
  tick_span = max(
      0,
      int(timing.get("output_end_tick", 0))
      - int(timing.get("output_start_tick", 0)),
  )
  return tick_span * 60.0 / (tempo_bpm * ppqn)


class UnixDatagramLiveChunkReceiver:
  """Small Unix datagram receiver for PC4MS live ADG chunks."""

  def __init__(self, path: str | Path, *, max_payload_bytes: int = 1_048_576):
    self.path = Path(path)
    self.max_payload_bytes = int(max_payload_bytes)
    self._socket: socket.socket | None = None

  def __enter__(self) -> "UnixDatagramLiveChunkReceiver":
    self.path.parent.mkdir(parents=True, exist_ok=True)
    try:
      self.path.unlink()
    except FileNotFoundError:
      pass
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    sock.bind(str(self.path))
    self._socket = sock
    return self

  def __exit__(self, exc_type, exc, tb) -> None:
    if self._socket is not None:
      self._socket.close()
      self._socket = None
    try:
      self.path.unlink()
    except FileNotFoundError:
      pass

  def receive(self, *, timeout: float | None = None) -> Pc4msLiveChunk:
    if self._socket is None:
      raise RuntimeError("receiver is not bound")
    self._socket.settimeout(timeout)
    data = self._socket.recv(self.max_payload_bytes)
    return parse_live_chunk_message(data)


class Pc4msMagentaLiveDaemon:
  """Long-lived JAX sidecar that turns PC4MS ADG chunks into ring audio."""

  def __init__(
      self,
      *,
      producer: JaxLiveProducer,
      ring: SharedMemoryInt16InterleavedStereoRing,
      prompt: str,
      chunk_frames: int,
      temperature: float,
      top_k: int,
      cfg_musiccoca: float,
      cfg_notes: float,
      cfg_drums: float,
      max_frames: int | None = None,
      tail_seconds: float = 0.0,
      ring_backpressure: bool = True,
      ring_write_poll_seconds: float = 0.001,
      ring_write_timeout_seconds: float | None = None,
      should_stop: Any | None = None,
  ):
    if chunk_frames not in (4, 8):
      raise ValueError("pc4ms live test supports --chunk-frames 4 or 8")
    from magenta_rt.jax.system import discretize_cfg

    self._producer = producer
    self._ring = ring
    self._prompt = prompt
    self._chunk_frames = int(chunk_frames)
    self._temperature = float(temperature)
    self._top_k = int(top_k)
    self._cfgs = [
        discretize_cfg(cfg_musiccoca, 0.2, 40),
        discretize_cfg(cfg_notes, 0.2, 40),
        discretize_cfg(cfg_drums, 1.0, 8),
    ]
    self._ring_writer = (
        PacingRingWriter(
            ring,
            poll_seconds=ring_write_poll_seconds,
            timeout_seconds=ring_write_timeout_seconds,
            should_stop=should_stop,
        )
        if ring_backpressure
        else ring.write
    )
    self._max_frames = max_frames
    self._tail_seconds = tail_seconds
    self._style_tokens = producer.prepare_style_tokens(prompt, use_mapper=True)
    self._state = producer.init_state()
    self.records: list[dict[str, Any]] = []

  def handle_chunk(self, chunk: Pc4msLiveChunk) -> dict[str, Any]:
    schedule = conditioning_schedule_from_live_chunk(
        chunk,
        tail_seconds=self._tail_seconds,
        max_frames=self._max_frames,
    )
    prepared = self._producer.prepare_schedule(
        schedule,
        style_tokens=self._style_tokens,
        cfgs=self._cfgs,
        temperature=self._temperature,
        top_k=self._top_k,
        chunk_frames=self._chunk_frames,
    )
    samples, self._state, producer_metrics = self._producer.generate_int16(
        prepared,
        state=self._state,
        mode="chunked",
        ring_writer=self._ring_writer,
    )
    ring_header = self._ring.header
    record = {
        "schema": DAEMON_METRICS_SCHEMA,
        "timestamp": int(time.time()),
        "source": {
            "take_id": chunk.take_id,
            "chunk_index": chunk.chunk_index,
            "tempo_bpm": chunk.tempo_bpm,
            "ppqn": chunk.ppqn,
            "output_start_tick": chunk.output_start_tick,
            "output_end_tick": chunk.output_end_tick,
            "pc4ms_metrics": chunk.metrics,
        },
        "conditioning": schedule.summary,
        "producer": producer_metrics.to_dict(),
        "ring": {
            "path": str(self._ring.path),
            "available_frames": self._ring.available_frames,
            "underrun_frames": ring_header.underrun_frames,
            "overrun_frames": ring_header.overrun_frames,
            "low_water_frames": ring_header.low_water_frames,
            "backpressure": _ring_writer_snapshot(self._ring_writer),
        },
        "output": {
            "shape": list(samples.shape),
            "dtype": str(samples.dtype),
        },
    }
    self.records.append(record)
    return record


def run_daemon(args: argparse.Namespace) -> None:
  if args.producer_mode != "chunked":
    raise SystemExit("PC4MS live daemon intentionally supports chunked mode only")
  ring = SharedMemoryInt16InterleavedStereoRing(
      args.ring,
      capacity_frames=DEFAULT_MODEL_FRAME_SIZE * args.chunk_frames * args.ring_chunks,
      chunk_frames=args.chunk_frames,
  )
  stop = False

  def _stop(_signum, _frame):
    nonlocal stop
    stop = True

  signal.signal(signal.SIGINT, _stop)
  signal.signal(signal.SIGTERM, _stop)

  try:
    import jax
    from magenta_rt import MagentaRT2Jax

    mrt = MagentaRT2Jax(
        size=args.model,
        checkpoint=args.checkpoint,
        temperature=args.temperature,
        top_k=args.top_k,
        cfg_musiccoca=args.cfg_musiccoca,
        cfg_notes=args.cfg_notes,
        cfg_drums=args.cfg_drums,
    )
    daemon = Pc4msMagentaLiveDaemon(
        producer=JaxLiveProducer(mrt, jax_module=jax),
        ring=ring,
        prompt=args.prompt,
        chunk_frames=args.chunk_frames,
        temperature=args.temperature,
        top_k=args.top_k,
        cfg_musiccoca=args.cfg_musiccoca,
        cfg_notes=args.cfg_notes,
        cfg_drums=args.cfg_drums,
        max_frames=args.max_frames,
        tail_seconds=args.tail_seconds,
        ring_backpressure=args.ring_backpressure,
        ring_write_poll_seconds=args.ring_write_poll_ms / 1000.0,
        ring_write_timeout_seconds=args.ring_write_timeout_seconds,
        should_stop=lambda: stop,
    )
    with UnixDatagramLiveChunkReceiver(args.socket) as receiver:
      print(
          f"PC4MS Magenta live daemon listening on {args.socket}; "
          f"ring {args.ring}; chunk_frames={args.chunk_frames}"
      )
      while not stop:
        try:
          chunk = receiver.receive(timeout=0.2)
        except socket.timeout:
          continue
        record = daemon.handle_chunk(chunk)
        if args.metrics_log is not None:
          args.metrics_log.parent.mkdir(parents=True, exist_ok=True)
          with args.metrics_log.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(
            f"chunk {chunk.chunk_index}: "
            f"{record['producer']['steps_per_second']:.1f} steps/s, "
            f"RTF {record['producer']['rtf']:.3f}, "
            f"ring available {record['ring']['available_frames']} frames"
        )
  finally:
    ring.close()


def _ring_writer_snapshot(writer: Any) -> dict[str, Any]:
  if isinstance(writer, PacingRingWriter):
    return {
        "enabled": True,
        "wait_seconds": writer.wait_seconds,
        "wait_count": writer.wait_count,
        "timeout_count": writer.timeout_count,
        "stopped_count": writer.stopped_count,
    }
  return {"enabled": False}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Receive PC4MS live ADG chunks and write MRT2 audio to a ring."
  )
  parser.add_argument("--socket", type=Path, default=DEFAULT_SOCKET_PATH)
  parser.add_argument("--ring", type=Path, default=DEFAULT_RING_PATH)
  parser.add_argument("--ring-chunks", type=int, default=4)
  parser.add_argument("--chunk-frames", type=int, choices=[4, 8], default=8)
  parser.add_argument("--model", default="mrt2_small")
  parser.add_argument("--checkpoint", default=None)
  parser.add_argument("--prompt", default="tight acoustic funk drums")
  parser.add_argument("--temperature", type=float, default=1.1)
  parser.add_argument("--top-k", type=int, default=40)
  parser.add_argument("--cfg-musiccoca", type=float, default=3.0)
  parser.add_argument("--cfg-notes", type=float, default=3.0)
  parser.add_argument("--cfg-drums", type=float, default=4.0)
  parser.add_argument("--max-frames", type=int, default=None)
  parser.add_argument("--tail-seconds", type=float, default=0.0)
  parser.add_argument("--producer-mode", choices=["chunked"], default="chunked")
  parser.add_argument("--metrics-log", type=Path, default=None)
  parser.add_argument(
      "--no-ring-backpressure",
      dest="ring_backpressure",
      action="store_false",
      help="Allow live writes to overwrite old ring frames instead of pacing.",
  )
  parser.set_defaults(ring_backpressure=True)
  parser.add_argument("--ring-write-poll-ms", type=float, default=1.0)
  parser.add_argument("--ring-write-timeout-seconds", type=float, default=None)
  return parser.parse_args(argv)


def main() -> None:
  run_daemon(parse_args())


if __name__ == "__main__":
  main()
