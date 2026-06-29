#!/usr/bin/env python3
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

"""Benchmark JAX live producer modes on existing full PC4MS ADG takes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any

import jax

from magenta_rt import MagentaRT2Jax
from magenta_rt import adg_dataset
from magenta_rt import aig_bridge
from magenta_rt.jax.live_producer import JaxLiveProducer
from magenta_rt.jax.live_producer import ProducerMode
from magenta_rt.jax.system import discretize_cfg
from magenta_rt.live_ring import DEFAULT_MODEL_FRAME_SIZE
from magenta_rt.live_ring import Int16InterleavedStereoRing


DEFAULT_DATASET_ROOT = Path(
    os.environ.get(
        "PC4MS_ADG_TAKE_ROOT",
        "/home/dev/work-base-20260421/workspace/systems/pc4-microkit-studio/"
        ".pc4ms/session-store/workbench-session/generated-drum-midi-takes",
    )
)
DEFAULT_LOG = Path("outputs/jax_live_producer/benchmark.jsonl")


def main() -> None:
  args = parse_args()
  modes = _selected_modes(args.mode)
  take = _select_take(args)
  schedule = _schedule_from_take(take, args.max_frames, args.tail_seconds)
  producer = _create_producer(args)

  args.log.parent.mkdir(parents=True, exist_ok=True)
  records = []
  for chunk_frames in _selected_chunk_frames(args):
    prepared = _prepare_schedule(args, producer, schedule, chunk_frames)
    for mode in modes:
      ring = None
      null_host = None
      if args.with_ring or args.null_host:
        ring = Int16InterleavedStereoRing(
            capacity_frames=(
                DEFAULT_MODEL_FRAME_SIZE * chunk_frames * args.ring_chunks
            ),
            chunk_frames=chunk_frames,
        )
      if args.null_host:
        assert ring is not None
        null_host = NullHostConsumer(
            ring,
            prime_frames=DEFAULT_MODEL_FRAME_SIZE * chunk_frames,
        )
      if mode == "scan-chunked":
        producer.precompile_scan_chunks(prepared)
      samples_i16, _, metrics = producer.generate_int16(
          prepared,
          mode=mode,
          ring_writer=_ring_writer(ring, null_host),
      )
      record = {
          "schema": "magenta_rt.jax_live_producer_benchmark.v1",
          "timestamp": int(time.time()),
          "take": {
              "take_id": take.take_id,
              "manifest": str(take.manifest),
              "adg_toml": str(take.adg_toml),
              "adg_events": str(take.adg_events),
          },
          "model": {
              "name": args.model,
              "checkpoint": args.checkpoint,
              "devices": [str(device) for device in jax.devices()],
          },
          "conditioning": {
              "frames": int(schedule.summary["frames"]),
              "duration_seconds": float(schedule.summary["duration_seconds"]),
              "schema": schedule.summary["schema"],
          },
          "producer": metrics.to_dict(),
          "output": {
              "shape": list(samples_i16.shape),
              "dtype": str(samples_i16.dtype),
          },
          "ring": None if ring is None else {
              "available_frames": ring.available_frames,
              "underrun_frames": ring.header.underrun_frames,
              "overrun_frames": ring.header.overrun_frames,
              "null_host_consumed_frames": (
                  None if null_host is None else null_host.consumed_frames
              ),
          },
      }
      records.append(record)
      with args.log.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
      print(
          f"{mode} chunk={chunk_frames}: "
          f"{metrics.steps_per_second:.1f} steps/s, "
          f"RTF {metrics.rtf:.3f}, copy {metrics.copy_seconds:.3f}s, "
          f"dispatch {metrics.dispatch_seconds:.3f}s"
      )

  if args.gate:
    _check_gate(records)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Benchmark JAX producer modes on full ADG takes only."
  )
  parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
  parser.add_argument("--take-id", default=None)
  parser.add_argument(
      "--adg-bundle-manifest",
      type=Path,
      default=None,
      help="Explicit full ADG take manifest.json.",
  )
  parser.add_argument("--model", default="mrt2_small")
  parser.add_argument("--checkpoint", default=None)
  parser.add_argument("--prompt", default="tight acoustic funk drums")
  parser.add_argument("--temperature", type=float, default=1.1)
  parser.add_argument("--top-k", type=int, default=40)
  parser.add_argument("--cfg-musiccoca", type=float, default=3.0)
  parser.add_argument("--cfg-notes", type=float, default=3.0)
  parser.add_argument("--cfg-drums", type=float, default=4.0)
  parser.add_argument("--max-frames", type=int, default=64)
  parser.add_argument("--tail-seconds", type=float, default=0.0)
  parser.add_argument("--chunk-frames", type=int, default=8)
  parser.add_argument(
      "--chunk-frame-options",
      default=None,
      help="Comma-separated chunk sizes to benchmark with one loaded model.",
  )
  parser.add_argument(
      "--mode",
      choices=[
          "serial",
          "sync-diagnosis",
          "pipeline-1",
          "chunked",
          "scan-chunked",
          "all",
      ],
      default="all",
  )
  parser.add_argument("--with-ring", action="store_true")
  parser.add_argument(
      "--null-host",
      action="store_true",
      help="Consume the int16 ring after one chunk of priming.",
  )
  parser.add_argument("--ring-chunks", type=int, default=4)
  parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
  parser.add_argument("--gate", action="store_true")
  return parser.parse_args()


def _select_take(args: argparse.Namespace) -> adg_dataset.FullAdgTake:
  if args.adg_bundle_manifest is not None:
    return adg_dataset.load_full_adg_take(args.adg_bundle_manifest)
  return adg_dataset.select_full_adg_take(
      args.dataset_root, take_id=args.take_id
  )


def _schedule_from_take(
    take: adg_dataset.FullAdgTake, max_frames: int | None, tail_seconds: float
) -> aig_bridge.ConditioningSchedule:
  events = aig_bridge.load_pc4ms_adg_events(take.adg_events)
  timeline = aig_bridge.load_pc4ms_adg_timeline(take.adg_toml)
  return aig_bridge.build_conditioning_schedule_from_pc4ms_adg_events(
      events,
      tempo_bpm=float(timeline["tempo_bpm"]),
      ppqn=int(timeline["ppqn"]),
      tail_seconds=tail_seconds,
      max_frames=max_frames,
  )


def _create_producer(args: argparse.Namespace) -> JaxLiveProducer:
  mrt = MagentaRT2Jax(
      size=args.model,
      checkpoint=args.checkpoint,
      temperature=args.temperature,
      top_k=args.top_k,
      cfg_musiccoca=args.cfg_musiccoca,
      cfg_notes=args.cfg_notes,
      cfg_drums=args.cfg_drums,
  )
  return JaxLiveProducer(mrt, jax_module=jax)


def _prepare_schedule(
    args: argparse.Namespace,
    producer: JaxLiveProducer,
    schedule: aig_bridge.ConditioningSchedule,
    chunk_frames: int,
) -> Any:
  style_tokens = producer.prepare_style_tokens(args.prompt, use_mapper=True)
  cfgs = [
      discretize_cfg(args.cfg_musiccoca, 0.2, 40),
      discretize_cfg(args.cfg_notes, 0.2, 40),
      discretize_cfg(args.cfg_drums, 1.0, 8),
  ]
  prepared = producer.prepare_schedule(
      schedule,
      style_tokens=style_tokens,
      cfgs=cfgs,
      temperature=args.temperature,
      top_k=args.top_k,
      chunk_frames=chunk_frames,
  )
  return prepared


def _selected_modes(value: str) -> list[ProducerMode]:
  if value == "all":
    return ["serial", "sync-diagnosis", "pipeline-1", "chunked", "scan-chunked"]
  return [value]


def _selected_chunk_frames(args: argparse.Namespace) -> list[int]:
  if args.chunk_frame_options is None:
    return [args.chunk_frames]
  values = []
  for raw in args.chunk_frame_options.split(","):
    value = int(raw.strip())
    if value <= 0:
      raise SystemExit("chunk frame options must be positive")
    values.append(value)
  return values


class NullHostConsumer:
  """Callback-like ring consumer for JAX-to-null smoke runs."""

  def __init__(self, ring: Int16InterleavedStereoRing, *, prime_frames: int):
    self._ring = ring
    self._prime_frames = prime_frames
    self._started = False
    self.consumed_frames = 0

  def write(self, chunk):
    result = self._ring.write(chunk)
    if not self._started and self._ring.available_frames >= self._prime_frames:
      self._started = True
    if self._started:
      self._ring.read(chunk.shape[0])
      self.consumed_frames += int(chunk.shape[0])
    result["available_frames"] = self._ring.available_frames
    return result


def _ring_writer(ring, null_host):
  if null_host is not None:
    return null_host.write
  if ring is not None:
    return ring.write
  return None


def _check_gate(records: list[dict[str, Any]]) -> None:
  chunked = [r for r in records if r["producer"]["mode"] == "chunked"]
  if not chunked:
    raise SystemExit("gate requires a chunked producer record")
  scan_chunked = [
      r for r in records if r["producer"]["mode"] == "scan-chunked"
  ]
  if scan_chunked:
    baseline = [
        r for r in chunked if r["producer"]["chunk_frames"] == 8
    ] or chunked
    baseline_metrics = baseline[-1]["producer"]
    best_scan = max(
        scan_chunked,
        key=lambda r: r["producer"]["steps_per_second"],
    )
    scan_metrics = best_scan["producer"]
    if (
        scan_metrics["steps_per_second"]
        < baseline_metrics["steps_per_second"] * 1.05
    ):
      raise SystemExit(
          "Scan gate failed: best scan-chunked producer did not clear "
          ">=5% over chunked baseline"
      )
    return
  metrics = chunked[-1]["producer"]
  if metrics["steps_per_second"] < 30.0 or metrics["rtf"] > 0.833:
    raise SystemExit(
        "Gate B failed: chunked producer did not clear "
        ">=30 steps/s and RTF <=0.833"
    )
  if metrics["prompt_embedded_in_hot_loop"]:
    raise SystemExit("Gate B failed: prompt embedding happened in hot loop")


if __name__ == "__main__":
  main()
