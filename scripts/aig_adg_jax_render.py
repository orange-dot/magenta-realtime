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

"""Render AIG/ADG GesturePacket artifacts through Magenta RT JAX.

This is a lossy bridge: AIG/ADG remains the semantic authority, while Magenta
RealTime 2 receives only the conditioning lanes it already exposes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import jax
import numpy as np

from magenta_rt import MagentaRT2Jax
from magenta_rt import adg_dataset
from magenta_rt import aig_bridge
from magenta_rt import audio
from magenta_rt.jax.live_producer import JaxLiveProducer
from magenta_rt.jax.system import discretize_cfg


def main():
  args = parse_args()

  packet_document = None
  atom_document = aig_bridge.load_atom_specs(args.atom_specs)
  full_adg_take = None
  if args.adg_bundle_manifest is not None:
    full_adg_take = adg_dataset.load_full_adg_take(args.adg_bundle_manifest)
    events = aig_bridge.load_pc4ms_adg_events(full_adg_take.adg_events)
    timeline = aig_bridge.load_pc4ms_adg_timeline(full_adg_take.adg_toml)
    schedule = aig_bridge.build_conditioning_schedule_from_pc4ms_adg_events(
        events,
        tempo_bpm=float(timeline["tempo_bpm"]),
        ppqn=int(timeline["ppqn"]),
        tail_seconds=args.tail_seconds,
        max_frames=args.max_frames,
    )
  else:
    assert args.gesture_packets is not None
    packet_document = aig_bridge.load_gesture_packets(args.gesture_packets)
    schedule = aig_bridge.build_conditioning_schedule(
        packet_document,
        tail_seconds=args.tail_seconds,
        max_frames=args.max_frames,
    )

  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.manifest.parent.mkdir(parents=True, exist_ok=True)
  args.conditioning.parent.mkdir(parents=True, exist_ok=True)

  np.savez_compressed(
      args.conditioning,
      notes=schedule.notes,
      drums=schedule.drums,
  )

  start = time.time()
  waveform = render_with_jax(args, schedule)
  elapsed = time.time() - start
  waveform.write(str(args.output))

  manifest = build_manifest(
      args,
      packet_document,
      atom_document,
      full_adg_take,
      schedule,
      waveform,
      elapsed,
  )
  args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

  frames = schedule.summary["frames"]
  print(
      f"Rendered {frames} frames in {elapsed:.1f}s "
      f"({frames / elapsed:.1f} steps/s)"
  )
  print(f"Saved WAV: {args.output}")
  print(f"Saved conditioning: {args.conditioning}")
  print(f"Saved manifest: {args.manifest}")


def parse_args():
  parser = argparse.ArgumentParser(
      description="Render AIG GesturePacket conditioning through MRT2 JAX."
  )
  source = parser.add_mutually_exclusive_group(required=True)
  source.add_argument("--gesture-packets", type=Path)
  source.add_argument(
      "--adg-bundle-manifest",
      type=Path,
      help="Existing PC4MS full ADG take manifest.json.",
  )
  parser.add_argument("--atom-specs", type=Path, default=None)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--manifest", type=Path, required=True)
  parser.add_argument("--conditioning", type=Path, required=True)
  parser.add_argument("--model", default="mrt2_small")
  parser.add_argument("--prompt", required=True)
  parser.add_argument("--checkpoint", default=None)
  parser.add_argument("--temperature", type=float, default=1.1)
  parser.add_argument("--top-k", type=int, default=40)
  parser.add_argument("--cfg-musiccoca", type=float, default=3.0)
  parser.add_argument("--cfg-notes", type=float, default=3.0)
  parser.add_argument("--cfg-drums", type=float, default=4.0)
  parser.add_argument("--tail-seconds", type=float, default=2.0)
  parser.add_argument(
      "--max-frames",
      type=int,
      default=None,
      help="Debug limiter; omit for full-take rendering.",
  )
  parser.add_argument("--progress-every", type=int, default=25)
  parser.add_argument("--chunk-frames", type=int, default=8)
  parser.add_argument(
      "--producer-mode",
      choices=[
          "serial",
          "sync-diagnosis",
          "pipeline-1",
          "chunked",
          "scan-chunked",
      ],
      default="chunked",
  )
  return parser.parse_args()


def render_with_jax(args, schedule: aig_bridge.ConditioningSchedule):
  mrt = MagentaRT2Jax(
      size=args.model,
      checkpoint=args.checkpoint,
      temperature=args.temperature,
      top_k=args.top_k,
      cfg_musiccoca=args.cfg_musiccoca,
      cfg_notes=args.cfg_notes,
      cfg_drums=args.cfg_drums,
  )

  producer = JaxLiveProducer(mrt, jax_module=jax)
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
      chunk_frames=args.chunk_frames,
  )
  if args.producer_mode == "scan-chunked":
    producer.precompile_scan_chunks(prepared)
  samples_i16, _, metrics = producer.generate_int16(
      prepared,
      mode=args.producer_mode,
  )
  print(
      f"Producer {metrics.mode}: {metrics.steps_per_second:.1f} steps/s, "
      f"RTF {metrics.rtf:.3f}, copy {metrics.copy_seconds:.3f}s"
  )
  samples = samples_i16.astype(np.float32) / 32768.0
  return audio.Waveform(samples, sample_rate=aig_bridge.MAGENTA_SAMPLE_RATE)


def build_manifest(
    args,
    packet_document: dict[str, Any] | None,
    atom_document: dict[str, Any] | None,
    full_adg_take: adg_dataset.FullAdgTake | None,
    schedule: aig_bridge.ConditioningSchedule,
    waveform: audio.Waveform,
    elapsed: float,
) -> dict[str, Any]:
  devices = [str(device) for device in jax.devices()]
  atom_count = len(atom_document.get("atoms", [])) if atom_document else None
  if full_adg_take is not None:
    source = {
        "kind": "pc4ms_full_adg_take",
        "take_id": full_adg_take.take_id,
        "adg_bundle_manifest": str(full_adg_take.manifest),
        "adg_toml": str(full_adg_take.adg_toml),
        "adg_events": str(full_adg_take.adg_events),
        "adg_summary": str(full_adg_take.adg_summary),
        "midi": str(full_adg_take.midi),
        "atom_count": atom_count,
    }
  else:
    assert packet_document is not None
    source = {
        "kind": "aig_gesture_packets",
        "gesture_packets": str(args.gesture_packets),
        "atom_specs": str(args.atom_specs) if args.atom_specs else None,
        "packet_title": packet_document.get("title"),
        "packet_count": len(packet_document.get("packets", [])),
        "atom_count": atom_count,
    }
  return {
      "schema": "magenta_rt.aig_adg_jax_render.v1",
      "source": source,
      "magenta": {
          "backend": "jax",
          "model": args.model,
          "checkpoint": args.checkpoint,
          "prompt": args.prompt,
          "temperature": args.temperature,
          "top_k": args.top_k,
          "cfg_musiccoca": args.cfg_musiccoca,
          "cfg_notes": args.cfg_notes,
          "cfg_drums": args.cfg_drums,
          "producer_mode": args.producer_mode,
          "chunk_frames": args.chunk_frames,
          "devices": devices,
          "elapsed_seconds": elapsed,
          "steps_per_second": schedule.summary["frames"] / elapsed,
      },
      "conditioning": {
          **schedule.summary,
          "conditioning_npz": str(args.conditioning),
          "packet_reports": schedule.packet_reports,
      },
      "output": {
          "wav": str(args.output),
          "audio": aig_bridge.audio_stats(waveform.samples, waveform.sample_rate),
      },
  }


if __name__ == "__main__":
  main()
