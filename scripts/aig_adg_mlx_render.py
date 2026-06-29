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

"""Render AIG/ADG GesturePacket artifacts through Magenta RT MLX.

This is a lossy bridge: AIG/ADG remains the semantic authority, while Magenta
RealTime 2 receives only the conditioning lanes it already exposes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from magenta_rt import aig_bridge
from magenta_rt import audio
from magenta_rt.mlx.runtime import prepare_cpu_runtime_env


def main():
  args = parse_args()

  prepare_cpu_runtime_env(args.device)

  packet_document = aig_bridge.load_gesture_packets(args.gesture_packets)
  atom_document = aig_bridge.load_atom_specs(args.atom_specs)
  source_schedule = aig_bridge.build_conditioning_schedule(
      packet_document,
      tail_seconds=args.tail_seconds,
  )
  schedule = aig_bridge.window_conditioning_schedule(
      source_schedule,
      start_frame=args.start_frame,
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
  waveform = render_with_mlx(args, schedule)
  elapsed = time.time() - start
  waveform.write(str(args.output))

  manifest = build_manifest(
      args,
      packet_document,
      atom_document,
      schedule,
      waveform,
      elapsed,
  )
  args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

  frames = schedule.summary["frames"]
  print(
      f"Rendered {frames} frames in {elapsed:.1f}s "
      f"({frames / elapsed:.2f} steps/s)",
      flush=True,
  )
  print(f"Saved WAV: {args.output}", flush=True)
  print(f"Saved conditioning: {args.conditioning}", flush=True)
  print(f"Saved manifest: {args.manifest}", flush=True)


def parse_args():
  parser = argparse.ArgumentParser(
      description="Render AIG GesturePacket conditioning through MRT2 MLX."
  )
  parser.add_argument("--gesture-packets", type=Path, required=True)
  parser.add_argument("--atom-specs", type=Path, default=None)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--manifest", type=Path, required=True)
  parser.add_argument("--conditioning", type=Path, required=True)
  parser.add_argument("--model", default="mrt2_small")
  parser.add_argument("--prompt", required=True)
  parser.add_argument("--checkpoint", default=None)
  parser.add_argument("--bits", type=int, default=8, choices=[2, 3, 4, 5, 6, 8])
  parser.add_argument("--quantize-group-size", type=int, default=None)
  parser.add_argument("--temperature", type=float, default=1.1)
  parser.add_argument("--top-k", type=int, default=40)
  parser.add_argument("--cfg-musiccoca", type=float, default=3.0)
  parser.add_argument("--cfg-notes", type=float, default=3.0)
  parser.add_argument("--cfg-drums", type=float, default=4.0)
  parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="cpu")
  parser.add_argument("--warmup-steps", type=int, default=0)
  parser.add_argument("--tail-seconds", type=float, default=2.0)
  parser.add_argument(
      "--start-frame",
      type=int,
      default=0,
      help="Magenta conditioning frame to start from.",
  )
  parser.add_argument(
      "--max-frames",
      type=int,
      default=None,
      help="Debug limiter; omit for full-take rendering.",
  )
  parser.add_argument("--progress-every", type=int, default=25)
  return parser.parse_args()


def render_with_mlx(args, schedule: aig_bridge.ConditioningSchedule):
  try:
    from magenta_rt import MagentaRT2Mlx
  except ImportError as exc:
    raise RuntimeError(
        "Could not import the MLX backend. For --device cpu, use an environment "
        "with the CPU MLX wheel; CUDA MLX wheels may still require CUDA shared "
        "libraries before the script can select the CPU device."
    ) from exc

  mrt = MagentaRT2Mlx(
      size=args.model,
      checkpoint=args.checkpoint,
      temperature=args.temperature,
      top_k=args.top_k,
      cfg_musiccoca=args.cfg_musiccoca,
      cfg_notes=args.cfg_notes,
      cfg_drums=args.cfg_drums,
      bits=args.bits,
      quantize_group_size=args.quantize_group_size,
      device=args.device,
      warmup_steps=args.warmup_steps,
  )

  embedding = mrt.embed_style(args.prompt, use_mapper=True)

  state = None
  chunks = []
  frames = schedule.summary["frames"]
  for index in range(frames):
    notes = schedule.notes[index].astype(np.int32).tolist()
    drums = schedule.drums[index].astype(np.int32).tolist()
    step_waveform, state = mrt.generate(
        style=embedding,
        notes=notes,
        drums=drums,
        cfg_musiccoca=args.cfg_musiccoca,
        cfg_notes=args.cfg_notes,
        cfg_drums=args.cfg_drums,
        temperature=args.temperature,
        top_k=args.top_k,
        frames=1,
        state=state,
    )
    chunks.append(step_waveform.samples)
    if args.progress_every > 0 and (index + 1) % args.progress_every == 0:
      print(f"Generated {index + 1}/{frames} frames", flush=True)

  samples = np.concatenate(chunks, axis=0)
  return audio.Waveform(samples, sample_rate=aig_bridge.MAGENTA_SAMPLE_RATE)


def build_manifest(
    args,
    packet_document: dict[str, Any],
    atom_document: dict[str, Any] | None,
    schedule: aig_bridge.ConditioningSchedule,
    waveform: audio.Waveform,
    elapsed: float,
) -> dict[str, Any]:
  atom_count = len(atom_document.get("atoms", [])) if atom_document else None
  return {
      "schema": "magenta_rt.aig_adg_mlx_render.v1",
      "source": {
          "gesture_packets": str(args.gesture_packets),
          "atom_specs": str(args.atom_specs) if args.atom_specs else None,
          "packet_title": packet_document.get("title"),
          "packet_count": len(packet_document.get("packets", [])),
          "atom_count": atom_count,
      },
      "magenta": {
          "backend": "mlx",
          "model": args.model,
          "checkpoint": args.checkpoint,
          "prompt": args.prompt,
          "bits": args.bits,
          "quantize_group_size": args.quantize_group_size,
          "device": args.device,
          "warmup_steps": args.warmup_steps,
          "temperature": args.temperature,
          "top_k": args.top_k,
          "cfg_musiccoca": args.cfg_musiccoca,
          "cfg_notes": args.cfg_notes,
          "cfg_drums": args.cfg_drums,
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
