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

"""MIDI input commands for the MRT CLI."""

from __future__ import annotations

import json
from pathlib import Path
import time

import click

from magenta_rt import aig_bridge
from magenta_rt import midi_drums
from magenta_rt.cli import main
from magenta_rt.live_ring import DEFAULT_MODEL_FRAME_SIZE
from magenta_rt.live_ring import PacingRingWriter
from magenta_rt.live_ring import SharedMemoryInt16InterleavedStereoRing


DEFAULT_MIDI_RING_PATH = "/tmp/mrt2-midi-drums.ring"
DEFAULT_DRUM_PROMPT = (
    "solo dry acoustic drum kit, no bass, no melody, no chords, no vocals"
)


@main.group()
def midi():
  """MIDI conditioning commands."""


@midi.command("drum-live")
@click.option("--midi", "midi_path", type=click.Path(path_type=Path), required=True)
@click.option("--ring", "ring_path", type=click.Path(path_type=Path), default=DEFAULT_MIDI_RING_PATH)
@click.option("--channel", type=int, default=10)
@click.option("--ring-chunks", type=int, default=96)
@click.option("--chunk-frames", type=click.Choice(["4", "8"]), default="4")
@click.option("--window-seconds", type=float, default=2.0)
@click.option("--model", default="mrt2_small")
@click.option("--checkpoint", default=None)
@click.option("--prompt", default=DEFAULT_DRUM_PROMPT)
@click.option("--temperature", type=float, default=1.1)
@click.option("--top-k", type=int, default=40)
@click.option("--cfg-musiccoca", type=float, default=3.0)
@click.option("--cfg-notes", type=float, default=3.0)
@click.option("--cfg-drums", type=float, default=4.0)
@click.option("--tail-seconds", type=float, default=2.0)
@click.option("--max-frames", type=int, default=None)
@click.option("--trim-leading-silence", is_flag=True, default=False)
@click.option("--metrics-log", type=click.Path(path_type=Path), default=None)
@click.option(
    "--ring-backpressure/--no-ring-backpressure",
    default=True,
    help="Pace live writes to ring free space instead of overwriting old frames.",
)
@click.option("--ring-write-poll-ms", type=float, default=1.0)
@click.option("--ring-write-timeout-seconds", type=float, default=None)
@click.option("--dry-run", is_flag=True, default=False)
def drum_live(
    midi_path,
    ring_path,
    channel,
    ring_chunks,
    chunk_frames,
    window_seconds,
    model,
    checkpoint,
    prompt,
    temperature,
    top_k,
    cfg_musiccoca,
    cfg_notes,
    cfg_drums,
    tail_seconds,
    max_frames,
    trim_leading_silence,
    metrics_log,
    ring_backpressure,
    ring_write_poll_ms,
    ring_write_timeout_seconds,
    dry_run,
):
  """Render a standard drum-map MIDI file to the shared live ring."""

  chunk_frames = int(chunk_frames)
  if ring_chunks <= 0:
    raise click.ClickException("--ring-chunks must be positive")
  if window_seconds <= 0:
    raise click.ClickException("--window-seconds must be positive")

  take = midi_drums.load_midi_drum_take(midi_path, channel=channel)
  schedule = midi_drums.build_conditioning_schedule_from_midi_drums(
      take,
      tail_seconds=tail_seconds,
      max_frames=max_frames,
      trim_leading_silence=trim_leading_silence,
  )
  if dry_run:
    click.echo(
        json.dumps(
            {
                "schema": "magenta_rt.midi_drum_live_dry_run.v1",
                "take": take.summary(),
                "conditioning": _conditioning_summary(schedule),
                "defaults": {
                    "ring": str(ring_path),
                    "chunk_frames": chunk_frames,
                    "window_seconds": window_seconds,
                    "prompt": prompt,
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return

  _run_midi_drum_live(
      take=take,
      schedule=schedule,
      ring_path=ring_path,
      ring_chunks=ring_chunks,
      chunk_frames=chunk_frames,
      window_seconds=window_seconds,
      model=model,
      checkpoint=checkpoint,
      prompt=prompt,
      temperature=temperature,
      top_k=top_k,
      cfg_musiccoca=cfg_musiccoca,
      cfg_notes=cfg_notes,
      cfg_drums=cfg_drums,
      metrics_log=metrics_log,
      ring_backpressure=ring_backpressure,
      ring_write_poll_seconds=ring_write_poll_ms / 1000.0,
      ring_write_timeout_seconds=ring_write_timeout_seconds,
  )


def _run_midi_drum_live(
    *,
    take: midi_drums.MidiDrumTake,
    schedule: aig_bridge.ConditioningSchedule,
    ring_path: Path,
    ring_chunks: int,
    chunk_frames: int,
    window_seconds: float,
    model: str,
    checkpoint: str | None,
    prompt: str,
    temperature: float,
    top_k: int,
    cfg_musiccoca: float,
    cfg_notes: float,
    cfg_drums: float,
    metrics_log: Path | None,
    ring_backpressure: bool,
    ring_write_poll_seconds: float,
    ring_write_timeout_seconds: float | None,
) -> None:
  from magenta_rt import MagentaRT2Jax
  from magenta_rt.jax.live_producer import JaxLiveProducer
  from magenta_rt.jax.system import discretize_cfg
  import jax

  ring = SharedMemoryInt16InterleavedStereoRing(
      ring_path,
      capacity_frames=DEFAULT_MODEL_FRAME_SIZE * chunk_frames * ring_chunks,
      chunk_frames=chunk_frames,
  )
  started = time.time()
  try:
    mrt = MagentaRT2Jax(
        size=model,
        checkpoint=checkpoint,
        temperature=temperature,
        top_k=top_k,
        cfg_musiccoca=cfg_musiccoca,
        cfg_notes=cfg_notes,
        cfg_drums=cfg_drums,
    )
    producer = JaxLiveProducer(mrt, jax_module=jax)
    ring_writer = (
        PacingRingWriter(
            ring,
            poll_seconds=ring_write_poll_seconds,
            timeout_seconds=ring_write_timeout_seconds,
        )
        if ring_backpressure
        else ring.write
    )
    style_tokens = producer.prepare_style_tokens(prompt, use_mapper=True)
    cfgs = [
        discretize_cfg(cfg_musiccoca, 0.2, 40),
        discretize_cfg(cfg_notes, 0.2, 40),
        discretize_cfg(cfg_drums, 1.0, 8),
    ]
    state = producer.init_state()
    total_frames = int(schedule.summary["frames"])
    window_frames = max(
        1, int(round(window_seconds * aig_bridge.MAGENTA_FRAME_RATE))
    )
    click.echo(
        json.dumps(
            {
                "schema": "magenta_rt.midi_drum_live_start.v1",
                "midi": str(take.path),
                "ring": str(ring_path),
                "model": model,
                "devices": [str(device) for device in jax.devices()],
                "conditioning_frames": total_frames,
                "duration_seconds": schedule.summary["duration_seconds"],
                "chunk_frames": chunk_frames,
                "window_frames": window_frames,
            },
            sort_keys=True,
        )
    )

    for window_index, start_frame in enumerate(range(0, total_frames, window_frames)):
      end_frame = min(total_frames, start_frame + window_frames)
      window = midi_drums.window_conditioning_schedule(
          schedule,
          start_frame=start_frame,
          end_frame=end_frame,
      )
      prepared = producer.prepare_schedule(
          window,
          style_tokens=style_tokens,
          cfgs=cfgs,
          temperature=temperature,
          top_k=top_k,
          chunk_frames=chunk_frames,
      )
      samples, state, metrics = producer.generate_int16(
          prepared,
          state=state,
          mode="chunked",
          ring_writer=ring_writer,
      )
      header = ring.header
      record = {
          "schema": "magenta_rt.midi_drum_live_metrics.v1",
          "timestamp": int(time.time()),
          "source": take.summary(),
          "window": {
              "index": window_index,
              "start_frame": start_frame,
              "end_frame": end_frame,
              "frames": end_frame - start_frame,
          },
          "conditioning": _conditioning_summary(window),
          "producer": metrics.to_dict(),
          "ring": {
              "path": str(ring_path),
              "available_frames": ring.available_frames,
              "underrun_frames": header.underrun_frames,
              "overrun_frames": header.overrun_frames,
              "low_water_frames": header.low_water_frames,
              "backpressure": _ring_writer_snapshot(ring_writer),
          },
          "output": {
              "shape": list(samples.shape),
              "dtype": str(samples.dtype),
          },
      }
      if metrics_log is not None:
        metrics_log.parent.mkdir(parents=True, exist_ok=True)
        with metrics_log.open("a") as handle:
          handle.write(json.dumps(record, sort_keys=True) + "\n")
      click.echo(
          "window "
          f"{window_index}: {metrics.steps_per_second:.1f} steps/s, "
          f"RTF {metrics.rtf:.3f}, "
          f"ring available {record['ring']['available_frames']} frames"
      )
  finally:
    ring.close()
  elapsed = time.time() - started
  click.echo(
      json.dumps(
          {
              "schema": "magenta_rt.midi_drum_live_done.v1",
              "elapsed_seconds": elapsed,
              "conditioning_frames": int(schedule.summary["frames"]),
              "duration_seconds": schedule.summary["duration_seconds"],
              "ring": str(ring_path),
          },
          sort_keys=True,
      )
  )


def _conditioning_summary(
    schedule: aig_bridge.ConditioningSchedule,
) -> dict[str, object]:
  summary = dict(schedule.summary)
  summary.pop("packet_reports", None)
  return summary


def _ring_writer_snapshot(writer) -> dict[str, object]:
  if isinstance(writer, PacingRingWriter):
    return {
        "enabled": True,
        "wait_seconds": writer.wait_seconds,
        "wait_count": writer.wait_count,
        "timeout_count": writer.timeout_count,
        "stopped_count": writer.stopped_count,
    }
  return {"enabled": False}
