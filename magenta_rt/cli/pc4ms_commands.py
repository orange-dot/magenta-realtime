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

"""PC4MS integration commands for the MRT CLI."""

from __future__ import annotations

from pathlib import Path
import json

import click

from magenta_rt.cli import main


@main.group()
def pc4ms():
  """PC4 Microkit Studio integration commands."""


@pc4ms.command("live-daemon")
@click.option("--socket", "socket_path", type=click.Path(path_type=Path), default="/tmp/pc4ms-magenta-live.sock")
@click.option("--ring", "ring_path", type=click.Path(path_type=Path), default="/tmp/mrt2-pc4ms-live.ring")
@click.option("--ring-chunks", type=int, default=4)
@click.option("--chunk-frames", type=click.Choice(["4", "8"]), default="8")
@click.option("--model", default="mrt2_small")
@click.option("--checkpoint", default=None)
@click.option("--prompt", default="tight acoustic funk drums")
@click.option("--temperature", type=float, default=1.1)
@click.option("--top-k", type=int, default=40)
@click.option("--cfg-musiccoca", type=float, default=3.0)
@click.option("--cfg-notes", type=float, default=3.0)
@click.option("--cfg-drums", type=float, default=4.0)
@click.option("--max-frames", type=int, default=None)
@click.option("--tail-seconds", type=float, default=0.0)
@click.option("--metrics-log", type=click.Path(path_type=Path), default=None)
@click.option(
    "--ring-backpressure/--no-ring-backpressure",
    default=True,
    help="Pace live writes to ring free space instead of overwriting old frames.",
)
@click.option("--ring-write-poll-ms", type=float, default=1.0)
@click.option("--ring-write-timeout-seconds", type=float, default=None)
def live_daemon(
    socket_path,
    ring_path,
    ring_chunks,
    chunk_frames,
    model,
    checkpoint,
    prompt,
    temperature,
    top_k,
    cfg_musiccoca,
    cfg_notes,
    cfg_drums,
    max_frames,
    tail_seconds,
    metrics_log,
    ring_backpressure,
    ring_write_poll_ms,
    ring_write_timeout_seconds,
):
  """Receive PC4MS ADG chunks and write MRT2 audio to the shared ring."""

  from argparse import Namespace
  from magenta_rt.pc4ms_live import run_daemon

  run_daemon(
      Namespace(
          socket=socket_path,
          ring=ring_path,
          ring_chunks=ring_chunks,
          chunk_frames=int(chunk_frames),
          model=model,
          checkpoint=checkpoint,
          prompt=prompt,
          temperature=temperature,
          top_k=top_k,
          cfg_musiccoca=cfg_musiccoca,
          cfg_notes=cfg_notes,
          cfg_drums=cfg_drums,
          max_frames=max_frames,
          tail_seconds=tail_seconds,
          producer_mode="chunked",
          metrics_log=metrics_log,
          ring_backpressure=ring_backpressure,
          ring_write_poll_ms=ring_write_poll_ms,
          ring_write_timeout_seconds=ring_write_timeout_seconds,
      )
  )


@pc4ms.command("replay-live-chunks")
@click.option("--socket", "socket_path", type=click.Path(path_type=Path), default="/tmp/pc4ms-magenta-live.sock")
@click.option(
    "--dataset-root",
    type=click.Path(path_type=Path),
    default=(
        "/home/dev/work-base-20260421/workspace/systems/pc4-microkit-studio/"
        ".pc4ms/session-store/workbench-session/generated-drum-midi-takes"
    ),
)
@click.option("--take-id", default=None)
@click.option("--adg-bundle-manifest", type=click.Path(path_type=Path), default=None)
@click.option("--duration-seconds", type=float, default=None)
@click.option("--max-chunks", type=int, default=None)
@click.option("--pace", type=click.Choice(["realtime", "fast"]), default="realtime")
@click.option("--prime-chunks", type=int, default=0)
@click.option("--timeout-seconds", type=float, default=30.0)
@click.option("--log", "log_path", type=click.Path(path_type=Path), default=None)
def replay_live_chunks(
    socket_path,
    dataset_root,
    take_id,
    adg_bundle_manifest,
    duration_seconds,
    max_chunks,
    pace,
    prime_chunks,
    timeout_seconds,
    log_path,
):
  """Replay saved PC4MS live ADG chunks into the Magenta daemon socket."""

  from magenta_rt import adg_dataset
  from magenta_rt import aig_bridge
  from magenta_rt.pc4ms_live import live_payload_from_debug_chunk
  from magenta_rt.pc4ms_live import send_live_payloads

  if adg_bundle_manifest is not None:
    take = adg_dataset.load_full_adg_take(adg_bundle_manifest)
  else:
    take = adg_dataset.select_full_adg_take(dataset_root, take_id=take_id)
  if take.live_chunks is None:
    raise click.ClickException(f"take has no live_chunks debug file: {take.take_id}")
  timeline = aig_bridge.load_pc4ms_adg_timeline(take.adg_toml)
  chunks = json.loads(take.live_chunks.read_text())
  if not isinstance(chunks, list):
    raise click.ClickException(f"live_chunks must contain a JSON list: {take.live_chunks}")

  payloads = []
  audio_seconds = 0.0
  for chunk in chunks:
    if not isinstance(chunk, dict):
      continue
    if max_chunks is not None and len(payloads) >= max_chunks:
      break
    if duration_seconds is not None and payloads and audio_seconds >= duration_seconds:
      break
    payload = live_payload_from_debug_chunk(
        take_id=take.take_id,
        tempo_bpm=float(timeline["tempo_bpm"]),
        ppqn=int(timeline["ppqn"]),
        profile_id=str(timeline.get("profile_id", "")),
        debug_chunk=chunk,
    )
    payloads.append(payload)
    timing = payload["timing"]
    audio_seconds += (
        max(0, timing["output_end_tick"] - timing["output_start_tick"])
        * 60.0
        / (float(timeline["tempo_bpm"]) * int(timeline["ppqn"]))
    )

  record = send_live_payloads(
      socket_path=socket_path,
      payloads=payloads,
      pace_realtime=(pace == "realtime"),
      prime_chunks=prime_chunks,
      timeout_seconds=timeout_seconds,
  )
  record["take"] = {
      "take_id": take.take_id,
      "manifest": str(take.manifest),
      "live_chunks": str(take.live_chunks),
  }
  if log_path is not None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as handle:
      handle.write(json.dumps(record, sort_keys=True) + "\n")
  click.echo(json.dumps(record, sort_keys=True))
