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

"""Lossy AIG GesturePacket to Magenta RealTime conditioning bridge."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
import tomllib
from typing import Any

import numpy as np


MAGENTA_FRAME_RATE = 25
MAGENTA_SAMPLE_RATE = 48_000
NUM_MIDI_PITCHES = 128

LOSSY_FIELDS = [
    "strength_velocity",
    "body",
    "transient",
    "recovery",
    "presence",
    "density",
    "protect_timing",
    "protect_anchor",
    "phrase_role",
    "variation_seed",
    "variation_group",
    "surface_touch",
    "tuplet",
    "relationship_identity",
    "state_writes",
    "coupling_claims",
]


@dataclasses.dataclass(frozen=True)
class ConditioningSchedule:
  """Frame-wise Magenta notes/drums conditioning derived from AIG packets."""

  notes: np.ndarray
  drums: np.ndarray
  summary: dict[str, Any]
  packet_reports: list[dict[str, Any]]


def load_gesture_packets(path: str | Path) -> dict[str, Any]:
  """Loads an AIG GesturePacket TOML document."""

  with Path(path).open("rb") as handle:
    document = tomllib.load(handle)
  packets = document.get("packets")
  if not isinstance(packets, list):
    raise ValueError(f"{path} does not contain a GesturePacket `packets` list")
  return document


def load_atom_specs(path: str | Path | None) -> dict[str, Any] | None:
  """Loads an optional AIG AtomSpec TOML document."""

  if path is None:
    return None
  with Path(path).open("rb") as handle:
    document = tomllib.load(handle)
  atoms = document.get("atoms")
  if not isinstance(atoms, list):
    raise ValueError(f"{path} does not contain an AtomSpec `atoms` list")
  return document


def load_pc4ms_adg_events(path: str | Path) -> list[dict[str, Any]]:
  """Loads a PC4MS ``*.adg-events.json`` list."""

  path = Path(path)
  payload = json.loads(path.read_text())
  if not isinstance(payload, list):
    raise ValueError(f"{path} must contain a JSON event list")
  for index, event in enumerate(payload):
    if not isinstance(event, dict):
      raise ValueError(f"{path} event {index} must be an object")
  return payload


def load_pc4ms_adg_timeline(path: str | Path) -> dict[str, Any]:
  """Loads timing metadata from a PC4MS ``*.adg.toml`` document."""

  with Path(path).open("rb") as handle:
    document = tomllib.load(handle)
  timeline = document.get("timeline")
  if not isinstance(timeline, dict):
    raise ValueError(f"{path} does not contain a PC4MS ADG `timeline` table")
  return {
      "tempo_bpm": float(timeline.get("tempo_bpm", 120.0)),
      "ppqn": int(timeline.get("ppqn", 480)),
      "output_channel": int(timeline.get("output_channel", 9)),
      "title": str(document.get("aig", {}).get("title", "")),
      "profile_id": str(document.get("aig", {}).get("profile_id", "")),
  }


def build_conditioning_schedule(
    packet_document: dict[str, Any],
    *,
    frame_rate: int = MAGENTA_FRAME_RATE,
    output_sample_rate: int = MAGENTA_SAMPLE_RATE,
    tail_seconds: float = 2.0,
    max_frames: int | None = None,
) -> ConditioningSchedule:
  """Converts AIG GesturePackets to frame-wise MRT2 note/drum states.

  The conversion is intentionally lossy. MRT2's public conditioning shape only
  exposes 128 MIDI pitch states plus a single drum/no-drum lane, so ADG ports
  that cannot influence this state are recorded in the summary instead of being
  hidden.
  """

  packets = list(packet_document.get("packets", []))
  packet_sample_rate = int(
      packet_document.get("sample_rate_hz") or output_sample_rate
  )
  frame_samples = output_sample_rate / frame_rate
  max_source_frame = 0
  for packet in packets:
    start = int(packet.get("start_frame", 0))
    duration = max(1, int(packet.get("duration_frames", 1)))
    max_source_frame = max(max_source_frame, start + duration)

  source_seconds = max_source_frame / packet_sample_rate
  total_frames = max(1, math.ceil((source_seconds + tail_seconds) * frame_rate))
  if max_frames is not None:
    total_frames = min(total_frames, max_frames)

  notes = np.zeros((total_frames, NUM_MIDI_PITCHES), dtype=np.int8)
  drums = np.zeros((total_frames, 1), dtype=np.int8)
  packet_reports = []
  role_counts: dict[str, int] = {}
  pitch_counts: dict[str, int] = {}
  quantization_errors = []
  mapped_packets = 0

  for packet in packets:
    pitch = pitch_for_packet(packet)
    role = _clean(packet.get("role"))
    role_counts[role] = role_counts.get(role, 0) + 1
    if pitch is None:
      packet_reports.append(_packet_report(packet, None, None, None, 0.0))
      continue

    start_sample = int(packet.get("start_frame", 0))
    duration_samples = max(1, int(packet.get("duration_frames", 1)))
    start_seconds = start_sample / packet_sample_rate
    end_seconds = (start_sample + duration_samples) / packet_sample_rate
    start_frame = int(round(start_seconds * frame_rate))
    end_frame = max(start_frame + 1, int(math.ceil(end_seconds * frame_rate)))
    if start_frame >= total_frames:
      continue
    end_frame = min(end_frame, total_frames)

    quantized_sample = round(start_frame * frame_samples)
    quantization_error = abs(start_sample - quantized_sample) / packet_sample_rate
    quantization_errors.append(quantization_error)

    notes[start_frame, pitch] = 2
    if start_frame + 1 < end_frame:
      sustain = notes[start_frame + 1 : end_frame, pitch]
      sustain[sustain == 0] = 1
    drums[start_frame:end_frame, 0] = 1

    mapped_packets += 1
    pitch_key = str(pitch)
    pitch_counts[pitch_key] = pitch_counts.get(pitch_key, 0) + 1
    packet_reports.append(
        _packet_report(packet, pitch, start_frame, end_frame, quantization_error)
    )

  active_frames = int(np.count_nonzero(drums[:, 0]))
  summary = {
      "schema": "magenta_rt.aig_conditioning.v1",
      "packet_count": len(packets),
      "mapped_packet_count": mapped_packets,
      "frame_rate": frame_rate,
      "sample_rate_hz": output_sample_rate,
      "source_sample_rate_hz": packet_sample_rate,
      "frames": int(total_frames),
      "duration_seconds": total_frames / frame_rate,
      "active_drum_frames": active_frames,
      "role_counts": role_counts,
      "pitch_counts": pitch_counts,
      "lossy_fields": LOSSY_FIELDS,
      "quantization": {
          "mean_error_ms": _mean_ms(quantization_errors),
          "max_error_ms": _max_ms(quantization_errors),
          "count": len(quantization_errors),
      },
  }
  return ConditioningSchedule(
      notes=notes,
      drums=drums,
      summary=summary,
      packet_reports=packet_reports,
  )


def build_conditioning_schedule_from_pc4ms_adg_events(
    events: list[dict[str, Any]],
    *,
    tempo_bpm: float,
    ppqn: int,
    frame_rate: int = MAGENTA_FRAME_RATE,
    output_sample_rate: int = MAGENTA_SAMPLE_RATE,
    tail_seconds: float = 2.0,
    max_frames: int | None = None,
) -> ConditioningSchedule:
  """Converts a full PC4MS ADG event list to frame-wise MRT2 conditioning.

  This path consumes existing full ADG takes. It does not ask PC4MS to generate
  new ADG and it deliberately ignores older MIDI-only takes.
  """

  if tempo_bpm <= 0:
    raise ValueError("tempo_bpm must be positive")
  if ppqn <= 0:
    raise ValueError("ppqn must be positive")

  tick_seconds = 60.0 / (tempo_bpm * ppqn)
  max_end_tick = 0
  for event in events:
    tick = int(event.get("tick", 0)) + int(event.get("micro_offset_ticks", 0))
    duration = max(1, int(event.get("duration_ticks", max(1, ppqn // 4))))
    max_end_tick = max(max_end_tick, tick + duration)

  source_seconds = max(0.0, max_end_tick * tick_seconds)
  total_frames = max(1, math.ceil((source_seconds + tail_seconds) * frame_rate))
  if max_frames is not None:
    total_frames = min(total_frames, max_frames)

  notes = np.zeros((total_frames, NUM_MIDI_PITCHES), dtype=np.int8)
  drums = np.zeros((total_frames, 1), dtype=np.int8)
  packet_reports = []
  role_counts: dict[str, int] = {}
  pitch_counts: dict[str, int] = {}
  quantization_errors = []
  mapped_events = 0

  for event in events:
    role = _clean(event.get("role"))
    role_counts[role] = role_counts.get(role, 0) + 1
    packet_like = _adg_event_as_packet(event)
    pitch = pitch_for_packet(packet_like)
    if pitch is None:
      packet_reports.append(_adg_event_report(event, None, None, None, 0.0))
      continue

    start_tick = int(event.get("tick", 0)) + int(
        event.get("micro_offset_ticks", 0)
    )
    duration_ticks = max(1, int(event.get("duration_ticks", max(1, ppqn // 4))))
    start_seconds = max(0.0, start_tick * tick_seconds)
    end_seconds = max(
        start_seconds + tick_seconds,
        (start_tick + duration_ticks) * tick_seconds,
    )
    start_frame = int(round(start_seconds * frame_rate))
    end_frame = max(start_frame + 1, int(math.ceil(end_seconds * frame_rate)))
    if start_frame >= total_frames:
      continue
    end_frame = min(end_frame, total_frames)

    quantized_tick = round(start_frame / frame_rate / tick_seconds)
    quantization_error = abs(start_tick - quantized_tick) * tick_seconds
    quantization_errors.append(quantization_error)

    notes[start_frame, pitch] = 2
    if start_frame + 1 < end_frame:
      sustain = notes[start_frame + 1 : end_frame, pitch]
      sustain[sustain == 0] = 1
    drums[start_frame:end_frame, 0] = 1

    mapped_events += 1
    pitch_key = str(pitch)
    pitch_counts[pitch_key] = pitch_counts.get(pitch_key, 0) + 1
    packet_reports.append(
        _adg_event_report(event, pitch, start_frame, end_frame, quantization_error)
    )

  summary = {
      "schema": "magenta_rt.pc4ms_adg_conditioning.v1",
      "event_count": len(events),
      "mapped_event_count": mapped_events,
      "frame_rate": frame_rate,
      "sample_rate_hz": output_sample_rate,
      "tempo_bpm": float(tempo_bpm),
      "ppqn": int(ppqn),
      "frames": int(total_frames),
      "duration_seconds": total_frames / frame_rate,
      "active_drum_frames": int(np.count_nonzero(drums[:, 0])),
      "role_counts": role_counts,
      "pitch_counts": pitch_counts,
      "lossy_fields": LOSSY_FIELDS,
      "quantization": {
          "mean_error_ms": _mean_ms(quantization_errors),
          "max_error_ms": _max_ms(quantization_errors),
          "count": len(quantization_errors),
      },
  }
  return ConditioningSchedule(
      notes=notes,
      drums=drums,
      summary=summary,
      packet_reports=packet_reports,
  )


def window_conditioning_schedule(
    schedule: ConditioningSchedule,
    *,
    start_frame: int = 0,
    max_frames: int | None = None,
) -> ConditioningSchedule:
  """Returns a render window from a full conditioning schedule."""

  if start_frame < 0:
    raise ValueError("start_frame must be non-negative")
  source_frames = int(schedule.summary["frames"])
  if start_frame >= source_frames:
    raise ValueError(
        f"start_frame {start_frame} is outside the {source_frames}-frame schedule"
    )
  if max_frames is not None and max_frames <= 0:
    raise ValueError("max_frames must be positive")

  end_frame = source_frames if max_frames is None else start_frame + max_frames
  end_frame = min(source_frames, end_frame)
  notes = schedule.notes[start_frame:end_frame].copy()
  drums = schedule.drums[start_frame:end_frame].copy()

  frame_rate = int(schedule.summary["frame_rate"])
  window_frames = int(notes.shape[0])
  overlapping_packets = _count_overlapping_packets(
      schedule.packet_reports, start_frame, end_frame
  )
  summary = {
      **schedule.summary,
      "frames": window_frames,
      "duration_seconds": window_frames / frame_rate,
      "active_drum_frames": int(np.count_nonzero(drums[:, 0])),
      "window": {
          "start_frame": int(start_frame),
          "end_frame": int(end_frame),
          "source_frames": source_frames,
          "start_seconds": start_frame / frame_rate,
          "end_seconds": end_frame / frame_rate,
          "source_duration_seconds": schedule.summary["duration_seconds"],
          "overlapping_packet_count": overlapping_packets,
      },
  }
  return ConditioningSchedule(
      notes=notes,
      drums=drums,
      summary=summary,
      packet_reports=schedule.packet_reports,
  )


def pitch_for_packet(packet: dict[str, Any]) -> int | None:
  """Maps an AIG packet's role/kind/ports to a GM drum pitch."""

  role = _clean(packet.get("role"))
  kind = _clean(packet.get("gesture_kind"))
  ports = _merged_ports(packet)

  if role == "kick":
    return 36
  if role == "snare":
    return 38
  if role == "hat":
    choke = _port(ports, "choke", "choke_force", "adg.choke")
    openness = _port(ports, "openness", "air_opening", "adg.openness")
    wash = _port(ports, "wash", "tail_credit", "adg.wash")
    if kind == "choke" or choke >= 0.35:
      return 44
    if kind == "open" or openness >= 0.35 or wash >= 0.35:
      return 46
    return 42
  if role == "ride":
    return 51
  if role == "ride_bell":
    return 53
  if role == "crash":
    return 49
  if role == "tom":
    if kind == "flash":
      return 50
    if kind == "pressure":
      return 47
    return 45
  return None


def audio_stats(samples: np.ndarray, sample_rate: int) -> dict[str, Any]:
  """Computes simple validation metrics for a rendered waveform."""

  finite = bool(np.isfinite(samples).all())
  peak = float(np.max(np.abs(samples))) if samples.size else 0.0
  rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
  return {
      "shape": list(samples.shape),
      "sample_rate_hz": int(sample_rate),
      "duration_seconds": float(samples.shape[0] / sample_rate),
      "finite": finite,
      "non_silent": bool(peak > 0.0),
      "peak": peak,
      "rms": rms,
  }


def _packet_report(
    packet: dict[str, Any],
    pitch: int | None,
    start_frame: int | None,
    end_frame: int | None,
    quantization_error: float,
) -> dict[str, Any]:
  return {
      "packet_id": packet.get("id", ""),
      "source_event_id": _source_event_id(packet),
      "role": _clean(packet.get("role")),
      "gesture_kind": _clean(packet.get("gesture_kind")),
      "pitch": pitch,
      "start_frame": start_frame,
      "end_frame": end_frame,
      "quantization_error_ms": quantization_error * 1000.0,
  }


def _adg_event_as_packet(event: dict[str, Any]) -> dict[str, Any]:
  ports = {
      "openness": float(event.get("openness", 0.0)),
      "density": float(event.get("density", 0.0)),
      "body": float(event.get("body", 0.0)),
      "transient": float(event.get("transient", 0.0)),
  }
  return {
      "id": event.get("id", ""),
      "source_event_id": event.get("id", ""),
      "role": event.get("role"),
      "gesture_kind": event.get("kind"),
      "common_ports": ports,
      "dialect_ports": ports,
  }


def _adg_event_report(
    event: dict[str, Any],
    pitch: int | None,
    start_frame: int | None,
    end_frame: int | None,
    quantization_error: float,
) -> dict[str, Any]:
  return {
      "packet_id": event.get("id", ""),
      "source_event_id": event.get("id", ""),
      "role": _clean(event.get("role")),
      "gesture_kind": _clean(event.get("kind")),
      "pitch": pitch,
      "start_frame": start_frame,
      "end_frame": end_frame,
      "quantization_error_ms": quantization_error * 1000.0,
  }


def _count_overlapping_packets(
    packet_reports: list[dict[str, Any]],
    start_frame: int,
    end_frame: int,
) -> int:
  count = 0
  for report in packet_reports:
    packet_start = report.get("start_frame")
    packet_end = report.get("end_frame")
    if packet_start is None or packet_end is None:
      continue
    if int(packet_start) < end_frame and int(packet_end) > start_frame:
      count += 1
  return count


def _source_event_id(packet: dict[str, Any]) -> str:
  value = packet.get("source_event_id", "")
  if isinstance(value, dict):
    return str(value.get("0", ""))
  return str(value)


def _merged_ports(packet: dict[str, Any]) -> dict[str, float]:
  ports = {}
  for key in ("common_ports", "dialect_ports", "state_writes"):
    values = packet.get(key) or {}
    if isinstance(values, dict):
      for name, value in values.items():
        if isinstance(value, (int, float)):
          ports[str(name)] = float(value)
          if key == "dialect_ports":
            ports[f"adg.{name}"] = float(value)
  return ports


def _port(ports: dict[str, float], *names: str) -> float:
  for name in names:
    value = ports.get(name)
    if value is not None:
      return float(value)
  return 0.0


def _clean(value: Any) -> str:
  return str(value or "").strip().lower()


def _mean_ms(values: list[float]) -> float:
  if not values:
    return 0.0
  return float(np.mean(np.array(values, dtype=np.float64)) * 1000.0)


def _max_ms(values: list[float]) -> float:
  if not values:
    return 0.0
  return float(max(values) * 1000.0)
