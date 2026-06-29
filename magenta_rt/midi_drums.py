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

"""Standard MIDI drum-map conditioning for Magenta RT live preview."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Any

import numpy as np

from magenta_rt import aig_bridge


DEFAULT_DRUM_CHANNEL = 10
DEFAULT_TEMPO_MICROS_PER_QUARTER = 500_000


class MidiDrumError(ValueError):
  """A MIDI file cannot be interpreted as a PPQN drum take."""


@dataclasses.dataclass(frozen=True)
class MidiTempoEvent:
  tick: int
  tempo_micros_per_quarter: int


@dataclasses.dataclass(frozen=True)
class MidiDrumNote:
  start_tick: int
  end_tick: int
  pitch: int
  velocity: int
  channel: int
  track_index: int

  @property
  def duration_ticks(self) -> int:
    return max(1, self.end_tick - self.start_tick)


@dataclasses.dataclass(frozen=True)
class MidiDrumTake:
  path: Path
  midi_format: int
  ticks_per_beat: int
  selected_channel: int
  tempos: tuple[MidiTempoEvent, ...]
  notes: tuple[MidiDrumNote, ...]
  channel_event_counts: dict[int, int]
  track_names: tuple[str, ...]
  last_tick: int

  @property
  def first_note_tick(self) -> int:
    if not self.notes:
      return 0
    return min(note.start_tick for note in self.notes)

  @property
  def duration_seconds(self) -> float:
    return tick_to_seconds(self.last_tick, self.tempos, self.ticks_per_beat)

  def summary(self) -> dict[str, Any]:
    pitch_counts: dict[str, int] = {}
    for note in self.notes:
      key = str(note.pitch)
      pitch_counts[key] = pitch_counts.get(key, 0) + 1
    return {
        "schema": "magenta_rt.midi_drum_take.v1",
        "path": str(self.path),
        "midi_format": self.midi_format,
        "ticks_per_beat": self.ticks_per_beat,
        "selected_channel": self.selected_channel,
        "tempo_events": [
            dataclasses.asdict(tempo) for tempo in self.tempos
        ],
        "tempo_bpm_initial": tempo_to_bpm(
            self.tempos[0].tempo_micros_per_quarter
        ),
        "note_count": len(self.notes),
        "first_note_tick": self.first_note_tick,
        "last_tick": self.last_tick,
        "duration_seconds": self.duration_seconds,
        "pitch_range": (
            None
            if not self.notes
            else [
                min(note.pitch for note in self.notes),
                max(note.pitch for note in self.notes),
            ]
        ),
        "pitch_counts": pitch_counts,
        "channel_event_counts": {
            str(channel): count
            for channel, count in sorted(self.channel_event_counts.items())
        },
        "track_names": list(self.track_names),
    }


def load_midi_drum_take(
    path: str | Path, *, channel: int = DEFAULT_DRUM_CHANNEL
) -> MidiDrumTake:
  """Loads a PPQN SMF file and keeps only one human-numbered MIDI channel."""

  path = Path(path)
  data = path.read_bytes()
  reader = _MidiReader(data)
  if reader.read_bytes(4) != b"MThd":
    raise MidiDrumError(f"{path} is not a Standard MIDI File")
  header_size = reader.read_u32()
  if header_size < 6:
    raise MidiDrumError("MIDI header is shorter than 6 bytes")
  header = reader.read_bytes(header_size)
  midi_format = int.from_bytes(header[0:2], "big")
  track_count = int.from_bytes(header[2:4], "big")
  division = int.from_bytes(header[4:6], "big")
  if midi_format not in (0, 1):
    raise MidiDrumError(f"unsupported MIDI format {midi_format}; expected 0 or 1")
  if division & 0x8000:
    raise MidiDrumError("SMPTE time-division MIDI is not supported")
  ticks_per_beat = division
  selected_channel = _normalize_channel(channel)

  tempos: list[MidiTempoEvent] = []
  notes: list[MidiDrumNote] = []
  channel_event_counts: dict[int, int] = {}
  track_names: list[str] = []
  last_tick = 0
  for track_index in range(track_count):
    track = _read_track(reader, path)
    parsed = _parse_track(
        track,
        track_index=track_index,
        selected_channel=selected_channel,
    )
    tempos.extend(parsed["tempos"])
    notes.extend(parsed["notes"])
    track_names.extend(parsed["track_names"])
    for channel_key, count in parsed["channel_event_counts"].items():
      channel_event_counts[channel_key] = (
          channel_event_counts.get(channel_key, 0) + count
      )
    last_tick = max(last_tick, parsed["last_tick"])

  tempos = _normalize_tempos(tempos)
  notes.sort(key=lambda note: (note.start_tick, note.pitch, note.track_index))
  if notes:
    last_tick = max(last_tick, max(note.end_tick for note in notes))
  return MidiDrumTake(
      path=path,
      midi_format=midi_format,
      ticks_per_beat=ticks_per_beat,
      selected_channel=selected_channel,
      tempos=tuple(tempos),
      notes=tuple(notes),
      channel_event_counts=channel_event_counts,
      track_names=tuple(track_names),
      last_tick=last_tick,
  )


def build_conditioning_schedule_from_midi_drums(
    take: MidiDrumTake,
    *,
    frame_rate: int = aig_bridge.MAGENTA_FRAME_RATE,
    output_sample_rate: int = aig_bridge.MAGENTA_SAMPLE_RATE,
    tail_seconds: float = 2.0,
    max_frames: int | None = None,
    trim_leading_silence: bool = False,
) -> aig_bridge.ConditioningSchedule:
  """Converts a MIDI drum take to MRT2 note/drum conditioning lanes."""

  base_seconds = (
      tick_to_seconds(take.first_note_tick, take.tempos, take.ticks_per_beat)
      if trim_leading_silence
      else 0.0
  )
  source_seconds = max(
      0.0,
      tick_to_seconds(take.last_tick, take.tempos, take.ticks_per_beat)
      - base_seconds,
  )
  total_frames = max(1, math.ceil((source_seconds + tail_seconds) * frame_rate))
  if max_frames is not None:
    total_frames = min(total_frames, max_frames)

  notes = np.zeros(
      (total_frames, aig_bridge.NUM_MIDI_PITCHES), dtype=np.int8
  )
  drums = np.zeros((total_frames, 1), dtype=np.int8)
  packet_reports = []
  pitch_counts: dict[str, int] = {}
  quantization_errors = []
  mapped_notes = 0

  for index, note in enumerate(take.notes):
    if note.pitch < 0 or note.pitch >= aig_bridge.NUM_MIDI_PITCHES:
      continue
    start_seconds = (
        tick_to_seconds(note.start_tick, take.tempos, take.ticks_per_beat)
        - base_seconds
    )
    end_seconds = (
        tick_to_seconds(note.end_tick, take.tempos, take.ticks_per_beat)
        - base_seconds
    )
    start_seconds = max(0.0, start_seconds)
    end_seconds = max(start_seconds + _tick_seconds_at_note(take, note), end_seconds)
    start_frame = int(round(start_seconds * frame_rate))
    end_frame = max(start_frame + 1, int(math.ceil(end_seconds * frame_rate)))
    if start_frame >= total_frames:
      continue
    end_frame = min(end_frame, total_frames)

    quantized_seconds = start_frame / frame_rate
    quantization_errors.append(abs(start_seconds - quantized_seconds))
    notes[start_frame, note.pitch] = 2
    if start_frame + 1 < end_frame:
      sustain = notes[start_frame + 1 : end_frame, note.pitch]
      sustain[sustain == 0] = 1
    drums[start_frame:end_frame, 0] = 1
    mapped_notes += 1
    key = str(note.pitch)
    pitch_counts[key] = pitch_counts.get(key, 0) + 1
    packet_reports.append(
        {
            "source_note_index": index,
            "pitch": note.pitch,
            "velocity": note.velocity,
            "channel": note.channel,
            "track_index": note.track_index,
            "start_tick": note.start_tick,
            "end_tick": note.end_tick,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "quantization_error_ms": quantization_errors[-1] * 1000.0,
        }
    )

  summary = {
      "schema": "magenta_rt.midi_drum_conditioning.v1",
      "source": take.summary(),
      "note_count": len(take.notes),
      "mapped_note_count": mapped_notes,
      "frame_rate": int(frame_rate),
      "sample_rate_hz": int(output_sample_rate),
      "ticks_per_beat": int(take.ticks_per_beat),
      "selected_channel": int(take.selected_channel),
      "frames": int(total_frames),
      "duration_seconds": total_frames / frame_rate,
      "trim_leading_silence": bool(trim_leading_silence),
      "active_drum_frames": int(np.count_nonzero(drums[:, 0])),
      "pitch_counts": pitch_counts,
      "quantization": {
          "mean_error_ms": _mean_ms(quantization_errors),
          "max_error_ms": _max_ms(quantization_errors),
          "count": len(quantization_errors),
      },
  }
  return aig_bridge.ConditioningSchedule(
      notes=notes,
      drums=drums,
      summary=summary,
      packet_reports=packet_reports,
  )


def window_conditioning_schedule(
    schedule: aig_bridge.ConditioningSchedule,
    *,
    start_frame: int,
    end_frame: int,
) -> aig_bridge.ConditioningSchedule:
  """Returns a compact window while preserving source summary metadata."""

  if start_frame < 0 or end_frame <= start_frame:
    raise ValueError("invalid conditioning window")
  source_frames = int(schedule.summary["frames"])
  end_frame = min(end_frame, source_frames)
  notes = schedule.notes[start_frame:end_frame].copy()
  drums = schedule.drums[start_frame:end_frame].copy()
  frame_rate = int(schedule.summary["frame_rate"])
  reports = [
      report
      for report in schedule.packet_reports
      if _overlaps(
          int(report.get("start_frame", -1)),
          int(report.get("end_frame", -1)),
          start_frame,
          end_frame,
      )
  ]
  summary = {
      **schedule.summary,
      "frames": int(notes.shape[0]),
      "duration_seconds": int(notes.shape[0]) / frame_rate,
      "window_start_frame": int(start_frame),
      "window_end_frame": int(end_frame),
      "window_source_frames": source_frames,
      "active_drum_frames": int(np.count_nonzero(drums[:, 0])),
  }
  return aig_bridge.ConditioningSchedule(
      notes=notes,
      drums=drums,
      summary=summary,
      packet_reports=reports,
  )


def tick_to_seconds(
    tick: int, tempos: tuple[MidiTempoEvent, ...], ticks_per_beat: int
) -> float:
  """Converts an absolute MIDI tick to seconds using a tempo map."""

  tick = max(0, int(tick))
  tempos = _normalize_tempos(list(tempos))
  elapsed = 0.0
  current_tick = 0
  current_tempo = tempos[0].tempo_micros_per_quarter
  for event in tempos[1:]:
    if event.tick >= tick:
      break
    delta = max(0, event.tick - current_tick)
    elapsed += delta * current_tempo / 1_000_000.0 / ticks_per_beat
    current_tick = event.tick
    current_tempo = event.tempo_micros_per_quarter
  elapsed += max(0, tick - current_tick) * current_tempo / 1_000_000.0 / ticks_per_beat
  return elapsed


def tempo_to_bpm(tempo_micros_per_quarter: int) -> float:
  return 60_000_000.0 / tempo_micros_per_quarter


def _read_track(reader: "_MidiReader", path: Path) -> bytes:
  if reader.read_bytes(4) != b"MTrk":
    raise MidiDrumError(f"{path} has an invalid MIDI track chunk")
  return reader.read_bytes(reader.read_u32())


def _parse_track(
    data: bytes, *, track_index: int, selected_channel: int
) -> dict[str, Any]:
  reader = _MidiReader(data)
  running_status: int | None = None
  tick = 0
  tempos: list[MidiTempoEvent] = []
  notes: list[MidiDrumNote] = []
  active: dict[tuple[int, int], list[tuple[int, int]]] = {}
  channel_event_counts: dict[int, int] = {}
  track_names: list[str] = []
  while not reader.eof:
    delta = reader.read_vlq()
    tick += delta
    status = reader.peek_u8()
    if status < 0x80:
      if running_status is None:
        raise MidiDrumError("MIDI running status encountered before status byte")
      status = running_status
    else:
      status = reader.read_u8()
      if status < 0xF0:
        running_status = status

    if status == 0xFF:
      meta_type = reader.read_u8()
      payload = reader.read_bytes(reader.read_vlq())
      if meta_type == 0x51 and len(payload) == 3:
        tempos.append(MidiTempoEvent(tick, int.from_bytes(payload, "big")))
      elif meta_type in (0x03, 0x04):
        track_names.append(payload.decode("utf-8", errors="replace"))
      continue
    if status in (0xF0, 0xF7):
      reader.read_bytes(reader.read_vlq())
      continue

    event_type = status & 0xF0
    channel = (status & 0x0F) + 1
    if event_type in (0x80, 0x90, 0xA0, 0xB0, 0xE0):
      first = reader.read_u8()
      second = reader.read_u8()
      if event_type in (0x80, 0x90):
        channel_event_counts[channel] = channel_event_counts.get(channel, 0) + 1
        if channel == selected_channel:
          _handle_note_event(
              notes=notes,
              active=active,
              event_type=event_type,
              tick=tick,
              channel=channel,
              pitch=first,
              velocity=second,
              track_index=track_index,
          )
    elif event_type in (0xC0, 0xD0):
      reader.read_u8()
    else:
      raise MidiDrumError(f"unsupported MIDI status 0x{status:02x}")

  for (channel, pitch), starts in active.items():
    for start_tick, velocity in starts:
      notes.append(
          MidiDrumNote(
              start_tick=start_tick,
              end_tick=max(start_tick + 1, tick),
              pitch=pitch,
              velocity=velocity,
              channel=channel,
              track_index=track_index,
          )
      )
  return {
      "tempos": tempos,
      "notes": notes,
      "channel_event_counts": channel_event_counts,
      "track_names": track_names,
      "last_tick": tick,
  }


def _handle_note_event(
    *,
    notes: list[MidiDrumNote],
    active: dict[tuple[int, int], list[tuple[int, int]]],
    event_type: int,
    tick: int,
    channel: int,
    pitch: int,
    velocity: int,
    track_index: int,
) -> None:
  key = (channel, pitch)
  if event_type == 0x90 and velocity > 0:
    active.setdefault(key, []).append((tick, velocity))
    return
  starts = active.get(key)
  if not starts:
    return
  start_tick, start_velocity = starts.pop(0)
  notes.append(
      MidiDrumNote(
          start_tick=start_tick,
          end_tick=max(start_tick + 1, tick),
          pitch=pitch,
          velocity=start_velocity,
          channel=channel,
          track_index=track_index,
      )
  )


def _normalize_tempos(tempos: list[MidiTempoEvent]) -> tuple[MidiTempoEvent, ...]:
  ordered = sorted(tempos, key=lambda tempo: tempo.tick)
  if not ordered or ordered[0].tick != 0:
    ordered.insert(0, MidiTempoEvent(0, DEFAULT_TEMPO_MICROS_PER_QUARTER))
  compact: list[MidiTempoEvent] = []
  for tempo in ordered:
    if compact and compact[-1].tick == tempo.tick:
      compact[-1] = tempo
    else:
      compact.append(tempo)
  return tuple(compact)


def _normalize_channel(channel: int) -> int:
  channel = int(channel)
  if channel < 1 or channel > 16:
    raise MidiDrumError("MIDI channel must be in the human range 1..16")
  return channel


def _tick_seconds_at_note(take: MidiDrumTake, note: MidiDrumNote) -> float:
  return (
      tick_to_seconds(note.start_tick + 1, take.tempos, take.ticks_per_beat)
      - tick_to_seconds(note.start_tick, take.tempos, take.ticks_per_beat)
  )


def _mean_ms(values: list[float]) -> float:
  if not values:
    return 0.0
  return float(np.mean(values) * 1000.0)


def _max_ms(values: list[float]) -> float:
  if not values:
    return 0.0
  return float(np.max(values) * 1000.0)


def _overlaps(start: int, end: int, window_start: int, window_end: int) -> bool:
  return start < window_end and end > window_start


class _MidiReader:
  def __init__(self, data: bytes):
    self._data = data
    self._offset = 0

  @property
  def eof(self) -> bool:
    return self._offset >= len(self._data)

  def read_bytes(self, length: int) -> bytes:
    if length < 0 or self._offset + length > len(self._data):
      raise MidiDrumError("unexpected end of MIDI data")
    chunk = self._data[self._offset : self._offset + length]
    self._offset += length
    return chunk

  def read_u8(self) -> int:
    return self.read_bytes(1)[0]

  def peek_u8(self) -> int:
    if self.eof:
      raise MidiDrumError("unexpected end of MIDI data")
    return self._data[self._offset]

  def read_u32(self) -> int:
    return int.from_bytes(self.read_bytes(4), "big")

  def read_vlq(self) -> int:
    value = 0
    for _ in range(4):
      byte = self.read_u8()
      value = (value << 7) | (byte & 0x7F)
      if not (byte & 0x80):
        return value
    raise MidiDrumError("MIDI variable-length quantity is too long")
