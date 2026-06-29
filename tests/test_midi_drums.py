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

from pathlib import Path
import tempfile
import unittest

from magenta_rt import midi_drums


class MidiDrumsTest(unittest.TestCase):

  def test_parses_channel_10_notes_and_running_status(self):
    with tempfile.TemporaryDirectory() as root:
      path = Path(root) / "drums.mid"
      path.write_bytes(_midi_file(_drum_track_with_running_status()))

      take = midi_drums.load_midi_drum_take(path, channel=10)

    self.assertEqual(take.midi_format, 0)
    self.assertEqual(take.ticks_per_beat, 480)
    self.assertEqual(take.selected_channel, 10)
    self.assertEqual(len(take.notes), 2)
    self.assertEqual([note.pitch for note in take.notes], [36, 38])
    self.assertEqual(take.channel_event_counts[10], 4)
    self.assertEqual(take.channel_event_counts[1], 2)

  def test_conditioning_preserves_gm_pitch_lanes(self):
    with tempfile.TemporaryDirectory() as root:
      path = Path(root) / "drums.mid"
      path.write_bytes(_midi_file(_drum_track_with_running_status()))
      take = midi_drums.load_midi_drum_take(path, channel=10)

    schedule = midi_drums.build_conditioning_schedule_from_midi_drums(
        take,
        tail_seconds=0.0,
    )

    self.assertEqual(schedule.summary["schema"], "magenta_rt.midi_drum_conditioning.v1")
    self.assertEqual(schedule.summary["note_count"], 2)
    self.assertEqual(schedule.summary["mapped_note_count"], 2)
    self.assertEqual(int(schedule.notes[0, 36]), 2)
    self.assertEqual(int(schedule.notes[3, 38]), 2)
    self.assertGreater(int(schedule.drums[:, 0].sum()), 0)

  def test_mixed_channel_filter_can_select_different_channel(self):
    with tempfile.TemporaryDirectory() as root:
      path = Path(root) / "drums.mid"
      path.write_bytes(_midi_file(_drum_track_with_running_status()))

      take = midi_drums.load_midi_drum_take(path, channel=1)

    self.assertEqual(len(take.notes), 1)
    self.assertEqual(take.notes[0].pitch, 60)

  def test_rejects_smpte_time_division(self):
    with tempfile.TemporaryDirectory() as root:
      path = Path(root) / "smpte.mid"
      path.write_bytes(_midi_file(_drum_track_with_running_status(), division=0xE250))

      with self.assertRaises(midi_drums.MidiDrumError):
        midi_drums.load_midi_drum_take(path)


def _midi_file(track: bytes, *, division: int = 480) -> bytes:
  header = (
      b"MThd"
      + (6).to_bytes(4, "big")
      + (0).to_bytes(2, "big")
      + (1).to_bytes(2, "big")
      + division.to_bytes(2, "big")
  )
  return header + b"MTrk" + len(track).to_bytes(4, "big") + track


def _drum_track_with_running_status() -> bytes:
  return b"".join(
      [
          _event(0, b"\xff\x03" + _vlq(len(b"fixture drums")) + b"fixture drums"),
          _event(0, b"\xff\x51\x03\x07\xa1\x20"),
          _event(0, bytes([0x99, 36, 100])),
          _event(120, bytes([38, 90])),
          _event(120, bytes([0x89, 36, 0])),
          _event(0, bytes([38, 0])),
          _event(0, bytes([0x90, 60, 100])),
          _event(120, bytes([0x80, 60, 0])),
          _event(0, b"\xff\x2f\x00"),
      ]
  )


def _event(delta: int, payload: bytes) -> bytes:
  return _vlq(delta) + payload


def _vlq(value: int) -> bytes:
  parts = [value & 0x7F]
  value >>= 7
  while value:
    parts.append(0x80 | (value & 0x7F))
    value >>= 7
  return bytes(reversed(parts))


if __name__ == "__main__":
  unittest.main()
