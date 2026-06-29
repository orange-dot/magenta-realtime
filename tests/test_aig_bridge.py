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

"""Tests for the lossy AIG to MRT2 conditioning bridge."""

import unittest

import numpy as np

from magenta_rt import aig_bridge


def _packet(
    packet_id,
    role,
    kind,
    start_frame,
    duration_frames=1920,
    common_ports=None,
    dialect_ports=None,
):
  return {
      "id": packet_id,
      "source_event_id": packet_id.removesuffix(".packet"),
      "dialect": "adg",
      "role": role,
      "gesture_kind": kind,
      "start_frame": start_frame,
      "duration_frames": duration_frames,
      "strength": 0.5,
      "seed": 1,
      "trajectory": {
          "shape": "four_phase",
          "phases": [{"name": "attack", "duration_frames": duration_frames}],
      },
      "common_ports": common_ports or {},
      "dialect_ports": dialect_ports or {},
      "state_writes": {},
  }


class AigBridgeTest(unittest.TestCase):

  def test_pitch_mapping_uses_role_kind_and_hat_ports(self):
    self.assertEqual(aig_bridge.pitch_for_packet(_packet("kick", "kick", "anchor", 0)), 36)
    self.assertEqual(aig_bridge.pitch_for_packet(_packet("snare", "snare", "ghost", 0)), 38)
    self.assertEqual(aig_bridge.pitch_for_packet(_packet("hat", "hat", "breath", 0)), 42)
    self.assertEqual(
        aig_bridge.pitch_for_packet(
            _packet("hat-open", "hat", "breath", 0, dialect_ports={"openness": 0.5})
        ),
        46,
    )
    self.assertEqual(
        aig_bridge.pitch_for_packet(
            _packet("hat-choke", "hat", "choke", 0, common_ports={"choke_force": 1.0})
        ),
        44,
    )
    self.assertEqual(aig_bridge.pitch_for_packet(_packet("tom", "tom", "flash", 0)), 50)
    self.assertEqual(aig_bridge.pitch_for_packet(_packet("crash", "crash", "flash", 0)), 49)

  def test_schedule_sets_onsets_sustains_and_drum_lane(self):
    document = {
        "version": 1,
        "title": "test",
        "bpm": 120.0,
        "sample_rate_hz": 48000,
        "packets": [
            _packet("kick.packet", "kick", "anchor", 0, duration_frames=3840),
            _packet("snare.packet", "snare", "ghost", 1920, duration_frames=1920),
        ],
    }

    schedule = aig_bridge.build_conditioning_schedule(document, tail_seconds=0)

    self.assertEqual(schedule.notes.shape, (2, 128))
    self.assertEqual(schedule.drums.shape, (2, 1))
    self.assertEqual(schedule.notes[0, 36], 2)
    self.assertEqual(schedule.notes[1, 36], 1)
    self.assertEqual(schedule.notes[1, 38], 2)
    self.assertTrue(np.array_equal(schedule.drums[:, 0], np.array([1, 1], dtype=np.int8)))
    self.assertEqual(schedule.summary["mapped_packet_count"], 2)
    self.assertEqual(schedule.summary["pitch_counts"]["36"], 1)

  def test_summary_reports_lossy_fields_and_quantization(self):
    document = {
        "version": 1,
        "title": "test",
        "bpm": 120.0,
        "sample_rate_hz": 48000,
        "packets": [
            _packet(
                "late.packet",
                "hat",
                "breath",
                100,
                duration_frames=1920,
                dialect_ports={"wash": 0.5},
            )
        ],
    }

    schedule = aig_bridge.build_conditioning_schedule(document, tail_seconds=0)

    self.assertIn("strength_velocity", schedule.summary["lossy_fields"])
    self.assertGreater(schedule.summary["quantization"]["max_error_ms"], 0.0)
    self.assertEqual(schedule.packet_reports[0]["pitch"], 46)

  def test_window_schedule_tracks_source_frame_window(self):
    document = {
        "version": 1,
        "title": "test",
        "bpm": 120.0,
        "sample_rate_hz": 48000,
        "packets": [
            _packet("kick.packet", "kick", "anchor", 3840, duration_frames=3840),
        ],
    }

    schedule = aig_bridge.build_conditioning_schedule(document, tail_seconds=0)
    window = aig_bridge.window_conditioning_schedule(
        schedule, start_frame=1, max_frames=2
    )

    self.assertEqual(window.notes.shape, (2, 128))
    self.assertEqual(window.summary["frames"], 2)
    self.assertEqual(window.summary["window"]["start_frame"], 1)
    self.assertEqual(window.summary["window"]["end_frame"], 3)
    self.assertEqual(window.summary["window"]["overlapping_packet_count"], 1)
    self.assertEqual(window.notes[1, 36], 2)

  def test_pc4ms_adg_events_build_conditioning_without_new_adg(self):
    events = [
        {
            "id": "kick-1",
            "tick": 0,
            "role": "kick",
            "kind": "anchor",
            "duration_ticks": 480,
            "micro_offset_ticks": 0,
        },
        {
            "id": "bell-1",
            "tick": 480,
            "role": "ride_bell",
            "kind": "pressure",
            "duration_ticks": 120,
            "micro_offset_ticks": 0,
        },
    ]

    schedule = aig_bridge.build_conditioning_schedule_from_pc4ms_adg_events(
        events,
        tempo_bpm=120.0,
        ppqn=480,
        tail_seconds=0.0,
    )

    self.assertEqual(schedule.summary["schema"], "magenta_rt.pc4ms_adg_conditioning.v1")
    self.assertEqual(schedule.summary["event_count"], 2)
    self.assertEqual(schedule.summary["mapped_event_count"], 2)
    self.assertEqual(schedule.notes[0, 36], 2)
    self.assertEqual(schedule.notes[12, 53], 2)
    self.assertEqual(schedule.packet_reports[1]["source_event_id"], "bell-1")


if __name__ == "__main__":
  unittest.main()
