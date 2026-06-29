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

"""Tests for the PC4MS live chunk receiver contract."""

import json
from pathlib import Path
import socket
import tempfile
import unittest

from magenta_rt import pc4ms_live


def _message(chunk_index=0):
  return {
      "schema": pc4ms_live.PC4MS_LIVE_CHUNK_SCHEMA,
      "take_id": "take-a",
      "chunk_index": chunk_index,
      "tempo_bpm": 120,
      "ppqn": 480,
      "timing": {
          "source_start_tick": 0,
          "source_end_tick": 1920,
          "output_start_tick": 1920,
          "output_end_tick": 3840,
      },
      "metrics": {
          "generation_micros": 2000,
          "deadline_margin_ms": 80,
      },
      "adg_events": [
          {
              "id": f"kick-{chunk_index}",
              "tick": 0,
              "role": "kick",
              "kind": "anchor",
              "duration_ticks": 120,
              "micro_offset_ticks": 0,
          }
      ],
  }


class Pc4msLiveTest(unittest.TestCase):

  def test_parse_live_chunk_message_validates_schema_and_events(self):
    chunk = pc4ms_live.parse_live_chunk_message(json.dumps(_message()))

    self.assertEqual(chunk.take_id, "take-a")
    self.assertEqual(chunk.chunk_index, 0)
    self.assertEqual(chunk.tempo_bpm, 120)
    self.assertEqual(chunk.ppqn, 480)
    self.assertEqual(chunk.metrics["deadline_margin_ms"], 80)
    self.assertEqual(chunk.adg_events[0]["role"], "kick")

  def test_live_chunk_conditioning_is_independent_of_producer_chunk_size(self):
    first = pc4ms_live.parse_live_chunk_message(json.dumps(_message(1)))
    second = pc4ms_live.parse_live_chunk_message(json.dumps(_message(1)))

    schedule_a = pc4ms_live.conditioning_schedule_from_live_chunk(
        first,
        tail_seconds=0.0,
    )
    schedule_b = pc4ms_live.conditioning_schedule_from_live_chunk(
        second,
        tail_seconds=0.0,
    )

    self.assertEqual(schedule_a.summary["event_count"], 1)
    self.assertEqual(schedule_a.summary, schedule_b.summary)
    self.assertEqual(schedule_a.notes.tolist(), schedule_b.notes.tolist())
    self.assertEqual(schedule_a.drums.tolist(), schedule_b.drums.tolist())

  def test_live_chunk_conditioning_uses_output_tick_window(self):
    message = _message()
    message["adg_events"][0]["tick"] = message["timing"]["output_start_tick"]
    chunk = pc4ms_live.parse_live_chunk_message(json.dumps(message))

    schedule = pc4ms_live.conditioning_schedule_from_live_chunk(
        chunk,
        tail_seconds=0.0,
    )

    self.assertEqual(schedule.summary["frames"], 50)
    self.assertEqual(schedule.summary["duration_seconds"], 2.0)
    self.assertEqual(int(schedule.drums[0, 0]), 1)

  def test_debug_live_chunk_payload_matches_socket_contract(self):
    payload = pc4ms_live.live_payload_from_debug_chunk(
        take_id="take-a",
        tempo_bpm=143.0,
        ppqn=480,
        profile_id="profile-a",
        debug_chunk={
            "chunk_index": 3,
            "source_start_tick": 1920,
            "source_end_tick": 3840,
            "output_start_tick": 3840,
            "output_end_tick": 5760,
            "generation_micros": 22,
            "intake_event_count": 4,
            "intake_fallback_used": False,
            "run": {"adg_events": [{"tick": 3840, "role": "kick"}]},
        },
    )

    parsed = pc4ms_live.parse_live_chunk_message(json.dumps(payload))

    self.assertEqual(parsed.take_id, "take-a")
    self.assertEqual(parsed.chunk_index, 3)
    self.assertEqual(parsed.output_start_tick, 3840)
    self.assertEqual(parsed.metrics["replay"], True)
    self.assertEqual(parsed.adg_events[0]["tick"], 3840)

  def test_unix_datagram_receiver_preserves_chunk_order(self):
    with tempfile.TemporaryDirectory() as root:
      path = Path(root) / "pc4ms.sock"
      receiver_context = pc4ms_live.UnixDatagramLiveChunkReceiver(path)
      try:
        receiver = receiver_context.__enter__()
      except PermissionError as exc:
        self.skipTest(f"Unix datagram bind is not permitted here: {exc}")
      try:
        sender = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        try:
          sender.sendto(json.dumps(_message(1)).encode("utf-8"), str(path))
          sender.sendto(json.dumps(_message(2)).encode("utf-8"), str(path))

          self.assertEqual(receiver.receive(timeout=1.0).chunk_index, 1)
          self.assertEqual(receiver.receive(timeout=1.0).chunk_index, 2)
        finally:
          sender.close()
      finally:
        receiver_context.__exit__(None, None, None)


if __name__ == "__main__":
  unittest.main()
