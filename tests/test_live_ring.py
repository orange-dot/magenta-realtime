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

"""Tests for the live-preview int16 interleaved ring."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from magenta_rt.live_ring import Int16InterleavedStereoRing
from magenta_rt.live_ring import PacingRingWriter
from magenta_rt.live_ring import RING_HEADER_SIZE
from magenta_rt.live_ring import RingHeader
from magenta_rt.live_ring import SharedMemoryInt16InterleavedStereoRing


def _stereo(values):
  values = np.array(values, dtype=np.int16)
  return np.stack([values, -values], axis=1)


class LiveRingTest(unittest.TestCase):

  def test_validates_format_sample_rate_and_channels(self):
    header = RingHeader()

    header.validate_consumer()

    with self.assertRaisesRegex(ValueError, "format mismatch"):
      header.validate_consumer(format="float32")
    with self.assertRaisesRegex(ValueError, "sample-rate mismatch"):
      header.validate_consumer(sample_rate=44100)
    with self.assertRaisesRegex(ValueError, "channel mismatch"):
      header.validate_consumer(channels=1)

  def test_binary_header_round_trips_for_rust_host(self):
    header = RingHeader(
        chunk_frames=4,
        capacity_frames=1234,
        write_cursor=400,
        read_cursor=100,
        underrun_frames=2,
        overrun_frames=3,
        low_water_frames=88,
    )

    data = header.to_bytes()
    parsed = RingHeader.from_bytes(data)

    self.assertEqual(len(data), RING_HEADER_SIZE)
    self.assertEqual(parsed.format, header.format)
    self.assertEqual(parsed.sample_rate, 48000)
    self.assertEqual(parsed.channels, 2)
    self.assertEqual(parsed.model_frame_size, 1920)
    self.assertEqual(parsed.chunk_frames, 4)
    self.assertEqual(parsed.capacity_frames, 1234)
    self.assertEqual(parsed.write_cursor, 400)
    self.assertEqual(parsed.read_cursor, 100)
    self.assertEqual(parsed.underrun_frames, 2)
    self.assertEqual(parsed.overrun_frames, 3)
    self.assertEqual(parsed.low_water_frames, 88)

  def test_wraparound_preserves_audio_frame_order(self):
    ring = Int16InterleavedStereoRing(capacity_frames=6)
    ring.write(_stereo([1, 2, 3, 4]))
    self.assertTrue(np.array_equal(ring.read(3), _stereo([1, 2, 3])))

    ring.write(_stereo([5, 6, 7, 8, 9]))
    actual = ring.read(6)

    self.assertTrue(np.array_equal(actual, _stereo([4, 5, 6, 7, 8, 9])))
    self.assertEqual(ring.header.read_cursor, 9)
    self.assertEqual(ring.header.write_cursor, 9)

  def test_read_underrun_zero_fills_and_counts_audio_frames(self):
    ring = Int16InterleavedStereoRing(capacity_frames=8)
    ring.write(_stereo([10, 11]))

    actual = ring.read(5)

    expected = np.concatenate(
        [_stereo([10, 11]), np.zeros((3, 2), dtype=np.int16)],
        axis=0,
    )
    self.assertTrue(np.array_equal(actual, expected))
    self.assertEqual(ring.header.underrun_frames, 3)
    self.assertEqual(ring.header.read_cursor, 2)

  def test_rejects_non_int16_or_non_stereo_payloads(self):
    ring = Int16InterleavedStereoRing(capacity_frames=8)

    with self.assertRaisesRegex(TypeError, "int16"):
      ring.write(np.zeros((4, 2), dtype=np.float32))
    with self.assertRaisesRegex(ValueError, "shape"):
      ring.write(np.zeros((4, 1), dtype=np.int16))

  def test_shared_memory_ring_writes_frame_cursors(self):
    with tempfile.TemporaryDirectory() as root:
      path = Path(root) / "mrt2-live.ring"
      ring = SharedMemoryInt16InterleavedStereoRing(
          path,
          capacity_frames=6,
          chunk_frames=4,
      )
      try:
        result = ring.write(_stereo([1, 2, 3, 4]))
        header = RingHeader.from_bytes(path.read_bytes()[:RING_HEADER_SIZE])

        self.assertEqual(result["written_frames"], 4)
        self.assertEqual(header.write_cursor, 4)
        self.assertEqual(header.read_cursor, 0)
        self.assertEqual(header.capacity_frames, 6)
        self.assertEqual(path.stat().st_size, RING_HEADER_SIZE + 6 * 2 * 2)
      finally:
        ring.close()

  def test_pacing_writer_writes_immediately_when_space_is_available(self):
    ring = Int16InterleavedStereoRing(capacity_frames=4)
    writer = PacingRingWriter(ring)

    result = writer.write(_stereo([1, 2]))

    self.assertEqual(result["backpressure_status"], "ready")
    self.assertEqual(result["backpressure_wait_count"], 0)
    self.assertEqual(result["written_frames"], 2)
    self.assertEqual(ring.available_frames, 2)

  def test_pacing_writer_skips_write_when_stop_is_requested(self):
    ring = Int16InterleavedStereoRing(capacity_frames=2)
    ring.write(_stereo([1, 2]))
    writer = PacingRingWriter(ring, should_stop=lambda: True)

    result = writer.write(_stereo([3]))

    self.assertEqual(result["backpressure_status"], "stopped")
    self.assertEqual(result["written_frames"], 0)
    self.assertEqual(result["dropped_frames"], 0)
    self.assertEqual(writer.stopped_count, 1)
    self.assertTrue(np.array_equal(ring.read(2), _stereo([1, 2])))

  def test_pacing_writer_timeout_falls_back_to_ring_overwrite(self):
    ring = Int16InterleavedStereoRing(capacity_frames=2)
    ring.write(_stereo([1, 2]))
    writer = PacingRingWriter(ring, timeout_seconds=0.0)

    result = writer.write(_stereo([3]))

    self.assertEqual(result["backpressure_status"], "timeout")
    self.assertEqual(result["backpressure_timeout_count"], 1)
    self.assertEqual(result["dropped_frames"], 1)
    self.assertEqual(ring.header.overrun_frames, 1)
    self.assertTrue(np.array_equal(ring.read(2), _stereo([2, 3])))


if __name__ == "__main__":
  unittest.main()
