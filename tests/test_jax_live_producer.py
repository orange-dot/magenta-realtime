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

"""Unit tests for JAX live producer scheduling helpers."""

import types
import unittest

import numpy as np

from magenta_rt.jax.live_producer import JaxLiveProducer
from magenta_rt.jax.live_producer import PreparedFrame
from magenta_rt.jax.live_producer import PreparedSchedule
from magenta_rt.jax.live_producer import PromptTokenUpdate
from magenta_rt.jax.live_producer import align_prompt_update_frame
from magenta_rt.jax.live_producer import chunk_frame_ranges
from magenta_rt.jax.live_producer import normalize_prompt_updates
from magenta_rt.jax.live_producer import style_tokens_for_frame


class JaxLiveProducerHelperTest(unittest.TestCase):

  def test_chunk_frame_ranges_preserve_order_across_boundaries(self):
    ranges = chunk_frame_ranges(total_frames=10, chunk_frames=4)
    flattened = [
        frame
        for start, end in ranges
        for frame in range(start, end)
    ]

    self.assertEqual(ranges, [(0, 4), (4, 8), (8, 10)])
    self.assertEqual(flattened, list(range(10)))

  def test_prompt_updates_apply_on_chunk_boundaries(self):
    base = (1, 1)
    update = PromptTokenUpdate(requested_frame=5, style_tokens=(2, 2))
    updates = normalize_prompt_updates([update], chunk_frames=8)

    self.assertEqual(align_prompt_update_frame(5, 8), 8)
    self.assertEqual(style_tokens_for_frame(base, updates, 7), (base, 0))
    self.assertEqual(style_tokens_for_frame(base, updates, 8), ((2, 2), 1))
    self.assertEqual(style_tokens_for_frame(base, updates, 9), ((2, 2), 1))


class _FakeJax:

  @staticmethod
  def device_put(value):
    return np.asarray(value)

  @staticmethod
  def device_get(value):
    return np.asarray(value)


class _FakeBlock:

  def __init__(self, values):
    self.values = values


class _FakeMrt:

  _params = object()

  def _build_conditioning(
      self, style, notes, drums, cfgs, temperature, top_k
  ):
    del temperature, top_k
    tokens = np.asarray(style + notes + drums + cfgs, dtype=np.int32) + 7
    return _FakeBlock(tokens.reshape(1, 1, -1)), {
        "temperature": np.asarray([1.1]),
        "top_k": np.asarray([40], dtype=np.int32),
    }


class JaxLiveProducerScanHelperTest(unittest.TestCase):

  def test_prepare_schedule_stores_device_token_matrix(self):
    schedule = types.SimpleNamespace(
        notes=np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int8),
        drums=np.asarray([[1], [0]], dtype=np.int8),
        summary={"frames": 2},
    )
    producer = JaxLiveProducer(
        _FakeMrt(),
        jax_module=_FakeJax(),
        sequence_layers_module=types.SimpleNamespace(),
    )

    prepared = producer.prepare_schedule(
        schedule,
        style_tokens=[10, 11],
        cfgs=[20, 21],
        temperature=1.1,
        top_k=40,
        chunk_frames=4,
    )

    self.assertEqual(prepared.token_values.shape, (2, 8))
    self.assertEqual(
        prepared.token_values[0].tolist(),
        [17, 18, 7, 8, 9, 8, 27, 28],
    )
    self.assertIsNotNone(prepared.constants)

  def test_scan_chunked_preserves_tail_order(self):
    producer = object.__new__(JaxLiveProducer)
    producer._jax = _FakeJax()
    producer._scan_chunk = types.MethodType(_fake_scan_chunk, producer)
    frames = tuple(
        PreparedFrame(
            frame_index=index,
            block=None,
            constants={},
            style_generation=0,
        )
        for index in range(5)
    )
    prepared = PreparedSchedule(
        frames=frames,
        token_values=np.arange(5, dtype=np.int32).reshape(5, 1),
        constants={},
        chunk_frames=4,
        frame_audio_frames=2,
        sample_rate=10,
    )

    samples, _, metrics = producer._generate_scan_chunked(
        prepared, state=object(), ring_writer=None
    )

    self.assertEqual(samples[:, 0].tolist(), [0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    self.assertEqual(metrics.mode, "scan-chunked")
    self.assertEqual(metrics.chunks, 2)
    self.assertEqual(metrics.frames, 5)


def _fake_scan_chunk(self, token_chunk, constants, state):
  del self, constants
  values = np.repeat(np.asarray(token_chunk[:, 0], dtype=np.int16), 2)
  return np.stack([values, values], axis=1), state


if __name__ == "__main__":
  unittest.main()
