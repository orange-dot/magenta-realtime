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

"""Tests for render worker request validation."""

import tempfile
import unittest
from pathlib import Path

from magenta_rt.render_worker import BadRequest
from magenta_rt.render_worker import NotFound
from magenta_rt.render_worker import PathPolicy
from magenta_rt.render_worker import RenderRequest


class RenderWorkerRequestTest(unittest.TestCase):

  def setUp(self):
    self._tmp = tempfile.TemporaryDirectory()
    root = Path(self._tmp.name)
    self.workspace = root / "workspace"
    self.magenta = root / "magenta"
    self.workspace.mkdir()
    self.magenta.mkdir()
    self.gesture_packets = self.workspace / "input.gesture-packets.toml"
    self.atom_specs = self.workspace / "input.atom-specs.toml"
    self.gesture_packets.write_text("[[packets]]\nid = 'p1'\n")
    self.atom_specs.write_text("[[atoms]]\nid = 'a1'\n")
    self.policy = PathPolicy(self.workspace, self.magenta)

  def tearDown(self):
    self._tmp.cleanup()

  def test_accepts_container_workspace_paths_and_null_atom_specs(self):
    payload = self._payload(atom_specs=None)

    request = RenderRequest.from_json(payload, self.policy)

    self.assertEqual(request.gesture_packets, self.gesture_packets)
    self.assertIsNone(request.atom_specs)
    self.assertEqual(request.max_frames, 50)
    self.assertEqual(request.top_k, 40)

  def test_accepts_full_adg_bundle_manifest_source(self):
    manifest = self.workspace / "take.adg-bundle" / "manifest.json"
    manifest.parent.mkdir()
    manifest.write_text("{}")
    payload = self._payload(
        gesture_packets=None,
        atom_specs=None,
        adg_bundle_manifest=str(manifest),
    )

    request = RenderRequest.from_json(payload, self.policy)

    self.assertIsNone(request.gesture_packets)
    self.assertEqual(request.adg_bundle_manifest, manifest)

  def test_requires_one_render_source(self):
    payload = self._payload(gesture_packets=None, atom_specs=None)

    with self.assertRaisesRegex(BadRequest, "one of gesture_packets"):
      RenderRequest.from_json(payload, self.policy)

  def test_rejects_relative_input_paths(self):
    payload = self._payload()
    payload["gesture_packets"] = "relative.toml"

    with self.assertRaisesRegex(BadRequest, "absolute path"):
      RenderRequest.from_json(payload, self.policy)

  def test_rejects_output_outside_workspace(self):
    payload = self._payload()
    payload["output"] = str(self.magenta / "render.wav")

    with self.assertRaisesRegex(BadRequest, "output must be under"):
      RenderRequest.from_json(payload, self.policy)

  def test_reports_missing_readable_path_as_not_found(self):
    payload = self._payload()
    payload["atom_specs"] = str(self.workspace / "missing.toml")

    with self.assertRaisesRegex(NotFound, "atom_specs does not exist"):
      RenderRequest.from_json(payload, self.policy)

  def test_rejects_non_positive_max_frames(self):
    payload = self._payload(max_frames=0)

    with self.assertRaisesRegex(BadRequest, "max_frames must be >= 1"):
      RenderRequest.from_json(payload, self.policy)

  def _payload(self, **overrides):
    payload = {
        "gesture_packets": str(self.gesture_packets),
        "atom_specs": str(self.atom_specs),
        "output": str(self.workspace / "render.wav"),
        "manifest": str(self.workspace / "render.manifest.json"),
        "conditioning": str(self.workspace / "render.conditioning.npz"),
        "prompt": "tight acoustic funk drums",
        "tail_seconds": 2.0,
        "max_frames": 50,
        "temperature": 1.1,
        "top_k": 40,
        "cfg_musiccoca": 3.0,
        "cfg_notes": 3.0,
        "cfg_drums": 4.0,
    }
    payload.update(overrides)
    return payload


if __name__ == "__main__":
  unittest.main()
