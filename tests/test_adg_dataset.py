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

"""Tests for PC4MS full-ADG take discovery."""

import json
import tempfile
import unittest
from pathlib import Path

from magenta_rt import adg_dataset


class FullAdgDatasetTest(unittest.TestCase):

  def test_discovers_only_complete_adg_bundles(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      root = Path(tmpdir)
      (root / "old.generated-live.mid").write_bytes(b"MThd")
      bundle = root / "drum-live-1.adg-bundle"
      bundle.mkdir()
      for name in [
          "take.adg.toml",
          "take.adg-events.json",
          "take.adg-summary.json",
          "take.generated-live.mid",
      ]:
        (bundle / name).write_text("{}")
      manifest = {
          "schema": adg_dataset.FULL_ADG_BUNDLE_SCHEMA,
          "take_id": "drum-live-1",
          "bundle_dir": str(bundle),
          "adg_toml": str(bundle / "take.adg.toml"),
          "adg_events": str(bundle / "take.adg-events.json"),
          "adg_summary": str(bundle / "take.adg-summary.json"),
          "midi": str(bundle / "take.generated-live.mid"),
      }
      (bundle / "manifest.json").write_text(json.dumps(manifest))

      takes = adg_dataset.discover_full_adg_takes(root)

    self.assertEqual([take.take_id for take in takes], ["drum-live-1"])
    self.assertEqual(takes[0].adg_toml.name, "take.adg.toml")

  def test_rejects_manifest_without_full_adg_schema(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      manifest = Path(tmpdir) / "manifest.json"
      manifest.write_text(json.dumps({"schema": "old.mid.only"}))

      with self.assertRaisesRegex(adg_dataset.AdgDatasetError, "not a"):
        adg_dataset.load_full_adg_take(manifest)


if __name__ == "__main__":
  unittest.main()
