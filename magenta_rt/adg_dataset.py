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

"""PC4MS full-ADG take discovery for Magenta RT local tests."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any


FULL_ADG_BUNDLE_SCHEMA = "pc4ms.generated_live_adg_bundle.v1"


class AdgDatasetError(ValueError):
  """A dataset entry is missing the full ADG bundle contract."""


@dataclasses.dataclass(frozen=True)
class FullAdgTake:
  """A complete PC4MS generated live ADG bundle.

  Older ``*.generated-live.mid`` files are intentionally not represented here.
  They do not carry the ADG authority/debug artifacts needed for the live MRT2
  conditioning tests.
  """

  take_id: str
  bundle_dir: Path
  manifest: Path
  adg_toml: Path
  adg_events: Path
  adg_summary: Path
  midi: Path
  live_chunks: Path | None = None
  per_chunk_traces: Path | None = None
  runtime_snapshot: Path | None = None
  trace: Path | None = None
  aig_export_request: Path | None = None


def discover_full_adg_takes(root: str | Path) -> list[FullAdgTake]:
  """Returns only complete ``*.adg-bundle`` takes under ``root``."""

  root = Path(root)
  takes = []
  for manifest in sorted(root.glob("*.adg-bundle/manifest.json")):
    try:
      takes.append(load_full_adg_take(manifest))
    except AdgDatasetError:
      continue
  return takes


def select_full_adg_take(
    root: str | Path, *, take_id: str | None = None
) -> FullAdgTake:
  """Selects a full ADG take, optionally by ``take_id``."""

  takes = discover_full_adg_takes(root)
  if not takes:
    raise AdgDatasetError(f"no full ADG takes found under {root}")
  if take_id is None:
    return takes[-1]
  for take in takes:
    if take.take_id == take_id:
      return take
  raise AdgDatasetError(f"full ADG take not found: {take_id}")


def load_full_adg_take(path: str | Path) -> FullAdgTake:
  """Loads a full ADG take from a bundle directory or manifest path."""

  path = Path(path)
  manifest = path / "manifest.json" if path.is_dir() else path
  if manifest.name != "manifest.json":
    raise AdgDatasetError(f"expected manifest.json or bundle dir, got {path}")
  if not manifest.exists():
    raise AdgDatasetError(f"missing ADG bundle manifest: {manifest}")

  payload = _load_json(manifest)
  if payload.get("schema") != FULL_ADG_BUNDLE_SCHEMA:
    raise AdgDatasetError(
        f"{manifest} is not a {FULL_ADG_BUNDLE_SCHEMA} manifest"
    )

  bundle_dir = _resolve_payload_path(
      manifest.parent, _required_string(payload, "bundle_dir")
  )
  take_id = _required_string(payload, "take_id")
  debug = payload.get("debug", {})
  if not isinstance(debug, dict):
    debug = {}
  aig_export = payload.get("aig_export", {})
  if not isinstance(aig_export, dict):
    aig_export = {}

  take = FullAdgTake(
      take_id=take_id,
      bundle_dir=bundle_dir,
      manifest=manifest,
      adg_toml=_resolve_existing_file(
          manifest.parent, _required_string(payload, "adg_toml"), "adg_toml"
      ),
      adg_events=_resolve_existing_file(
          manifest.parent,
          _required_string(payload, "adg_events"),
          "adg_events",
      ),
      adg_summary=_resolve_existing_file(
          manifest.parent,
          _required_string(payload, "adg_summary"),
          "adg_summary",
      ),
      midi=_resolve_existing_file(
          manifest.parent, _required_string(payload, "midi"), "midi"
      ),
      live_chunks=_optional_existing_file(
          manifest.parent, debug.get("live_chunks"), "live_chunks"
      ),
      per_chunk_traces=_optional_existing_file(
          manifest.parent, debug.get("per_chunk_traces"), "per_chunk_traces"
      ),
      runtime_snapshot=_optional_existing_file(
          manifest.parent, debug.get("runtime_snapshot"), "runtime_snapshot"
      ),
      trace=_optional_existing_file(
          manifest.parent, debug.get("trace"), "trace"
      ),
      aig_export_request=_optional_existing_file(
          manifest.parent, aig_export.get("request"), "aig_export.request"
      ),
  )
  _validate_bundle_identity(take)
  return take


def _validate_bundle_identity(take: FullAdgTake) -> None:
  if not take.bundle_dir.exists():
    raise AdgDatasetError(f"bundle_dir does not exist: {take.bundle_dir}")
  if take.manifest.parent.resolve(strict=False) != take.bundle_dir.resolve(
      strict=False
  ):
    raise AdgDatasetError(
        f"manifest is not inside bundle_dir: {take.manifest}"
    )


def _load_json(path: Path) -> dict[str, Any]:
  try:
    payload = json.loads(path.read_text())
  except json.JSONDecodeError as exc:
    raise AdgDatasetError(f"invalid JSON in {path}") from exc
  if not isinstance(payload, dict):
    raise AdgDatasetError(f"{path} must contain a JSON object")
  return payload


def _required_string(payload: dict[str, Any], field: str) -> str:
  value = payload.get(field)
  if not isinstance(value, str) or not value:
    raise AdgDatasetError(f"manifest field {field} must be a non-empty string")
  return value


def _optional_existing_file(
    base: Path, value: Any, field: str
) -> Path | None:
  if value is None:
    return None
  if not isinstance(value, str) or not value:
    raise AdgDatasetError(f"manifest field {field} must be a string")
  return _resolve_existing_file(base, value, field)


def _resolve_existing_file(base: Path, value: str, field: str) -> Path:
  path = _resolve_payload_path(base, value)
  if not path.exists():
    raise AdgDatasetError(f"{field} does not exist: {path}")
  if not path.is_file():
    raise AdgDatasetError(f"{field} must be a file: {path}")
  return path


def _resolve_payload_path(base: Path, value: str) -> Path:
  path = Path(value)
  if path.is_absolute():
    return path
  return (base / path).resolve(strict=False)
