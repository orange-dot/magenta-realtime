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

"""Long-running local render worker for AIG/ADG Magenta RT renders."""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
from http import HTTPStatus
import http.server
import json
import logging
import os
from pathlib import Path
import socketserver
import threading
import time
from typing import Any, Literal, TYPE_CHECKING

import numpy as np

from magenta_rt import adg_dataset
from magenta_rt import aig_bridge
from magenta_rt import paths

if TYPE_CHECKING:
  from magenta_rt import audio


BackendName = Literal["jax-gpu", "mlx-cpu"]
ProducerModeName = Literal["chunked", "scan-chunked"]
JsonDict = dict[str, Any]

LOGGER = logging.getLogger(__name__)


class WorkerError(Exception):
  """HTTP-safe worker error."""

  status = HTTPStatus.BAD_REQUEST


class BadRequest(WorkerError):
  """Request validation failed."""


class NotFound(WorkerError):
  """A referenced request path does not exist."""

  status = HTTPStatus.NOT_FOUND


@dataclasses.dataclass(frozen=True)
class WorkerConfig:
  """Process-level worker configuration."""

  backend: BackendName
  model: str
  checkpoint: str
  host: str
  port: int
  workspace_root: Path
  magenta_mount: Path
  mlx_bits: int
  mlx_warmup_steps: int
  progress_every: int
  chunk_frames: int
  producer_mode: ProducerModeName

  @classmethod
  def from_env(cls) -> "WorkerConfig":
    backend = _env_backend("MAGENTA_RT_RENDER_BACKEND", "jax-gpu")
    port = _env_int("MAGENTA_RT_PORT", 8080, minimum=1)
    mlx_bits = _env_int("MAGENTA_RT_MLX_BITS", 8, minimum=1)
    mlx_warmup_steps = _env_int(
        "MAGENTA_RT_MLX_WARMUP_STEPS", 0, minimum=0
    )
    progress_every = _env_int("MAGENTA_RT_PROGRESS_EVERY", 25, minimum=0)
    chunk_frames = _env_int("MAGENTA_RT_CHUNK_FRAMES", 8, minimum=1)
    producer_mode = _env_choice(
        "MAGENTA_RT_PRODUCER_MODE", "chunked", {"chunked", "scan-chunked"}
    )
    return cls(
        backend=backend,
        model=os.environ.get("MAGENTA_RT_MODEL", "mrt2_small"),
        checkpoint=os.environ.get(
            "MAGENTA_RT_CHECKPOINT", "mrt2_small.safetensors"
        ),
        host=os.environ.get("MAGENTA_RT_HOST", "0.0.0.0"),
        port=port,
        workspace_root=Path(
            os.environ.get("MAGENTA_RT_WORKSPACE_ROOT", "/workspace")
        ),
        magenta_mount=Path(
            os.environ.get("MAGENTA_RT_MAGENTA_ROOT", "/magenta")
        ),
        mlx_bits=mlx_bits,
        mlx_warmup_steps=mlx_warmup_steps,
        progress_every=progress_every,
        chunk_frames=chunk_frames,
        producer_mode=producer_mode,
    )


@dataclasses.dataclass(frozen=True)
class PathPolicy:
  """Container path policy for request files."""

  workspace_root: Path
  magenta_mount: Path

  def readable_path(self, value: Any, field: str) -> Path:
    path = _absolute_path(value, field)
    if not (
        _is_under(path, self.workspace_root)
        or _is_under(path, self.magenta_mount)
    ):
      raise BadRequest(
          f"{field} must be under {self.workspace_root} or "
          f"{self.magenta_mount}"
      )
    if not path.exists():
      raise NotFound(f"{field} does not exist: {path}")
    if not path.is_file():
      raise BadRequest(f"{field} must be a file: {path}")
    return path

  def optional_readable_path(self, value: Any, field: str) -> Path | None:
    if value is None:
      return None
    return self.readable_path(value, field)

  def writable_path(self, value: Any, field: str) -> Path:
    path = _absolute_path(value, field)
    if not _is_under(path, self.workspace_root):
      raise BadRequest(f"{field} must be under {self.workspace_root}")
    if path.exists() and path.is_dir():
      raise BadRequest(f"{field} must be a file path, not a directory: {path}")
    return path


@dataclasses.dataclass(frozen=True)
class RenderRequest:
  """Validated AIG/ADG render request."""

  gesture_packets: Path | None
  adg_bundle_manifest: Path | None
  atom_specs: Path | None
  output: Path
  manifest: Path
  conditioning: Path
  prompt: str
  tail_seconds: float
  max_frames: int | None
  temperature: float
  top_k: int
  cfg_musiccoca: float
  cfg_notes: float
  cfg_drums: float

  @classmethod
  def from_json(
      cls, payload: Mapping[str, Any], policy: PathPolicy
  ) -> "RenderRequest":
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
      raise BadRequest("prompt must be a non-empty string")

    gesture_packets_value = payload.get("gesture_packets")
    adg_bundle_manifest_value = payload.get("adg_bundle_manifest")
    if gesture_packets_value is None and adg_bundle_manifest_value is None:
      raise BadRequest(
          "one of gesture_packets or adg_bundle_manifest is required"
      )
    if gesture_packets_value is not None and adg_bundle_manifest_value is not None:
      raise BadRequest(
          "gesture_packets and adg_bundle_manifest are mutually exclusive"
      )

    return cls(
        gesture_packets=policy.optional_readable_path(
            gesture_packets_value, "gesture_packets"
        ),
        adg_bundle_manifest=policy.optional_readable_path(
            adg_bundle_manifest_value, "adg_bundle_manifest"
        ),
        atom_specs=policy.optional_readable_path(
            payload.get("atom_specs"), "atom_specs"
        ),
        output=policy.writable_path(payload.get("output"), "output"),
        manifest=policy.writable_path(payload.get("manifest"), "manifest"),
        conditioning=policy.writable_path(
            payload.get("conditioning"), "conditioning"
        ),
        prompt=prompt,
        tail_seconds=_payload_float(
            payload, "tail_seconds", 2.0, minimum=0.0
        ),
        max_frames=_payload_optional_int(
            payload, "max_frames", minimum=1
        ),
        temperature=_payload_float(
            payload, "temperature", 1.1, minimum=0.0
        ),
        top_k=_payload_int(payload, "top_k", 40, minimum=1),
        cfg_musiccoca=_payload_float(payload, "cfg_musiccoca", 3.0),
        cfg_notes=_payload_float(payload, "cfg_notes", 3.0),
        cfg_drums=_payload_float(payload, "cfg_drums", 4.0),
    )


@dataclasses.dataclass(frozen=True)
class RenderResult:
  """Completed render metadata."""

  response: JsonDict
  manifest: JsonDict


@dataclasses.dataclass(frozen=True)
class RenderedAudio:
  """Audio plus backend-specific producer metrics."""

  waveform: "audio.Waveform"
  producer_metrics: JsonDict


class LoadedRenderer:
  """Backend interface for a process-loaded Magenta model."""

  backend: BackendName
  model: str
  checkpoint: str
  load_seconds: float
  device_info: JsonDict

  def render_schedule(
      self, request: RenderRequest, schedule: aig_bridge.ConditioningSchedule
  ) -> RenderedAudio:
    raise NotImplementedError

  def manifest_backend_info(self) -> JsonDict:
    return {
        "backend": self.backend,
        "model": self.model,
        "checkpoint": self.checkpoint,
        "load_seconds": self.load_seconds,
        "devices": self.device_info,
    }


class JaxGpuRenderer(LoadedRenderer):
  """JAX GPU renderer preserving the per-frame streaming-step loop."""

  backend: BackendName = "jax-gpu"

  def __init__(self, config: WorkerConfig):
    import jax
    from magenta_rt import MagentaRT2Jax
    from magenta_rt.jax.live_producer import JaxLiveProducer
    from magenta_rt.jax.system import discretize_cfg

    self._jax = jax
    self._discretize_cfg = discretize_cfg
    self.model = config.model
    self.checkpoint = config.checkpoint
    self.progress_every = config.progress_every
    self.chunk_frames = config.chunk_frames
    self.producer_mode = config.producer_mode
    self.device_info = _jax_device_info(jax)
    if not self.device_info["cuda_devices"]:
      raise RuntimeError(
          "JAX GPU backend requires at least one CUDA/GPU JAX device; "
          f"visible devices were {self.device_info['devices']}"
      )

    started = time.time()
    self._mrt = MagentaRT2Jax(
        size=config.model,
        checkpoint=config.checkpoint,
    )
    self._producer = JaxLiveProducer(self._mrt, jax_module=jax)
    self.load_seconds = time.time() - started

  def render_schedule(
      self, request: RenderRequest, schedule: aig_bridge.ConditioningSchedule
  ) -> RenderedAudio:
    from magenta_rt import audio as audio_module

    style_tokens = self._producer.prepare_style_tokens(
        request.prompt, use_mapper=True
    )
    cfgs = [
        self._discretize_cfg(request.cfg_musiccoca, 0.2, 40),
        self._discretize_cfg(request.cfg_notes, 0.2, 40),
        self._discretize_cfg(request.cfg_drums, 1.0, 8),
    ]
    prepared = self._producer.prepare_schedule(
        schedule,
        style_tokens=style_tokens,
        cfgs=cfgs,
        temperature=request.temperature,
        top_k=request.top_k,
        chunk_frames=self.chunk_frames,
    )
    if self.producer_mode == "scan-chunked":
      self._producer.precompile_scan_chunks(prepared)
    samples_i16, _, metrics = self._producer.generate_int16(
        prepared, mode=self.producer_mode
    )
    samples = samples_i16.astype(np.float32) / 32768.0
    waveform = audio_module.Waveform(
        samples,
        sample_rate=aig_bridge.MAGENTA_SAMPLE_RATE,
    )
    return RenderedAudio(
        waveform=waveform,
        producer_metrics=metrics.to_dict(),
    )


class MlxCpuRenderer(LoadedRenderer):
  """MLX CPU renderer preserving the per-frame generate(frames=1) loop."""

  backend: BackendName = "mlx-cpu"

  def __init__(self, config: WorkerConfig):
    from magenta_rt.mlx.runtime import prepare_cpu_runtime_env

    prepare_cpu_runtime_env("cpu")

    import mlx.core as mx
    from magenta_rt import MagentaRT2Mlx

    self._mx = mx
    self.model = config.model
    self.checkpoint = config.checkpoint
    self.bits = config.mlx_bits
    self.warmup_steps = config.mlx_warmup_steps
    self.progress_every = config.progress_every
    started = time.time()
    self._mrt = MagentaRT2Mlx(
        size=config.model,
        checkpoint=config.checkpoint,
        bits=config.mlx_bits,
        device="cpu",
        warmup_steps=config.mlx_warmup_steps,
    )
    self.load_seconds = time.time() - started
    self.device_info = {
        "device": "cpu",
        "mlx_default_device": str(mx.default_device()),
    }

  def render_schedule(
      self, request: RenderRequest, schedule: aig_bridge.ConditioningSchedule
  ) -> RenderedAudio:
    from magenta_rt import audio as audio_module

    embedding = self._mrt.embed_style(request.prompt, use_mapper=True)

    state = None
    chunks = []
    frames = int(schedule.summary["frames"])
    for index in range(frames):
      notes = schedule.notes[index].astype(np.int32).tolist()
      drums = schedule.drums[index].astype(np.int32).tolist()
      step_waveform, state = self._mrt.generate(
          style=embedding,
          notes=notes,
          drums=drums,
          cfg_musiccoca=request.cfg_musiccoca,
          cfg_notes=request.cfg_notes,
          cfg_drums=request.cfg_drums,
          temperature=request.temperature,
          top_k=request.top_k,
          frames=1,
          state=state,
      )
      chunks.append(step_waveform.samples)
      _log_progress(self.progress_every, index, frames)

    samples = np.concatenate(chunks, axis=0)
    waveform = audio_module.Waveform(
        samples, sample_rate=aig_bridge.MAGENTA_SAMPLE_RATE
    )
    return RenderedAudio(
        waveform=waveform,
        producer_metrics={
            "mode": "serial",
            "frames": frames,
            "prompt_embedded_in_hot_loop": False,
        },
    )

  def manifest_backend_info(self) -> JsonDict:
    info = super().manifest_backend_info()
    info.update({
        "bits": self.bits,
        "warmup_steps": self.warmup_steps,
    })
    return info


class RenderService:
  """Serialized render service around one loaded backend."""

  def __init__(self, config: WorkerConfig, renderer: LoadedRenderer):
    self._config = config
    self._renderer = renderer
    self._policy = PathPolicy(config.workspace_root, config.magenta_mount)
    self._lock = threading.Lock()

  def health(self) -> JsonDict:
    return {
        "ok": True,
        "ready": True,
        "backend": self._renderer.backend,
        "model": self._renderer.model,
        "checkpoint": self._renderer.checkpoint,
        "magenta_home": str(paths.magenta_home()),
        "magenta_mount": str(self._config.magenta_mount),
        "workspace_root": str(self._config.workspace_root),
        "device_info": self._renderer.device_info,
        "load_seconds": self._renderer.load_seconds,
    }

  def render_aig_adg(self, payload: Mapping[str, Any]) -> RenderResult:
    request = RenderRequest.from_json(payload, self._policy)
    with self._lock:
      return self._render_aig_adg_locked(request)

  def _render_aig_adg_locked(self, request: RenderRequest) -> RenderResult:
    packet_document = None
    atom_document = None
    full_adg_take = None
    if request.adg_bundle_manifest is not None:
      full_adg_take = adg_dataset.load_full_adg_take(
          request.adg_bundle_manifest
      )
      events = aig_bridge.load_pc4ms_adg_events(full_adg_take.adg_events)
      timeline = aig_bridge.load_pc4ms_adg_timeline(full_adg_take.adg_toml)
      packet_document = None
      schedule = aig_bridge.build_conditioning_schedule_from_pc4ms_adg_events(
          events,
          tempo_bpm=float(timeline["tempo_bpm"]),
          ppqn=int(timeline["ppqn"]),
          tail_seconds=request.tail_seconds,
          max_frames=request.max_frames,
      )
    else:
      assert request.gesture_packets is not None
      packet_document = aig_bridge.load_gesture_packets(request.gesture_packets)
      atom_document = aig_bridge.load_atom_specs(request.atom_specs)
      schedule = aig_bridge.build_conditioning_schedule(
          packet_document,
          tail_seconds=request.tail_seconds,
          max_frames=request.max_frames,
      )

    _ensure_parent(request.output)
    _ensure_parent(request.manifest)
    _ensure_parent(request.conditioning)
    np.savez_compressed(
        request.conditioning,
        notes=schedule.notes,
        drums=schedule.drums,
    )

    started = time.time()
    rendered = self._renderer.render_schedule(request, schedule)
    elapsed = time.time() - started
    waveform = rendered.waveform
    waveform.write(str(request.output))

    manifest = self._build_manifest(
        request,
        packet_document,
        atom_document,
        full_adg_take,
        schedule,
        waveform,
        rendered.producer_metrics,
        elapsed,
    )
    request.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    frames = int(schedule.summary["frames"])
    steps_per_second = _steps_per_second(frames, elapsed)
    response = {
        "ok": True,
        "backend": self._renderer.backend,
        "model": self._renderer.model,
        "frames": frames,
        "duration_seconds": float(schedule.summary["duration_seconds"]),
        "elapsed_seconds": elapsed,
        "steps_per_second": steps_per_second,
        "output": str(request.output),
        "manifest": str(request.manifest),
        "conditioning": str(request.conditioning),
    }
    return RenderResult(response=response, manifest=manifest)

  def _build_manifest(
      self,
      request: RenderRequest,
      packet_document: JsonDict | None,
      atom_document: JsonDict | None,
      full_adg_take: adg_dataset.FullAdgTake | None,
      schedule: aig_bridge.ConditioningSchedule,
      waveform: "audio.Waveform",
      producer_metrics: JsonDict,
      elapsed: float,
  ) -> JsonDict:
    frames = int(schedule.summary["frames"])
    atom_count = len(atom_document.get("atoms", [])) if atom_document else None
    source = _manifest_source(request, packet_document, atom_count, full_adg_take)
    return {
        "schema": "magenta_rt.render_worker.aig_adg.v1",
        "source": source,
        "magenta": {
            **self._renderer.manifest_backend_info(),
            "prompt": request.prompt,
            "temperature": request.temperature,
            "top_k": request.top_k,
            "cfg_musiccoca": request.cfg_musiccoca,
            "cfg_notes": request.cfg_notes,
            "cfg_drums": request.cfg_drums,
            "elapsed_seconds": elapsed,
            "steps_per_second": _steps_per_second(frames, elapsed),
        },
        "producer": producer_metrics,
        "conditioning": {
            **schedule.summary,
            "conditioning_npz": str(request.conditioning),
            "packet_reports": schedule.packet_reports,
        },
        "output": {
            "wav": str(request.output),
            "audio": aig_bridge.audio_stats(
                waveform.samples, waveform.sample_rate
            ),
        },
    }


class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
  daemon_threads = True


def make_handler(service: RenderService) -> type[http.server.BaseHTTPRequestHandler]:
  """Creates an HTTP handler bound to a render service."""

  class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "MagentaRTRenderWorker/1.0"

    def do_GET(self):  # pylint: disable=invalid-name
      if self.path != "/health":
        self._write_json({"ok": False, "error": "not found"}, HTTPStatus.NOT_FOUND)
        return
      self._write_json(service.health())

    def do_POST(self):  # pylint: disable=invalid-name
      if self.path != "/render/aig-adg":
        self._write_json({"ok": False, "error": "not found"}, HTTPStatus.NOT_FOUND)
        return
      try:
        payload = self._read_json_body()
        result = service.render_aig_adg(payload)
      except WorkerError as exc:
        self._write_json({"ok": False, "error": str(exc)}, exc.status)
        return
      except Exception as exc:  # pylint: disable=broad-exception-caught
        LOGGER.exception("render failed")
        self._write_json(
            {"ok": False, "error": str(exc)},
            HTTPStatus.INTERNAL_SERVER_ERROR,
        )
        return
      self._write_json(result.response)

    def log_message(self, fmt: str, *args):
      LOGGER.info("%s - %s", self.address_string(), fmt % args)

    def _read_json_body(self) -> JsonDict:
      raw_length = self.headers.get("Content-Length")
      if raw_length is None:
        raise BadRequest("Content-Length is required")
      try:
        length = int(raw_length)
      except ValueError as exc:
        raise BadRequest("Content-Length must be an integer") from exc
      if length <= 0:
        raise BadRequest("request body must not be empty")
      if length > 1_000_000:
        raise BadRequest("request body is too large")

      raw = self.rfile.read(length)
      try:
        payload = json.loads(raw.decode("utf-8"))
      except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BadRequest("request body must be valid JSON") from exc
      if not isinstance(payload, dict):
        raise BadRequest("request body must be a JSON object")
      return payload

    def _write_json(
        self, payload: Mapping[str, Any], status: HTTPStatus = HTTPStatus.OK
    ):
      body = json.dumps(payload, sort_keys=True).encode("utf-8") + b"\n"
      self.send_response(int(status))
      self.send_header("Content-Type", "application/json")
      self.send_header("Content-Length", str(len(body)))
      self.end_headers()
      self.wfile.write(body)

  return Handler


def validate_assets(config: WorkerConfig) -> None:
  """Fail fast on the model/resource files this worker needs."""

  magenta_home = paths.magenta_home()
  checkpoint_path = _checkpoint_path(config.checkpoint)
  required = [
      checkpoint_path,
      paths.musiccoca_dir() / "spm.model",
      paths.musiccoca_dir() / "text_encoder.tflite",
      paths.musiccoca_dir() / "audio_preprocessor.tflite",
      paths.musiccoca_dir() / "music_encoder.tflite",
      paths.musiccoca_dir() / "pretrained_vector_quantizer.tflite",
      paths.musiccoca_dir() / "mapper.tflite",
      paths.spectrostream_dir() / "encoder.safetensors",
      paths.spectrostream_dir() / "decoder.safetensors",
      paths.spectrostream_dir() / "quantizer.safetensors",
  ]
  missing = [str(path) for path in required if not path.exists()]
  if not magenta_home.exists():
    missing.insert(0, str(magenta_home))
  if missing:
    raise FileNotFoundError(
        "missing Magenta render assets:\n" + "\n".join(missing)
    )


def load_renderer(config: WorkerConfig) -> LoadedRenderer:
  if config.backend == "jax-gpu":
    return JaxGpuRenderer(config)
  if config.backend == "mlx-cpu":
    return MlxCpuRenderer(config)
  raise ValueError(f"unsupported backend: {config.backend}")


def run_server(config: WorkerConfig, service: RenderService) -> None:
  handler = make_handler(service)
  server = ThreadingHTTPServer((config.host, config.port), handler)
  LOGGER.info(
      "serving Magenta render worker backend=%s model=%s on %s:%d",
      config.backend,
      config.model,
      config.host,
      config.port,
  )
  try:
    server.serve_forever()
  finally:
    server.server_close()


def main() -> None:
  logging.basicConfig(
      level=os.environ.get("MAGENTA_RT_LOG_LEVEL", "INFO").upper(),
      format="%(asctime)s %(levelname)s %(name)s: %(message)s",
  )
  config = WorkerConfig.from_env()
  validate_assets(config)

  started = time.time()
  renderer = load_renderer(config)
  LOGGER.info(
      "loaded backend=%s model=%s checkpoint=%s in %.2fs",
      renderer.backend,
      renderer.model,
      renderer.checkpoint,
      time.time() - started,
  )
  run_server(config, RenderService(config, renderer))


def _absolute_path(value: Any, field: str) -> Path:
  if not isinstance(value, str) or not value:
    raise BadRequest(f"{field} must be an absolute path string")
  path = Path(value)
  if not path.is_absolute():
    raise BadRequest(f"{field} must be an absolute path")
  return path


def _is_under(path: Path, root: Path) -> bool:
  try:
    path.resolve(strict=False).relative_to(root.resolve(strict=False))
  except ValueError:
    return False
  return True


def _payload_float(
    payload: Mapping[str, Any],
    field: str,
    default: float,
    *,
    minimum: float | None = None,
) -> float:
  value = payload.get(field, default)
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise BadRequest(f"{field} must be a number")
  value = float(value)
  if minimum is not None and value < minimum:
    raise BadRequest(f"{field} must be >= {minimum}")
  return value


def _payload_int(
    payload: Mapping[str, Any],
    field: str,
    default: int,
    *,
    minimum: int | None = None,
) -> int:
  value = payload.get(field, default)
  if isinstance(value, bool) or not isinstance(value, int):
    raise BadRequest(f"{field} must be an integer")
  if minimum is not None and value < minimum:
    raise BadRequest(f"{field} must be >= {minimum}")
  return value


def _payload_optional_int(
    payload: Mapping[str, Any],
    field: str,
    *,
    minimum: int | None = None,
) -> int | None:
  if field not in payload or payload[field] is None:
    return None
  return _payload_int(payload, field, 0, minimum=minimum)


def _env_backend(name: str, default: BackendName) -> BackendName:
  value = os.environ.get(name, default)
  if value not in ("jax-gpu", "mlx-cpu"):
    raise ValueError(f"{name} must be one of: jax-gpu, mlx-cpu")
  return value


def _env_choice(name: str, default: str, choices: set[str]) -> str:
  value = os.environ.get(name, default)
  if value not in choices:
    expected = ", ".join(sorted(choices))
    raise ValueError(f"{name} must be one of: {expected}")
  return value


def _env_int(name: str, default: int, *, minimum: int) -> int:
  raw = os.environ.get(name)
  if raw is None:
    return default
  try:
    value = int(raw)
  except ValueError as exc:
    raise ValueError(f"{name} must be an integer") from exc
  if value < minimum:
    raise ValueError(f"{name} must be >= {minimum}")
  return value


def _checkpoint_path(checkpoint: str) -> Path:
  path = Path(checkpoint)
  if path.is_absolute():
    return path
  return paths.magenta_home() / "checkpoints" / checkpoint


def _jax_device_info(jax_module) -> JsonDict:
  devices = jax_module.devices()
  device_rows = []
  cuda_devices = []
  for device in devices:
    platform = getattr(device, "platform", "")
    row = {
        "id": getattr(device, "id", None),
        "platform": platform,
        "device_kind": getattr(device, "device_kind", ""),
        "description": str(device),
    }
    device_rows.append(row)
    if platform in ("cuda", "gpu") or "cuda" in str(device).lower():
      cuda_devices.append(row)
  return {
      "default_backend": jax_module.default_backend(),
      "devices": device_rows,
      "cuda_devices": cuda_devices,
  }


def _ensure_parent(path: Path) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)


def _manifest_source(
    request: RenderRequest,
    packet_document: JsonDict | None,
    atom_count: int | None,
    full_adg_take: adg_dataset.FullAdgTake | None,
) -> JsonDict:
  if full_adg_take is not None:
    return {
        "kind": "pc4ms_full_adg_take",
        "take_id": full_adg_take.take_id,
        "adg_bundle_manifest": str(full_adg_take.manifest),
        "adg_toml": str(full_adg_take.adg_toml),
        "adg_events": str(full_adg_take.adg_events),
        "adg_summary": str(full_adg_take.adg_summary),
        "midi": str(full_adg_take.midi),
    }
  assert packet_document is not None
  return {
      "kind": "aig_gesture_packets",
      "gesture_packets": str(request.gesture_packets),
      "atom_specs": str(request.atom_specs) if request.atom_specs else None,
      "packet_title": packet_document.get("title"),
      "packet_count": len(packet_document.get("packets", [])),
      "atom_count": atom_count,
  }


def _steps_per_second(frames: int, elapsed: float) -> float:
  if elapsed <= 0:
    return 0.0
  return frames / elapsed


def _log_progress(progress_every: int, index: int, frames: int) -> None:
  if progress_every > 0 and (index + 1) % progress_every == 0:
    LOGGER.info("generated %d/%d frames", index + 1, frames)


if __name__ == "__main__":
  main()
