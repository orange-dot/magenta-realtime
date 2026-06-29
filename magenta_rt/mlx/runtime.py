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

"""Runtime environment helpers for MLX entry points."""

import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Literal

MlxDevice = Literal['auto', 'cpu', 'gpu']


def prepare_cpu_runtime_env(device: MlxDevice):
  """Prepare process environment before MLX CPU generation."""
  if device != 'cpu':
    return
  if os.environ.get('MAGENTA_RT_MLX_CPU_COMPILE') != '1':
    os.environ.setdefault('_MLX_DISABLE_COMPILE', '1')
  if sys.platform.startswith('linux'):
    _ensure_linux_gpp_wrapper()


def _ensure_linux_gpp_wrapper():
  """Add -fpermissive to MLX's hard-coded Linux CPU JIT g++ invocation."""
  if os.environ.get('MAGENTA_RT_MLX_CPU_GPP_WRAPPER') == '0':
    return

  wrapper_dir = Path(tempfile.gettempdir()) / (
      f'magenta-rt-mlx-gpp-wrapper-{os.getuid()}'
  )
  wrapper_path = wrapper_dir / 'g++'
  path_entries = [
      entry for entry in os.environ.get('PATH', '').split(os.pathsep)
      if entry and Path(entry).resolve() != wrapper_dir.resolve()
  ]
  real_gpp = shutil.which('g++', path=os.pathsep.join(path_entries))
  if real_gpp is None:
    return

  wrapper_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
  wrapper_body = (
      '#!/usr/bin/env sh\n'
      f'exec "{real_gpp}" -fpermissive "$@"\n'
  )
  if not wrapper_path.exists() or wrapper_path.read_text() != wrapper_body:
    wrapper_path.write_text(wrapper_body)
    wrapper_path.chmod(0o700)

  current_path = os.environ.get('PATH', '')
  current_entries = current_path.split(os.pathsep) if current_path else []
  if (
      not current_entries
      or Path(current_entries[0]).resolve() != wrapper_dir.resolve()
  ):
    os.environ['PATH'] = os.pathsep.join([str(wrapper_dir)] + current_entries)
