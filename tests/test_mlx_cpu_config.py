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

"""Tests for Linux-friendly MLX CPU configuration."""

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import numpy.testing as npt
import safetensors.numpy as safetensors_numpy

from magenta_rt.mlx.checkpoint_io import load_jax_params
from magenta_rt.mlx.generate import (
    _prepare_cpu_runtime_env,
    _resolve_generation_options,
)


class TestMlxGenerationOptions(unittest.TestCase):

  def test_cpu_defaults_to_raw_checkpoint_without_warmup(self):
    use_mlxfn, warmup_steps = _resolve_generation_options(
        device='cpu', use_mlxfn=None, warmup_steps=None
    )
    self.assertFalse(use_mlxfn)
    self.assertEqual(warmup_steps, 0)

  def test_cpu_rejects_explicit_mlxfn(self):
    with self.assertRaisesRegex(ValueError, '--no-mlxfn'):
      _resolve_generation_options(
          device='cpu', use_mlxfn=True, warmup_steps=None
      )

  def test_auto_keeps_mlxfn_default_and_warmup(self):
    use_mlxfn, warmup_steps = _resolve_generation_options(
        device='auto', use_mlxfn=None, warmup_steps=None
    )
    self.assertTrue(use_mlxfn)
    self.assertEqual(warmup_steps, 5)

  def test_cpu_sets_disable_compile_before_mlx_import(self):
    old_disable = os.environ.pop('_MLX_DISABLE_COMPILE', None)
    old_opt_in = os.environ.pop('MAGENTA_RT_MLX_CPU_COMPILE', None)
    old_wrapper = os.environ.get('MAGENTA_RT_MLX_CPU_GPP_WRAPPER')
    old_path = os.environ.get('PATH')
    try:
      os.environ['MAGENTA_RT_MLX_CPU_GPP_WRAPPER'] = '0'
      _prepare_cpu_runtime_env('cpu')
      self.assertEqual(os.environ.get('_MLX_DISABLE_COMPILE'), '1')
    finally:
      if old_path is not None:
        os.environ['PATH'] = old_path
      if old_disable is not None:
        os.environ['_MLX_DISABLE_COMPILE'] = old_disable
      else:
        os.environ.pop('_MLX_DISABLE_COMPILE', None)
      if old_opt_in is not None:
        os.environ['MAGENTA_RT_MLX_CPU_COMPILE'] = old_opt_in
      else:
        os.environ.pop('MAGENTA_RT_MLX_CPU_COMPILE', None)
      if old_wrapper is not None:
        os.environ['MAGENTA_RT_MLX_CPU_GPP_WRAPPER'] = old_wrapper
      else:
        os.environ.pop('MAGENTA_RT_MLX_CPU_GPP_WRAPPER', None)


class TestCheckpointIo(unittest.TestCase):

  def test_load_jax_params_returns_nested_numpy_arrays(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      path = Path(tmpdir) / 'checkpoint.safetensors'
      expected = np.arange(6, dtype=np.float32).reshape(2, 3)
      safetensors_numpy.save_file(
          {'params/depthformer/kernel': expected}, path
      )

      params = load_jax_params(path)
      actual = params['params']['depthformer']['kernel']

    self.assertIsInstance(actual, np.ndarray)
    npt.assert_array_equal(actual, expected)
