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

"""Checkpoint loading helpers shared by MLX weight loaders."""

import os

import flax.traverse_util as flaxtu
import safetensors.numpy as safetensors_numpy


def load_jax_params(path):
  """Load a JAX-named safetensors checkpoint as nested numpy arrays."""
  flat_weights = safetensors_numpy.load_file(os.fspath(path))
  nested_dict = {tuple(k.split('/')): v for k, v in flat_weights.items()}
  return flaxtu.unflatten_dict(nested_dict)
