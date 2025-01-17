"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

---

This file is a modified version of the file input_pipeline_interface.py from the maxtext project,
see https://github.com/AI-Hypercomputer/maxtext/MaxText/input_pipeline/input_pipeline_interface.py.

Synthetic Data Iterator
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from xlstm_jax.dataset.batch import LLMBatch

from .configs import SyntheticDataConfig


class SyntheticDataIterator:
    """Creates a synthetic data iterator for performance testing work.

    Args:
        config: Configuration for the synthetic data.
        mesh: Global device mesh for sharding.
    """

    def __init__(self, config: SyntheticDataConfig, mesh: Mesh):
        self.config = config
        self.mesh = mesh
        self.num_batches = config.num_batches
        data_pspec = P(mesh.axis_names)
        data_pspec_shardings = jax.tree_util.tree_map(lambda p: NamedSharding(mesh, p), data_pspec)
        self.data_generator = jax.jit(
            partial(SyntheticDataIterator.raw_generate_synthetic_data, config=self.config),
            out_shardings=data_pspec_shardings,
        )
        self._step_counter = 0

    def __iter__(self):
        self._step_counter = 0
        return self

    def __next__(self):
        if self._step_counter >= self.num_batches:
            raise StopIteration
        self._step_counter += 1
        with self.mesh:
            return self.data_generator()

    def __len__(self) -> int:
        return self.num_batches

    @property
    def dataset_size(self) -> int:
        return self.num_batches * self.config.global_batch_size

    @staticmethod
    def raw_generate_synthetic_data(config: SyntheticDataConfig) -> LLMBatch:
        """Generates a single batch of synthetic data"""
        output = LLMBatch(
            inputs=jnp.zeros((config.global_batch_size, config.max_target_length), dtype=jnp.int32),
            inputs_position=jnp.zeros((config.global_batch_size, config.max_target_length), dtype=jnp.int32),
            inputs_segmentation=jnp.ones((config.global_batch_size, config.max_target_length), dtype=jnp.int32),
            targets=jnp.zeros((config.global_batch_size, config.max_target_length), dtype=jnp.int32),
            targets_position=jnp.zeros((config.global_batch_size, config.max_target_length), dtype=jnp.int32),
            targets_segmentation=jnp.ones((config.global_batch_size, config.max_target_length), dtype=jnp.int32),
        )
        return output
