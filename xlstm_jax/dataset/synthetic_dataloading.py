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

Synthetic Data Iterator
"""

from typing import Literal

import jax
from jax.sharding import Mesh, PartitionSpec as P

from xlstm_jax.dataset.batch import LLMBatch

from .configs import SyntheticDataConfig


class SyntheticDataIterator:
    """Creates a synthetic data iterator for performance testing work"""

    def __init__(self, config: SyntheticDataConfig, mesh: Mesh, mode: Literal["train", "val"] = "train"):
        self.config = config
        self.mesh = mesh
        self.mode = mode
        self.num_batches = config.num_train_batches if mode == "train" else config.num_val_batches
        data_pspec = P(mesh.axis_names)
        data_pspec_shardings = jax.tree_util.tree_map(lambda p: jax.sharding.NamedSharding(mesh, p), data_pspec)
        self.data_generator = jax.jit(
            SyntheticDataIterator.raw_generate_synthetic_data, out_shardings=data_pspec_shardings, static_argnums=0
        )

    def __iter__(self):
        self.step_counter = 0
        return self

    def __next__(self):
        if self.step_counter >= self.num_batches:
            raise StopIteration
        self.step_counter += 1
        with self.mesh:
            return self.data_generator(self.config)

    def __len__(self):
        return self.num_batches

    @staticmethod
    def raw_generate_synthetic_data(config: SyntheticDataConfig):
        """Generates a single batch of synthetic data"""
        output = LLMBatch(
            inputs=jax.numpy.zeros((config.global_batch_size, config.max_target_length), dtype=jax.numpy.int32),
            inputs_position=jax.numpy.zeros(
                (config.global_batch_size, config.max_target_length), dtype=jax.numpy.int32
            ),
            inputs_segmentation=jax.numpy.ones(
                (config.global_batch_size, config.max_target_length), dtype=jax.numpy.int32
            ),
            targets=jax.numpy.zeros((config.global_batch_size, config.max_target_length), dtype=jax.numpy.int32),
            targets_position=jax.numpy.zeros(
                (config.global_batch_size, config.max_target_length), dtype=jax.numpy.int32
            ),
            targets_segmentation=jax.numpy.ones(
                (config.global_batch_size, config.max_target_length), dtype=jax.numpy.int32
            ),
        )

        return output
