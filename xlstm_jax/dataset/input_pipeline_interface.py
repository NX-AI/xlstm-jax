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

Input pipeline
"""

import logging

import jax
from jax.sharding import Mesh, PartitionSpec as P

from .configs import DataConfig, HFDataConfig, SyntheticDataConfig
from .multihost_dataloading import MultiHostDataLoadIterator
from .synthetic_dataloading import SyntheticDataIterator

LOGGER = logging.getLogger(__name__)

try:
    from .hf_data_processing import make_hf_iterator
except ImportError:
    LOGGER.warning("Grain not found, multi-host data loading and non-synthetic dataset will be disabled.")

    def make_hf_iterator(*args, **kwargs):
        raise NotImplementedError("Grain not found, multi-host data loading and non-synthetic dataset is disabled.")


DataIterator = MultiHostDataLoadIterator | SyntheticDataIterator


def get_process_loading_real_data(config: DataConfig, mesh: Mesh) -> list[int]:
    """Get list of processes loading data when expansion_factor_real_data != -1

    Args:
      config: Config of the dataset to load.
      mesh: Global device mesh for sharding.

    Returns:
      List of process indices that will load real data.
    """
    sharding = jax.sharding.NamedSharding(mesh, P(mesh.axis_names))
    devices_indices_map = sharding.devices_indices_map((config.global_batch_size, config.max_target_length))
    if config.global_batch_size_to_train_on > 0:
        batch_cutoff = config.global_batch_size_to_train_on
    else:
        batch_cutoff = config.global_batch_size
    process_loading_real_data = set()
    for p, indices in devices_indices_map.items():
        if indices[0].stop is None or indices[0].stop <= batch_cutoff:
            process_loading_real_data.add(p.process_index)
    return list(process_loading_real_data)


def make_mixed_train_iterator(config: DataConfig, mesh: Mesh) -> tuple[DataIterator, DataIterator | None]:
    """Return iterators according to dataset_type"""
    process_indices = get_process_loading_real_data(config, mesh)
    if config.expansion_factor_real_data != -1:  # assert number of hosts loading real data
        assert len(process_indices) == jax.process_count() // config.expansion_factor_real_data
    if jax.process_index() in process_indices:
        if isinstance(config, HFDataConfig):
            return make_hf_iterator(config, mesh, process_indices)
        else:
            raise NotImplementedError(f"Data loading for dataset type {type(config)} not implemented yet.")
    else:
        raise NotImplementedError(
            "Processes loading fake data not implemented yet. You are likely attempting per-device batch sizes of <1, which are not supported yet."
        )


def create_data_iterator(config: DataConfig, mesh: Mesh) -> tuple[DataIterator, DataIterator | None]:
    if isinstance(config, SyntheticDataConfig):
        return SyntheticDataIterator(config, mesh, "train"), SyntheticDataIterator(config, mesh, "val")
    elif isinstance(config, HFDataConfig):
        return make_mixed_train_iterator(config, mesh)
    else:
        raise NotImplementedError(f"Unknown dataset_type {type(config)}, dataset_type must be synthetic or hf.")
