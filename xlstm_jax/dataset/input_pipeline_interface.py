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

This file is a modified version of the file input_pipeline_interface.py from the maxtext project
https://github.com/AI-Hypercomputer/maxtext/blob/main/MaxText/input_pipeline/input_pipeline_interface.py.

Input pipeline
"""

import logging

import jax
from jax.sharding import Mesh, PartitionSpec as P

from .configs import DataConfig, GrainArrayRecordsDataConfig, HFHubDataConfig, SyntheticDataConfig
from .multihost_dataloading import MultiHostDataLoadIterator
from .synthetic_dataloading import SyntheticDataIterator

LOGGER = logging.getLogger(__name__)

try:
    from .grain_data_processing import make_grain_iterator

    GRAIN_AVAILABLE = True
except ImportError:
    LOGGER.warning("Grain not found, multi-host data loading and non-synthetic dataset will be disabled.")
    # Set variable to remind user of this.
    GRAIN_AVAILABLE = False


    def make_grain_iterator(*args, **kwargs):
        raise NotImplementedError("Grain not found, multi-host data loading and non-synthetic dataset is disabled.")

DataIterator = MultiHostDataLoadIterator | SyntheticDataIterator


def get_process_loading_data(config: DataConfig, mesh: Mesh) -> list[int]:
    """Get list of processes loading data.

    Args:
      config: Config of the dataset to load.
      mesh: Global device mesh for sharding.

    Returns:
      List of process indices that will load real data.
    """
    sharding = jax.sharding.NamedSharding(mesh, P(mesh.axis_names))
    devices_indices_map = sharding.devices_indices_map((config.global_batch_size, config.max_target_length))
    process_loading_data = set()
    for p, indices in devices_indices_map.items():
        if indices[0].stop is None or indices[0].stop <= config.global_batch_size:
            process_loading_data.add(p.process_index)
    return list(process_loading_data)


def create_data_iterator(config: DataConfig, mesh: Mesh) -> DataIterator:
    if isinstance(config, SyntheticDataConfig):
        return SyntheticDataIterator(config, mesh)
    if isinstance(config, (HFHubDataConfig, GrainArrayRecordsDataConfig)):
        if not GRAIN_AVAILABLE:  # Was checked before during import, but just in case.
            raise NotImplementedError("Grain is not available, multi-host data loading are disabled. Exiting.")
        process_indices = get_process_loading_data(config, mesh)
        if jax.process_index() not in process_indices:
            raise NotImplementedError(
                f"Current process {jax.process_index()} not in process_indices {process_indices}. "
                f"Global batch size must be larger than the number of hosts. Hosts not loading data is not supported."
            )
        return make_grain_iterator(config, mesh, process_indices)
    raise NotImplementedError(f"Unknown dataset_type {type(config)}, dataset_type must be synthetic or hf.")


def create_mixed_data_iterator(
        configs: list[HFHubDataConfig | GrainArrayRecordsDataConfig], mesh: Mesh,
        dataset_weights: list[float] | None = None
) -> DataIterator:
    """Create a data iterator that mixes multiple datasets.

    Each individual dataset will be loaded, and the iterator will return batches where each batch element is from
    one of the datasets. The frequency of each dataset is determined by the dataset_weights.

    Args:
        configs: List of DataConfig objects, determining the datasets to load.
        mesh: JAX mesh object. Used to distribute the data over multiple devices.
        dataset_weights: Mixing weights for the datasets. If None, all datasets will have equal weight.

    Returns:
        DataIterator object that can be used to iterate over the mixed dataset.
    """
    if not GRAIN_AVAILABLE:  # Was checked before during import, but just in case.
        raise NotImplementedError("Grain is not available, multi-host data loading are disabled. Exiting.")
    process_indices = get_process_loading_data(configs[0], mesh)
    if jax.process_index() not in process_indices:
        raise NotImplementedError(
            f"Current process {jax.process_index()} not in process_indices {process_indices}. "
            f"Global batch size must be larger than the number of hosts. Hosts not loading data is not supported."
        )
    return make_grain_iterator(configs, mesh, process_indices, dataset_weights)
