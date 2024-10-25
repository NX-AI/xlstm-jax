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

SPMD Multihost Dataloading Utilities.

Adapted from Sholto's:
https://github.com/sholtodouglas/multihost_dataloading
"""

import logging
import time
from collections.abc import Iterable, Iterator
from functools import partial  # pylint: disable=g-importing-member
from typing import Any

import jax
import jax.tree_util as jtu
import numpy as np
import tensorflow as tf  # pylint: disable=g-import-not-at-top
from jax.sharding import Mesh, NamedSharding, PartitionSpec

LOGGER = logging.getLogger(__name__)


def _build_global_shape_and_sharding(
    local_shape: tuple[int, ...], global_mesh: Mesh
) -> tuple[tuple[int, ...], NamedSharding]:
    sharding = NamedSharding(global_mesh, PartitionSpec(global_mesh.axis_names))

    global_shape = (jax.process_count() * local_shape[0],) + local_shape[1:]

    return global_shape, sharding


def _form_global_array(path, array: np.ndarray, global_mesh: Mesh) -> jax.Array:
    """Put local sharded array into local devices"""
    global_shape, sharding = _build_global_shape_and_sharding(np.shape(array), global_mesh)

    try:
        local_device_arrays = np.split(array, len(global_mesh.local_devices), axis=0)
    except ValueError as array_split_error:
        raise ValueError(
            f"Unable to put to devices shape {array.shape} with "
            f"local device count {len(global_mesh.local_devices)} "
            f"at {jtu.keystr(path)}"
        ) from array_split_error

    local_device_buffers = jax.device_put(local_device_arrays, global_mesh.local_devices)
    return jax.make_array_from_single_device_arrays(global_shape, sharding, local_device_buffers)


def get_next_batch_sharded(local_iterator: Iterator, global_mesh: Mesh) -> jax.Array:
    """Splits the host loaded data equally over all devices."""

    SLEEP_TIME = 10
    MAX_DATA_LOAD_ATTEMPTS = 30

    data_load_attempts = 0
    loaded_data_success = False
    while not loaded_data_success and data_load_attempts < MAX_DATA_LOAD_ATTEMPTS:
        data_load_attempts += 1
        try:
            local_data = next(local_iterator)
            loaded_data_success = True
        except tf.errors.FailedPreconditionError:
            print("Failed to get next data batch, retrying")
            time.sleep(SLEEP_TIME)

    # Try one last time, if this fails we will see the full stack trace.
    if not loaded_data_success:
        local_data = next(local_iterator)

    input_gdas = jtu.tree_map_with_path(partial(_form_global_array, global_mesh=global_mesh), local_data)

    return input_gdas


class MultiHostDataLoadIterator:
    """
    Create a MultiHostDataLoadIterator.

    Wrapper around a :class:`tf.data.Dataset` or Iterable to iterate over data in a multi-host setup.
    Folds get_next_batch_sharded into an iterator class, and supports breaking indefinite iterator into epochs.

    Args:
        dataloader: The dataloader to iterate over.
        global_mesh: The mesh to shard the data over.
        iterator_length: The length of the iterator. If provided, the iterator will stop after this many steps with a
            :class:`StopIteration` exception. Otherwise, will continue over the iterator until it raises an exception
            itself.
        dataset_size: size of the dataset. If provided, will be returned by get_dataset_size. Otherwise, will return
            `None`. Can be used to communicate the dataset size to functions that use the iterator.
        reset_after_epoch: Whether to reset the iterator between epochs or not. If `True`, the iterator will reset
            after each epoch, otherwise it will continue from where it left off. If you have an indefinite iterator
            (e.g. train iterator with grain and shuffle), this should be set to `False`. For un-shuffled iterators in
            grain (e.g. validation), this should be set to `True`.
    """

    def __init__(
        self,
        dataloader: tf.data.Dataset | Iterable,
        global_mesh: Mesh,
        iterator_length: int | None = None,
        dataset_size: int | None = None,
        reset_after_epoch: bool = False,
    ):
        self.global_mesh = global_mesh
        self.dataloader = dataloader
        self.iterator_length = iterator_length
        self.dataset_size = dataset_size
        self.reset_after_epoch = reset_after_epoch
        self.state_set = False
        self.step_counter = 0
        if isinstance(self.dataloader, tf.data.Dataset):
            self.local_iterator = self.dataloader.as_numpy_iterator()
        elif isinstance(self.dataloader, Iterable):
            self.local_iterator = iter(self.dataloader)
        else:
            raise ValueError("Type error: dataloader should be either tf.data.Dataset or Iterable.")

    def reset(self):
        self.step_counter = 0
        if self.reset_after_epoch:
            if isinstance(self.dataloader, tf.data.Dataset):
                self.local_iterator = self.dataloader.as_numpy_iterator()
            elif isinstance(self.dataloader, Iterable):
                self.local_iterator = iter(self.dataloader)
            else:
                raise ValueError("Type error: dataloader should be either tf.data.Dataset or grain.DataLoader.")

    def get_state(self) -> dict[str, Any]:
        state = {"step": self.step_counter, "iterator_length": self.iterator_length}
        if hasattr(self.local_iterator, "get_state"):
            state["iterator"] = self.local_iterator.get_state()
        LOGGER.info(f"Getting Dataloader State: {state}")
        return state

    def set_state(self, state: dict[str, Any]):
        LOGGER.info(f"Setting Dataloader State: {state}")
        assert (
            self.iterator_length == state["iterator_length"]
        ), "The iterator length in the state differs from the current iterator length. Cannot load state."
        if hasattr(self.local_iterator, "set_state"):
            self.step_counter = int(state["step"])
            self.local_iterator.set_state(state["iterator"])
            self.state_set = True
        else:
            LOGGER.warning("The local iterator has no `set_state` method. Skipping setting the state.")
            self.state_set = False

    def __iter__(self):
        if not self.state_set or self.step_counter >= self.iterator_length:
            # If we previously set the state, we do not want to reset it, except if we would have done it anyway.
            self.reset()
        self.state_set = False
        return self

    def __len__(self):
        return self.iterator_length if self.iterator_length is not None else float("inf")

    def __next__(self):
        if self.iterator_length is not None and self.step_counter >= self.iterator_length:
            raise StopIteration
        self.step_counter += 1
        return get_next_batch_sharded(self.local_iterator, self.global_mesh)
