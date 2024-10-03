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

LLM Data Iterator with Grain
"""

import math
from typing import Any

import grain.python as grain
from jax.sharding import Mesh

from . import grain_transforms
from .batch import LLMBatch
from .multihost_dataloading import MultiHostDataLoadIterator


def make_grain_llm_iterator(
    dataloading_host_index: int,
    dataloading_host_count: int,
    global_mesh: Mesh,
    dataset: Any,
    global_batch_size: int,
    max_target_length: int,
    shuffle: bool,
    data_shuffle_seed: int,
    num_epochs: int,
    operations: list[grain.MapTransform] | None = None,
    grain_packing: bool = False,
    shift: bool = True,
    worker_count: int = 1,
    worker_buffer_size: int = 1,
    drop_remainder: bool = True,
    reset_after_epoch: bool = False,
) -> MultiHostDataLoadIterator:
    """
    Create a multi-host dataloader with grain for LLM training.

    The dataloader will perform batch packing, padding, and shifting of the data to create
    a batch of LLMBatch objects. The LLMBatch object will contain the input and target data
    for the model.

    Args:
        dataloading_host_index: The index of the dataloading host. Will be used to select the
            correct shard of the dataset. In JAX, this is equivalent to :func:`jax.process_index()`.
        dataloading_host_count: The number of dataloading hosts. Will be used to determine the
            shard size. In JAX, this is equivalent to :func:`jax.process_count()`.
        global_mesh: The global mesh to shard the data over.
        dataset: The dataset to load. Should provide a `__getitem__` method to access elements.
        global_batch_size: The global batch size.
        max_target_length: The maximum target length.
        shuffle: Whether to shuffle the dataset. If you want a different shuffle order each
            epoch, you need to provide the number of epochs in `num_epochs`.
        data_shuffle_seed: The shuffle seed.
        num_epochs: The number of epochs to train for. The dataset will be repeated for so
            many epochs, and the shuffle order will be different for each epoch. Note that
            batches of an epoch can spill over into the first batch of the next epoch, to
            avoid dropping data. The argument `drop_remainder` controls whether the very last
            batch of all epochs together is dropped.
        operations: A list of `grain` operations to apply to the dataset before batching.
        grain_packing: Whether to perform packing of the data. This is useful for datasets
            with a lot of padding, as batch elements will be packed together in a sequence
            to reduce the amount of padding. This can improve throughput efficiency. NOTE:
            if packing is enabled, the length of the iterator cannot be determined in advance
            and is likely incorrect in the iterator (will be set to maximum number of batches).
        shift: Whether to shift the input data to create the target data.
        worker_count: The number of workers to use. In `grain`, a single worker is usually
            sufficient, as the data loading is done in parallel across hots.
        worker_buffer_size: The buffer size for the workers.
        drop_remainder: Whether to drop the remainder of the dataset. Note that in case of
            providing a number of epochs, the last batch of all epochs together will be
            dropped if this is set to `True`. If set to `False`, the last batch of all epochs
            together will be included in the iterator.
        reset_after_epoch: Whether to reset the iterator after each epoch. If set to `True`,
            the iterator will start from the beginning of the dataset after each epoch. If set
            to `False`, the iterator will continue from where it left off in the dataset. Note
            that resetting the iterator can be expensive in a multi-host setup and can fail if
            the multi-processing pool could not be set up.
    """
    if operations is None:
        operations = []

    if grain_packing:
        operations.append(
            grain.experimental.PackAndBatchOperation(
                batch_size=global_batch_size // dataloading_host_count,
                length_struct={"inputs": max_target_length, "targets": max_target_length},
            )
        )
        operations.append(grain_transforms.ReformatPacking())
    else:
        operations.append(grain_transforms.PadToMaxLength(max_target_length))
        operations.append(
            grain.Batch(batch_size=global_batch_size // dataloading_host_count, drop_remainder=drop_remainder)
        )

    if shift:
        operations.append(grain_transforms.ShiftData(axis=1))

    operations.append(grain_transforms.CollacteToBatch(batch_class=LLMBatch))

    index_sampler = grain.IndexSampler(
        num_records=len(dataset),
        num_epochs=num_epochs,
        shard_options=grain.ShardOptions(
            shard_index=dataloading_host_index, shard_count=dataloading_host_count, drop_remainder=drop_remainder
        ),
        shuffle=shuffle,
        seed=data_shuffle_seed,
    )

    dataloader = grain.DataLoader(
        data_source=dataset,
        operations=operations,
        sampler=index_sampler,
        worker_count=worker_count,
        worker_buffer_size=worker_buffer_size,
    )

    if drop_remainder:
        iterator_length = len(dataset) // global_batch_size
    else:
        iterator_length = int(math.ceil(len(dataset) / global_batch_size))

    multihost_gen = MultiHostDataLoadIterator(
        dataloader,
        global_mesh,
        iterator_length=iterator_length,
        dataset_size=len(dataset),
        reset_after_epoch=reset_after_epoch,
    )
    return multihost_gen
