"""Copyright 2023 Google LLC.

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

import logging
import math
from typing import Any

import grain.python as grain
from grain._src.python.dataset.transformations.packing import FirstFitPackIterDataset
from grain._src.python.dataset.transformations.prefetch import ThreadPrefetchIterDataset
from jax.sharding import Mesh

from . import grain_transforms
from .batch import LLMBatch
from .multihost_dataloading import MultiHostDataLoadIterator

LOGGER = logging.getLogger(__name__)


def make_grain_llm_iterator(
    dataloading_host_index: int,
    dataloading_host_count: int,
    global_mesh: Mesh,
    dataset: Any,
    global_batch_size: int,
    max_target_length: int,
    shuffle: bool,
    data_shuffle_seed: int,
    num_epochs: int | None = None,
    operations: list[grain.MapTransform] | None = None,
    grain_packing: bool = False,
    grain_packing_bin_count: int | None = None,
    shift: bool = True,
    shift_target: bool = True,
    eod_token_id: int = 0,
    worker_count: int = 1,
    worker_buffer_size: int = 1,
    apply_padding: bool = True,
    drop_remainder: bool = True,
    batch_class: type = LLMBatch,
    reset_after_epoch: bool = False,
    use_thread_prefetch: bool = False,
) -> MultiHostDataLoadIterator:
    """Create a multi-host dataloader for LLM training.

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
        shuffle: Whether to shuffle the dataset.
        data_shuffle_seed: The shuffle seed.
        num_epochs: The number of epochs to train for. The dataset will be repeated for so
            many epochs, and the shuffle order will be different for each epoch. If None,
            the dataset will be repeated infinitely. Note that batches of an epoch can
            spill over into the first batch of the next epoch, to avoid dropping data.
            The argument `drop_remainder` controls whether the very last batch of all epochs
            together is dropped.
        operations: A list of `grain` operations to apply to the dataset before batching.
        grain_packing: Whether to perform packing of the data. This is useful for datasets
            with a lot of padding, as batch elements will be packed together in a sequence
            to reduce the amount of padding. This can improve throughput efficiency. NOTE:
            if packing is enabled, the length of the iterator cannot be determined in advance
            and is likely incorrect in the iterator (will be set to maximum number of batches).
        grain_packing_bin_count: The number of packing bins to use. If not provided, the
            bin count will be set to the batch size. It can be beneficial to increase the packing
            bins to reduce padding.
        shift: Whether to shift the input data to create the target data.
        shift_target: Whether to shift the targets left (True) or inputs right (False).
        eod_token_id: The token ID to use for the end-of-document token. Used if shifting the
            the inputs right and adding an end-of-document token to the sequence. If not
            provided, the default value of 0 will be used. Recommended to set this to a value
            explicitly with the tokenizer's EOD token ID.
        worker_count: The number of workers to use. In `grain`, a single worker is usually
            sufficient, as the data loading is done in parallel across hosts.
        worker_buffer_size: The buffer size for the workers.
        apply_padding:  Pad sequence to the maximum target length.
        drop_remainder: Whether to drop the remainder of the dataset. Note that in case of
            providing a number of epochs, the last batch of all epochs together will be
            dropped if this is set to `True`. If set to `False`, the last batch of all epochs
            together will be included in the iterator.
        batch_class: Batch class used to collate data to batch.
        reset_after_epoch: Whether to reset the iterator after each epoch. If set to `True`,
            the iterator will start from the beginning of the dataset after each epoch. If set
            to `False`, the iterator will continue from where it left off in the dataset. Note
            that resetting the iterator can be expensive in a multi-host setup and can fail if
            the multi-processing pool could not be set up.
        use_thread_prefetch: Whether to use thread prefetching instead of multi-processing.

    Returns:
        A :class:`MultiHostDataLoadIterator` object that can be used to iterate over the dataset.
    """
    # Convert dataset to MapDataset.
    grain_dataset = grain.MapDataset.source(dataset)
    # Shard dataset. We do this by slicing the dataset into the correct shard for the host.
    # If the dataset is not evenly divisible by the number of hosts, the first N hosts should
    # have one more element than the rest. This is to ensure that the dataset is evenly divided
    # across all hosts.
    dataset_size = len(grain_dataset)
    slice_size = dataset_size // dataloading_host_count
    slice_remainder = dataset_size % dataloading_host_count
    host_slice_start = dataloading_host_index * slice_size + min(dataloading_host_index, slice_remainder)
    host_slice_end = host_slice_start + slice_size + (1 if dataloading_host_index < slice_remainder else 0)
    host_slice = slice(host_slice_start, host_slice_end)
    grain_dataset = grain_dataset.slice(host_slice)
    LOGGER.info(f"Host {dataloading_host_index} has slice {host_slice} with global dataset size {dataset_size}.")
    # Set seed. This includes the seed for the shuffling and random operations. We add the host index to the seed to
    # ensure different seeds for different hosts.
    grain_dataset = grain_dataset.seed(data_shuffle_seed + dataloading_host_index)
    # Shuffle dataset. This shuffling already supports different seeds for different epochs.
    if shuffle:
        grain_dataset = grain_dataset.shuffle()
    # Repeat the dataset for the number of epochs. Note that this simply forwards the index to the parent dataset.
    if num_epochs is None or num_epochs > 0:
        grain_dataset = grain_dataset.repeat(num_epochs)
    # Apply operations.
    if operations is not None:
        for operation in operations:
            grain_dataset = grain_dataset.map(operation)
    # Convert dataset to iter dataset. Most operations work on either map and iter, except for packing, which requires
    # iter.
    LOGGER.info(f"Host {dataloading_host_index} has dataset length {len(grain_dataset)}.")
    grain_dataset = grain_dataset.to_iter_dataset()
    # Pack or batch dataset.
    if grain_packing:
        # Number of packing bins is independent of batch size in lazy API.
        if grain_packing_bin_count is None:
            grain_packing_bin_count = global_batch_size // dataloading_host_count
        grain_dataset = FirstFitPackIterDataset(
            grain_dataset,
            length_struct={"inputs": max_target_length, "targets": max_target_length},
            num_packing_bins=grain_packing_bin_count,
            shuffle_bins=shuffle,
        )
        LOGGER.info(f"Packing dataset with {grain_packing_bin_count} bins.")
        # Packing returns each element one by one. We batch them together separately.
        grain_dataset = grain_dataset.batch(global_batch_size // dataloading_host_count, drop_remainder=drop_remainder)
        # The output is already in the correct structure, but some keys need renamings to be consistent with the
        # old API.
        grain_dataset = grain_dataset.map(grain_transforms.ReformatLazyPacking())
    else:
        if apply_padding:
            grain_dataset = grain_dataset.map(grain_transforms.PadToMaxLength(max_target_length))
        grain_dataset = grain_dataset.batch(global_batch_size // dataloading_host_count, drop_remainder=drop_remainder)

    # Create targets by shifting.
    if shift:
        grain_dataset = grain_dataset.map(
            grain_transforms.ShiftData(axis=1, shift_target=shift_target, eod_token_id=eod_token_id)
        )

    if not grain_packing:
        grain_dataset = grain_dataset.map(grain_transforms.InferSegmentations(eod_token_id=eod_token_id))

    # Create LLMBatch objects.
    grain_dataset = grain_dataset.map(grain_transforms.CollateToBatch(batch_class=batch_class))

    # Setup prefetching with thread or multiprocessing.
    multiprocessing_options = None
    if use_thread_prefetch:
        LOGGER.info("Using thread prefetching.")
        grain_dataset = ThreadPrefetchIterDataset(
            grain_dataset,
            prefetch_buffer_size=int(worker_buffer_size * worker_count),
        )
    elif worker_count > 0:
        LOGGER.info(f"Using multiprocessing with {worker_count} workers.")
        multiprocessing_options = grain.MultiprocessingOptions(
            num_workers=worker_count,
            per_worker_buffer_size=worker_buffer_size,
        )
        # NOTE: At the moment this gives a warning about being deprecated. In the nightly version of
        # grain, this will be under the name `grain_dataset.mp_prefetch`.
        grain_dataset = grain_dataset.prefetch(multiprocessing_options)

    LOGGER.info(f"Final grain dataset: {grain_dataset}")

    if drop_remainder:
        iterator_length = len(dataset) // global_batch_size
    else:
        iterator_length = int(math.ceil(len(dataset) / global_batch_size))

    multihost_gen = MultiHostDataLoadIterator(
        grain_dataset,
        global_mesh,
        iterator_length=iterator_length,
        dataset_size=len(dataset),
        reset_after_epoch=reset_after_epoch,
    )
    return multihost_gen
