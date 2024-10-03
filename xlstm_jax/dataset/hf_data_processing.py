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

Input pipeline using Huggingface datasets.
"""

import logging
import math
from collections.abc import Sequence
from functools import partial
from itertools import chain
from typing import Any, SupportsIndex

import datasets
import grain.python as grain
import jax
import transformers
from jax.sharding import Mesh

from xlstm_jax.dataset import grain_transforms

from .configs import HFHubDataConfig, HFLocalDataConfig
from .grain_iterator import make_grain_llm_iterator
from .multihost_dataloading import MultiHostDataLoadIterator

LOGGER = logging.getLogger(__name__)


class PaddedDataset(grain.RandomAccessDataSource):
    """Dataset wrapper to pad the dataset to be a multiple of the global batch size."""

    def __init__(self, dataset: datasets.Dataset, full_dataset_length: int, column_name: str):
        """
        Initializes the PaddedDataset.

        Args:
            dataset: The dataset to pad.
            full_dataset_length: The full dataset length, including padding. Should be a multiple of the global
                batch size with which the dataset is loaded. Also lengths smaller than the dataset are supported
                and will return the smaller length.
            column_name: The column name in the dataset that is returned.
        """
        self.dataset = dataset
        self.full_dataset_length = full_dataset_length
        self.column_name = column_name

    def __len__(self):
        """Returns the full dataset length."""
        return self.full_dataset_length

    def __getitem__(self, record_key: SupportsIndex) -> Any:
        """Returns padding if the record key is out of bounds, otherwise returns the dataset record."""
        if record_key >= len(self.dataset):
            return {self.column_name: []}
        else:
            return self.dataset[record_key]

    def __repr__(self):
        return (
            f"PaddedDataset({self.dataset}, full_dataset_length={self.full_dataset_length}, "
            f"column_name={self.column_name})"
        )


def pad_hf_dataset(
    dataset: datasets.Dataset,
    global_batch_size: int,
    column_name: str,
) -> PaddedDataset:
    """
    Pads the dataset to match a multiple of the global batch size.

    Args:
        dataset: The dataset to pad.
        global_batch_size: The global batch size.
        column_name: The column name in the dataset.

    Returns:
        The padded dataset.
    """
    dataset_length = len(dataset)
    padded_length = int(math.ceil(dataset_length / global_batch_size)) * global_batch_size
    padded_dataset = PaddedDataset(dataset, padded_length, column_name)
    LOGGER.info(f"Padding dataset to length {padded_length}.")
    return padded_dataset


def group_texts(examples: dict[str, Sequence[Any]], block_size: int, eod_token: int = None) -> dict[str, Any]:
    """
    Groups texts together in a chunk of block_size.

    This reduces the padding in the pre-training data by saving slices of data in a single sequence. Alternative to
    this is an online packing algorithm, which may lead to small padding overheads.

    Args:
        examples: The data elements that should be grouped. The data elements should be batched, i.e., a list of lists.
        block_size: The size of the block to group the texts in.
        eod_token: The end-of-document token to add at the end of each block. If None, no
            end-of-document token is added.

    Returns:
        dict: The grouped data elements.
    """
    # Concatenate all texts. Add End-of-Document token if provided.
    if eod_token is not None:
        # Interleave the EOD token between the examples.
        def exmp_to_chain_fn(x):
            return chain(*chain.from_iterable(zip(x, [[eod_token]] * len(x))))
    else:

        def exmp_to_chain_fn(x):
            return chain(*x)

    concatenated_examples = {k: list(exmp_to_chain_fn(examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)] for k, t in concatenated_examples.items()
    }
    return result


def tokenize_dataset(
    dataset: datasets.Dataset,
    tokenizer_path: str,
    add_bos: bool,
    add_eos: bool,
    column_name: str,
    hf_access_token: str | None = None,
    num_proc: int | None = None,
    cache_dir: str | None = None,
) -> tuple[datasets.Dataset, transformers.AutoTokenizer]:
    """
    Tokenizes the dataset.

    Args:
        dataset: The dataset to tokenize.
        tokenizer: The tokenizer to use.
        add_bos: Whether to add the beginning of sequence token.
        add_eos: Whether to add the end of sequence token.
        column_name: The column name in the dataset.
        hf_access_token: The access token for HuggingFace.
        num_proc: The number of processes to use in the preprocessing maps.
        cache_dir: The cache directory for the tokenizer.

    Returns:
        The tokenized dataset and the tokenizer.
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_path,
        clean_up_tokenization_spaces=False,  # See https://github.com/huggingface/transformers/issues/31884
        legacy=False,
        token=hf_access_token,
        use_fast=True,
        add_bos=add_bos,
        add_eos=add_eos,
        cache_dir=cache_dir,
    )

    def _tokenize(example):
        return tokenizer(
            example, return_attention_mask=False, return_token_type_ids=False, truncation=False, max_length=None
        )

    dataset = dataset.map(
        _tokenize,
        input_columns=column_name,
        remove_columns=[column_name],
        batched=True,
        num_proc=num_proc,
        desc="Tokenizing",
    )
    return dataset, tokenizer


def preprocess_hf_dataset(
    dataset: datasets.Dataset,
    tokenizer_path: str,
    column_name: str,
    max_target_length: int,
    add_bos: bool = False,
    add_eos: bool = False,
    add_eod: bool = True,
    hf_access_token: str | None = None,
    num_proc: int | None = None,
    tokenizer_cache_dir: str | None = None,
) -> datasets.Dataset:
    """
    Preprocesses the HuggingFace dataset.

    Performs both tokenization and grouping of the texts.

    Args:
        dataset: The dataset to preprocess.
        tokenizer_path: The path to the tokenizer.
        column_name: The column name in the dataset.
        max_target_length: The maximum target length.
        add_bos: Whether to add the beginning of sequence token.
        add_eos: Whether to add the end of sequence token.
        add_eod: Whether to add an end of document token.
        hf_access_token: The access token for HuggingFace.
        num_proc: The number of processes to use in the preprocessing maps.
        tokenizer_cache_dir: The cache directory for the tokenizer.

    Returns:
        The preprocessed dataset.
    """
    dataset, tokenizer = tokenize_dataset(
        dataset,
        tokenizer_path,
        add_bos,
        add_eos,
        column_name,
        hf_access_token,
        num_proc,
        tokenizer_cache_dir,
    )
    dataset = dataset.select_columns(["input_ids"])

    dataset = dataset.map(
        partial(group_texts, block_size=max_target_length, eod_token=tokenizer.eos_token_id if add_eod else None),
        batched=True,
        batch_size=10 * max_target_length,
        num_proc=num_proc,
        desc="Grouping texts",
    )
    dataset = dataset.select_columns(["input_ids"]).rename_column("input_ids", column_name)
    return dataset


def preprocessing_pipeline(
    dataloading_host_index: int,
    dataloading_host_count: int,
    global_mesh: Mesh,
    dataset: Any,
    data_column_name: str,
    tokenize: bool,
    global_batch_size: int,
    max_target_length: int,
    shuffle: bool,
    data_shuffle_seed: int,
    num_epochs: int,
    tokenizer_path: str | None = None,
    hf_access_token: str | None = None,
    hf_num_map_processes: int | None = None,
    add_bos: bool = True,
    add_eos: bool = True,
    grain_packing: bool = False,
    shift: bool = True,
    worker_count: int = 1,
    worker_buffer_size: int = 1,
    drop_remainder: bool = True,
    tokenizer_cache_dir: str | None = None,
    max_steps_per_epoch: int | None = None,
) -> MultiHostDataLoadIterator:
    """
    Pipeline for preprocessing HF dataset.

    Args:
        dataloading_host_index: The index of the data loading host. Will be used to select the
            correct shard of the dataset. In JAX, this is equivalent to :func:`jax.process_index()`.
        dataloading_host_count: The number of data loading hosts. Will be used to determine the
            shard size. In JAX, this is equivalent to :func:`jax.process_count()`.
        global_mesh: The global mesh to shard the data over.
        dataset: The dataset to load. Should provide a `__getitem__` method to access elements.
        data_column_name: The column name for the data in the dataset.
        tokenize: Whether to tokenize the data.
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
        tokenizer_path: The path to the tokenizer.
        hf_access_token: The access token for HuggingFace.
        hf_num_map_processes: The number of processes to use in the preprocessing maps. If `None`,
            no multiprocessing is used.
        add_bos: Whether to add the beginning of sequence token.
        add_eos: Whether to add the end of sequence token.
        grain_packing: Whether to perform packing of the data. This is useful for datasets
            with a lot of padding, as batch elements will be packed together in a sequence
            to reduce the amount of padding. This can improve throughput efficiency. NOTE:
            if packing is enabled, the length of the iterator cannot be determined in advance
            and is likely incorrect in the iterator (will be set to maximum number of batches).
        shift: Whether to shift the input data to create the target data.
        worker_count: The number of workers to use. In grain, a single worker is usually
            sufficient, as the data loading is done in parallel across hosts.
        worker_buffer_size: The buffer size for the workers.
        drop_remainder: Whether to drop the remainder of the dataset. Note that in case of
            providing a number of epochs, the last batch of all epochs together will be
            dropped if this is set to True. If set to False, the last batch of all epochs
            together will be included in the iterator.
        tokenizer_cache_dir: The cache directory for the tokenizer.
        max_steps_per_epoch: The maximum number of steps per epoch. If provided, the iterator
            will stop after this many steps with a :class:`StopIteration` exception. Otherwise,
            will continue over the iterator until all batches are consumed.

    Returns:
        MultiHostDataLoadIterator: The multi-host data loading iterator.
    """

    assert global_batch_size % global_mesh.size == 0, "Batch size should be divisible number of global devices."
    assert not grain_packing, "We are currently not using grain packing."

    if tokenize:
        dataset = preprocess_hf_dataset(
            dataset,
            tokenizer_path,
            data_column_name,
            max_target_length,
            add_bos=add_bos,
            add_eos=add_eos,
            hf_access_token=hf_access_token,
            num_proc=hf_num_map_processes,
            tokenizer_cache_dir=tokenizer_cache_dir,
        )
    else:
        dataset = dataset.select_columns([data_column_name])

    if not drop_remainder:
        if grain_packing:
            LOGGER.warning(
                "Dropping remainder can lead to process stalls with packing. "
                "Recommended to only use with infinite data loaders."
            )
        dataset = pad_hf_dataset(dataset, global_batch_size, data_column_name)

    if max_steps_per_epoch is not None:
        LOGGER.info(f"Limiting number of steps per epoch to {max_steps_per_epoch}.")
        if isinstance(dataset, PaddedDataset):
            dataset.full_dataset_length = max_steps_per_epoch * global_batch_size
        else:
            dataset = PaddedDataset(dataset, max_steps_per_epoch * global_batch_size, data_column_name)

    LOGGER.info(f"Dataset size: {len(dataset)}")

    operations = [grain_transforms.HFNormalizeFeatures(data_column_name)]
    multihost_gen = make_grain_llm_iterator(
        dataloading_host_index,
        dataloading_host_count,
        global_mesh,
        dataset,
        global_batch_size,
        max_target_length,
        shuffle,
        data_shuffle_seed,
        num_epochs,
        operations=operations,
        grain_packing=grain_packing,
        shift=shift,
        worker_count=worker_count,
        worker_buffer_size=worker_buffer_size,
        drop_remainder=True,  # remainder is padded up if not dropped
        reset_after_epoch=(num_epochs == 1),  # only reset if we go over a single epoch
    )

    # Return multi-host jax.Array prep iterator
    return multihost_gen


def make_hf_hub_iterator(
    config: HFHubDataConfig,
    global_mesh: Mesh,
    process_indices: list[int],
    dataloading_host_index: int | None = None,
    dataloading_host_count: int | None = None,
) -> tuple[MultiHostDataLoadIterator, MultiHostDataLoadIterator]:
    """
    Load, preprocess dataset and return iterators for huggingface datasets.

    Args:
        config: HFHubDataConfig object with dataset configuration.
        global_mesh: The global mesh to shard the data over.
        process_indices: List of process indices that should load the real data. This is used to determine the data
            loading host index and host count if not provided.
        dataloading_host_index: The index of the data loading host. Will be used to select the correct shard of the
            dataset. If `None`, determined from process_indices and :func:`jax.process_index()`.
        dataloading_host_count: The number of data loading hosts. Will be used to determine the shard size. If not
            provided, determined from `process_indices`.

    Returns:
        Tuple of training and evaluation iterators.
    """
    if dataloading_host_index is None:
        dataloading_host_index = process_indices.index(jax.process_index())
    if dataloading_host_count is None:
        dataloading_host_count = len(process_indices)
    LOGGER.info(f"Loading training data of path {config.hf_path}.")
    train_ds = datasets.load_dataset(
        config.hf_path,
        data_dir=config.hf_data_dir,
        data_files=config.hf_train_files,
        cache_dir=config.hf_cache_dir,
        split="train",
        streaming=False,
        token=config.hf_access_token,
        num_proc=config.hf_num_data_processes,
    )
    train_iter = preprocessing_pipeline(
        dataloading_host_index=dataloading_host_index,
        dataloading_host_count=dataloading_host_count,
        global_mesh=global_mesh,
        dataset=train_ds,
        data_column_name=config.train_data_column,
        tokenize=config.tokenize_train_data,
        tokenizer_path=config.tokenizer_path,
        hf_access_token=config.hf_access_token,
        hf_num_map_processes=config.hf_num_map_processes,
        global_batch_size=config.global_batch_size,
        max_target_length=config.max_target_length,
        num_epochs=config.num_train_epochs,
        shuffle=config.shuffle_train_data,
        data_shuffle_seed=config.data_shuffle_seed,
        add_bos=config.add_bos,
        add_eos=config.add_eos,
        drop_remainder=True,
        tokenizer_cache_dir=config.hf_cache_dir,
    )

    LOGGER.info(f"Loading evaluation data of path {config.hf_path}.")
    eval_ds = datasets.load_dataset(
        config.hf_path,
        data_dir=config.hf_data_dir,
        data_files=config.hf_eval_files,
        cache_dir=config.hf_cache_dir,
        split=config.hf_eval_split,
        streaming=False,
        token=config.hf_access_token,
        num_proc=config.hf_num_data_processes,
    )
    if config.global_batch_size_for_eval > 0:
        eval_batch_size = config.global_batch_size_for_eval
    else:
        eval_batch_size = config.global_batch_size

    eval_iter = preprocessing_pipeline(
        dataloading_host_index=dataloading_host_index,
        dataloading_host_count=dataloading_host_count,
        global_mesh=global_mesh,
        dataset=eval_ds,
        data_column_name=config.eval_data_column,
        tokenize=config.tokenize_eval_data,
        tokenizer_path=config.tokenizer_path,
        hf_access_token=config.hf_access_token,
        hf_num_map_processes=config.hf_num_map_processes,
        global_batch_size=eval_batch_size,
        max_target_length=config.max_target_length,
        num_epochs=1_000_000,  # Infinite epochs for evals
        shuffle=False,
        data_shuffle_seed=config.data_shuffle_seed,
        add_bos=config.add_bos,
        add_eos=config.add_eos,
        drop_remainder=False,
        tokenizer_cache_dir=config.hf_cache_dir,
        max_steps_per_epoch=config.eval_max_steps_per_epoch,
    )
    # We also drop the remainder for evals as in multi-host settings, we need to have the same batch size for all hosts.

    return train_iter, eval_iter


def make_hf_local_iterator(
    config: HFLocalDataConfig,
    global_mesh: Mesh,
    process_indices: list[int],
    dataloading_host_index: int | None = None,
    dataloading_host_count: int | None = None,
) -> tuple[MultiHostDataLoadIterator, MultiHostDataLoadIterator]:
    """
    Load a preprocessed dataset from disk and return iterators for huggingface datasets.

    Args:
        config: HFLocalDataConfig object with dataset configuration.
        global_mesh: The global mesh to shard the data over.
        process_indices: List of process indices that should load the real data. This is used to
            determine the dataloading host index and host count if not provided.
        dataloading_host_index: The index of the dataloading host. Will be used to select the
            correct shard of the dataset. If None, determined from process_indices and
            jax.process_index().
        dataloading_host_count: The number of dataloading hosts. Will be used to determine the
            shard size. If not provided, determined from process_indices.

    Returns:
        Tuple of training and evaluation iterators.
    """
    if dataloading_host_index is None:
        dataloading_host_index = process_indices.index(jax.process_index())
    if dataloading_host_count is None:
        dataloading_host_count = len(process_indices)

    # Load training data from disk.
    train_path = config.data_path / config.train_split
    LOGGER.info(f"Loading training data from local path {train_path}.")
    assert train_path.exists(), f"Training data path {train_path} does not exist."
    train_ds = datasets.load_from_disk(train_path.absolute().as_posix())

    # Create training iterator.
    train_iter = preprocessing_pipeline(
        dataloading_host_index=dataloading_host_index,
        dataloading_host_count=dataloading_host_count,
        global_mesh=global_mesh,
        dataset=train_ds,
        data_column_name=config.train_data_column,
        tokenize=False,
        global_batch_size=config.global_batch_size,
        max_target_length=config.max_target_length,
        num_epochs=config.num_train_epochs,
        shuffle=config.shuffle_train_data,
        data_shuffle_seed=config.data_shuffle_seed,
        drop_remainder=True,
    )

    # Load evaluation data from disk.
    eval_path = config.data_path / config.eval_split
    LOGGER.info(f"Loading evaluation data from local path {eval_path}.")
    assert eval_path.exists(), f"Evaluation data path {eval_path} does not exist."
    eval_ds = datasets.load_from_disk(eval_path.absolute().as_posix())

    # Create evaluation iterator.
    if config.global_batch_size_for_eval > 0:
        eval_batch_size = config.global_batch_size_for_eval
    else:
        eval_batch_size = config.global_batch_size
    eval_iter = preprocessing_pipeline(
        dataloading_host_index=dataloading_host_index,
        dataloading_host_count=dataloading_host_count,
        global_mesh=global_mesh,
        dataset=eval_ds,
        data_column_name=config.eval_data_column,
        tokenize=False,
        global_batch_size=eval_batch_size,
        max_target_length=config.max_target_length,
        num_epochs=1_000_000,  # Infinite epochs for evals
        shuffle=False,
        data_shuffle_seed=config.data_shuffle_seed,
        drop_remainder=False,
        max_steps_per_epoch=config.eval_max_steps_per_epoch,
    )
    # We also drop the remainder for evals as in multi-host settings, we need to have the same batch size for all hosts.

    return train_iter, eval_iter
