#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import glob
import logging
import math
import os
import re
from functools import partial
from pathlib import Path
from typing import Any, SupportsIndex

import datasets
import grain.python as grain
import jax
from jax.sharding import Mesh

from .configs import GrainArrayRecordsDataConfig, HFHubDataConfig
from .grain_iterator import make_grain_llm_dataset, make_grain_multihost_iterator
from .grain_transforms import (
    AddEODToken,
    HFNormalizeFeatures,
    HFTokenize,
    ParseArrayRecords,
    ParseTokenizedArrayRecords,
)
from .hf_tokenizer import load_tokenizer
from .multihost_dataloading import MultiHostDataLoadIterator

LOGGER = logging.getLogger(__name__)


def preprocess_dataset(
    dataloading_host_index: int,
    dataloading_host_count: int,
    dataset: Any,
    data_column_name: str,
    tokenize: bool,
    global_batch_size: int,
    max_target_length: int,
    shuffle: bool,
    data_shuffle_seed: int,
    tokenizer_path: str | None = None,
    hf_access_token: str | None = None,
    add_bos: bool = True,
    add_eos: bool = True,
    add_eod: bool = True,
    grain_packing: bool = False,
    grain_packing_bin_count: int | None = None,
    shift: bool = True,
    drop_remainder: bool = True,
    num_epochs: int | None = None,
    tokenizer_cache_dir: str | None = None,
    max_steps_per_epoch: int | None = None,
    eod_token_id: int | None = None,
) -> tuple[grain.IterDataset, grain.RandomAccessDataSource]:
    """
    Pipeline for preprocessing an array_records or huggingface dataset.

    Args:
        dataloading_host_index: The index of the data loading host. Will be used to select the
            correct shard of the dataset. In JAX, this is equivalent to :func:`jax.process_index()`.
        dataloading_host_count: The number of data loading hosts. Will be used to determine the
            shard size. In JAX, this is equivalent to :func:`jax.process_count()`.
        dataset: The dataset to load. Should provide a `__getitem__` method to access elements.
        data_column_name: The column name for the data in the dataset.
        tokenize: Whether to tokenize the data.
        global_batch_size: The global batch size.
        max_target_length: The maximum target length.
        shuffle: Whether to shuffle the dataset.
        data_shuffle_seed: The shuffle seed.
        tokenizer_path: The path to the tokenizer.
        hf_access_token: The access token for HuggingFace.
        add_bos: Whether to add the beginning of sequence token.
        add_eos: Whether to add the end of sequence token.
        add_eod: Whether to add an end of document token.
        grain_packing: Whether to perform packing of the data. This is useful for datasets
            with a lot of padding, as batch elements will be packed together in a sequence
            to reduce the amount of padding. This can improve throughput efficiency. Note:
            if packing is enabled, the length of the iterator cannot be determined in advance
            and is likely incorrect in the iterator (will be set to maximum number of batches).
        grain_packing_bin_count: The number of packing bins to use. If not provided, the
            bin count will be set to the batch size. It can be beneficial to increase the packing
            bins to reduce padding.
        shift: Whether to shift the input data to create the target data.
        drop_remainder: Whether to drop the remainder of the dataset. Note that in case of
            providing a number of epochs, the last batch of all epochs together will be
            dropped if this is set to `True`. If set to `False`, the last batch of all epochs
            together will be included in the iterator.
        num_epochs: The number of epochs to train for. The dataset will be repeated for so
            many epochs, and the shuffle order will be different for each epoch. If None,
            the dataset will be repeated infinitely. Note that batches of an epoch can
            spill over into the first batch of the next epoch, to avoid dropping data.
            The argument `drop_remainder` controls whether the very last batch of all epochs
            together is dropped. By default, use None (infinite epochs) for training and validation.
        tokenizer_cache_dir: The cache directory for the tokenizer.
        max_steps_per_epoch: The maximum number of steps per epoch. If provided, the iterator
            will stop after this many steps with a :class:`StopIteration` exception. Otherwise,
            will continue over the iterator until all batches are consumed.
        eod_token_id: The token ID to use for the end-of-document token. If `tokenizer_path` is
            provided, the tokenizer's EOD token ID is used.

    Returns:
        The preprocessed grain dataset and the original data source.
    """
    assert (
        global_batch_size % dataloading_host_count == 0
    ), "Batch size should be divisible by the number of dataloading hosts."

    # Load tokenizer if provided and set eod_token_id.
    create_tokenizer_fn = partial(
        load_tokenizer, tokenizer_path, add_bos, add_eos, hf_access_token, tokenizer_cache_dir
    )
    if tokenizer_path is not None:
        tokenizer = create_tokenizer_fn()
        LOGGER.info(f"Loaded tokenizer from {tokenizer_path} with vocab size {tokenizer.vocab_size}.")
        if eod_token_id is not None and eod_token_id != tokenizer.eos_token_id:
            LOGGER.warning(
                "EOD token ID has been set explicitly and differs from the tokenizer's EOD token ID: "
                f"tokenizer={tokenizer.eos_token_id}, provided={eod_token_id}. Using provided EOD token ID."
            )
        else:
            eod_token_id = tokenizer.eos_token_id
            LOGGER.info(f"Using EOD token ID: {eod_token_id}.")

    if tokenize:
        assert tokenizer_path is not None, "Tokenizer path must be provided if tokenize is True."

        # Create initial operations before those from make_grain_llm_iterator.
        operations = []
        if isinstance(dataset, grain.ArrayRecordDataSource):
            operations.append(ParseArrayRecords(column_name=data_column_name))
        else:
            dataset = dataset.select_columns([data_column_name])
        operations.extend(
            [
                HFTokenize(
                    create_tokenizer_fn=create_tokenizer_fn,
                    column_name=data_column_name,
                    max_length=max_target_length,
                    add_eod=add_eod,
                    eod_token_id=eod_token_id,
                ),
                HFNormalizeFeatures("input_ids"),
            ]
        )
    else:
        assert isinstance(dataset, grain.ArrayRecordDataSource), "Pre-processed datasets must be ArrayRecordDataSource."
        operations = [
            ParseTokenizedArrayRecords(column_name="input_ids"),
            AddEODToken(eod_token_id=eod_token_id, add_eod=add_eod, max_length=max_target_length),
            HFNormalizeFeatures("input_ids"),
        ]

    if not drop_remainder:
        if grain_packing:
            LOGGER.warning(
                "drop_remainder is set to False, but grain_packing is enabled. This is currently incompatible. "
                "With online-packing, we cannot determine the number of batches beforehand."
            )
        else:
            dataset = pad_dataset(dataset, global_batch_size, data_column_name)

    if max_steps_per_epoch is not None:
        LOGGER.info(f"Limiting number of steps per epoch to {max_steps_per_epoch}.")
        if grain_packing:
            LOGGER.warning(
                "Trying to use max_steps_per_epoch, but grain_packing is enabled. Will limit number of examples to "
                "max_steps_per_epoch * global_batch_size, but this may lead to fewer batches than max_steps_per_epoch."
            )
        if len(dataset) <= max_steps_per_epoch * global_batch_size:
            LOGGER.info(
                f"Dataset size is {len(dataset)}, which is smaller than max_steps_per_epoch * global_batch_size "
                f"{max_steps_per_epoch * global_batch_size}. Skipping limiting the number of steps per epoch."
            )
        elif isinstance(dataset, PaddedDataset):
            dataset.full_dataset_length = max_steps_per_epoch * global_batch_size
        else:
            dataset = PaddedDataset(dataset, max_steps_per_epoch * global_batch_size, data_column_name)

    LOGGER.info(f"Dataset size: {len(dataset)}")

    grain_dataset = make_grain_llm_dataset(
        dataloading_host_index,
        dataloading_host_count,
        dataset,
        global_batch_size,
        max_target_length,
        shuffle,
        data_shuffle_seed,
        num_epochs=num_epochs,
        operations=operations,
        grain_packing=grain_packing,
        grain_packing_bin_count=grain_packing_bin_count,
        shift=shift,
        shift_target=False,  # we shift inputs
        eod_token_id=eod_token_id,
    )

    # Return grain dataset and data source.
    return grain_dataset, dataset


def make_grain_iterator(
    configs: GrainArrayRecordsDataConfig | HFHubDataConfig | list[GrainArrayRecordsDataConfig | HFHubDataConfig],
    global_mesh: Mesh,
    process_indices: list[int],
    dataset_weights: list[float] | None = None,
) -> MultiHostDataLoadIterator:
    """Load a dataset, create the preprocessing pipeline and return a multihost data-loading iterator.

    Args:
        configs: dataset configuration object for huggingface or arrayrecords dataset. If multiple configs are provided,
            the datasets will be loaded in parallel and the data will be interleaved in a mixing style. NOTE: the
            global batch size, worker count, worker buffer size, drop remainder, and batch rampup will be only used
            from the first config. The other configs are assumed to have the same values. Otherwise, warnings will be
            raised.
        global_mesh: The global mesh to shard the data over.
        process_indices: List of process indices that should load the real data. This is used to determine the data
            loading host index and host count.
        dataset_weights: The weights for the datasets. If provided, the datasets will be mixed according to the
            weights. Otherwise, a uniform mixing is used. If a single dataset is provided, the weights are ignored.

    Returns:
        data-loading iterator (for training or evaluation).
    """
    if not isinstance(configs, list):
        configs = [configs]
    if dataset_weights is not None:
        assert len(configs) == len(dataset_weights), "Number of datasets and weights must match."
    # Assert that global batch size, worker count, worker buffer size, and drop remainder are the same for all datasets.
    for attr_name in [
        "global_batch_size",
        "max_target_length",
        "worker_count",
        "worker_buffer_size",
        "drop_remainder",
        "batch_rampup_factors",
    ]:
        for config in configs[1:]:
            if getattr(config, attr_name) != getattr(configs[0], attr_name):
                LOGGER.warning(
                    f"Attribute {attr_name} differs between datasets to mix: {config} vs {configs[0]}. "
                    f"Using value from first dataset: {getattr(configs[0], attr_name)} != {getattr(config, attr_name)}."
                )

    # Load the datasets and preprocess them.
    grain_datasets, data_sources = [], []
    for config in configs:
        # Load the dataset from disk
        if isinstance(config, GrainArrayRecordsDataConfig):
            split_path = config.data_path / config.split
            LOGGER.info(f"Loading {config.split} data from local path {split_path}.")
            dataset = load_array_record_dataset(dataset_path=split_path)
        elif isinstance(config, HFHubDataConfig):
            LOGGER.info(f"Loading {config.split} data of path {config.hf_path}.")
            dataset = load_huggingface_dataset(config)
        else:
            raise ValueError(f"Unsupported config type {type(config)}.")

        if isinstance(config, HFHubDataConfig):
            tokenize = True
        else:
            tokenize = config.tokenize_data

        # Create data-loading iterator.
        grain_dataset, dataset = preprocess_dataset(
            dataloading_host_index=process_indices.index(jax.process_index()),
            dataloading_host_count=len(process_indices),
            dataset=dataset,
            data_column_name=config.data_column,
            tokenize=tokenize,
            global_batch_size=config.global_batch_size,
            max_target_length=config.max_target_length,
            shuffle=config.shuffle_data,
            data_shuffle_seed=config.data_shuffle_seed,
            tokenizer_path=config.tokenizer_path,
            hf_access_token=config.hf_access_token,
            add_bos=config.add_bos,
            add_eos=config.add_eos,
            add_eod=config.add_eod,
            grain_packing=config.grain_packing,
            grain_packing_bin_count=config.grain_packing_bin_count,
            drop_remainder=config.drop_remainder,
            num_epochs=None,
            tokenizer_cache_dir=config.hf_cache_dir,
            max_steps_per_epoch=config.max_steps_per_epoch,
        )
        grain_datasets.append(grain_dataset)
        data_sources.append(dataset)

    # Create the grain multihost iterator.
    iterator = make_grain_multihost_iterator(
        grain_datasets=grain_datasets,
        dataset_lengths=[len(d) for d in data_sources],
        global_mesh=global_mesh,
        global_batch_size=configs[0].global_batch_size,
        dataloading_host_count=len(process_indices),
        dataset_weights=dataset_weights,
        worker_count=configs[0].worker_count,
        worker_buffer_size=configs[0].worker_buffer_size,
        drop_remainder=configs[0].drop_remainder,
        batch_rampup_factors=configs[0].batch_rampup_factors,
    )
    return iterator


class PaddedDataset(grain.RandomAccessDataSource):
    """Dataset wrapper to pad the dataset to be a multiple of the global batch size."""

    def __init__(
        self,
        dataset: grain.ArrayRecordDataSource | datasets.Dataset,
        full_dataset_length: int,
        column_name: str,
    ):
        """Initializes the PaddedDataset.

        Args:
            dataset: The dataset to pad.
            full_dataset_length: The full dataset length, including padding. Should be a multiple of the global
                batch size with which the dataset is loaded. Also, lengths smaller than the dataset are supported
                and will return the smaller length.
            column_name: The column name in the dataset that is returned.
        """
        assert isinstance(dataset, grain.ArrayRecordDataSource | datasets.Dataset)
        self.dataset = dataset
        self.full_dataset_length = full_dataset_length
        self.column_name = column_name

    def __len__(self):
        """Returns the full dataset length."""
        return self.full_dataset_length

    def __getitem__(self, record_key: SupportsIndex) -> Any:
        """Returns padding if the record key is out of bounds, otherwise returns the dataset record."""
        if record_key >= len(self.dataset):
            return self.empty_sequence
        return self.dataset[record_key]

    def __repr__(self):
        return (
            f"PaddedDataset({self.dataset}, full_dataset_length={self.full_dataset_length}, "
            f"column_name={self.column_name})"
        )

    @property
    def empty_sequence(self):
        """Returns and empty sequence for padding, depending on the type of dataset."""
        if isinstance(self.dataset, grain.ArrayRecordDataSource):
            # AR datasets return bytestrings. We return an empty bytestring for padding.
            return b""
        if isinstance(self.dataset, datasets.Dataset):
            # HF datasets return dicts {self.column_name: some_text}. We return a dict with an empty string for padding.
            return {self.column_name: ""}
        raise ValueError(f"Unsupported dataset type {type(self.dataset)}.")


def pad_dataset(
    dataset: grain.ArrayRecordDataSource | datasets.Dataset,
    global_batch_size: int,
    column_name: str,
) -> PaddedDataset:
    """Pads the dataset to match a multiple of the global batch size.

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


def load_array_record_dataset(
    dataset_path: Path | str, file_extension: str = ".arecord"
) -> grain.ArrayRecordDataSource:
    """Take all files located at dataset_path and load it as grain.ArrayRecordDataSource.

    Assumes that the filenames are multiple shards where the shard idx is in the filename, e.g. train_000001.arecord'.
    We load the files in the order of the shard idx.

    Args:
        dataset_path: Path to the dataset folder, which contains .arecord files.
        file_extension: The file extension of the dataset files. Default is '.arecord'.

    Returns:
        grain.ArrayRecordDataSource: The dataset as grain.ArrayRecordDataSource.
    """
    assert os.path.exists(dataset_path), f"dataset path {dataset_path} does not exist."
    if isinstance(dataset_path, Path):
        dataset_path = dataset_path.absolute().as_posix()
    data_file_pattern = f"{dataset_path}/*{file_extension}"
    data_files = glob.glob(data_file_pattern)
    # sort the files by the shard idx in the filename, glob instead gives 00010 before 00002.
    escaped_extension = re.escape(file_extension)
    sorted_files = sorted(data_files, key=lambda x: int(re.search(r"_(\d+)" + escaped_extension + r"$", x).group(1)))
    dataset = grain.ArrayRecordDataSource(sorted_files)
    return dataset


def load_huggingface_dataset(config: HFHubDataConfig) -> datasets.Dataset:
    """Load a dataset from HuggingFace.

    Args:
        config: The HFHubDataConfig object.

    Returns:
        datasets.Dataset: The loaded dataset.
    """
    dataset = datasets.load_dataset(
        config.hf_path,
        data_dir=config.hf_data_dir,
        data_files=config.hf_data_files,
        cache_dir=config.hf_cache_dir,
        split=config.split,
        streaming=False,
        token=config.hf_access_token,
        num_proc=config.hf_num_data_processes,
    )
    return dataset
