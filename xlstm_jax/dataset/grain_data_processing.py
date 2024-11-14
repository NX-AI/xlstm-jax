import glob
import logging
import math
import re
from functools import partial
from pathlib import Path
from typing import Any, SupportsIndex

import grain.python as grain
import jax
from jax.sharding import Mesh

from xlstm_jax.dataset import grain_transforms, load_tokenizer

from .configs import GrainArrayRecordsDataConfig
from .grain_iterator import make_grain_llm_iterator
from .multihost_dataloading import MultiHostDataLoadIterator

LOGGER = logging.getLogger(__name__)


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
    tokenizer_path: str | None = None,
    hf_access_token: str | None = None,
    add_bos: bool = True,
    add_eos: bool = True,
    add_eod: bool = True,
    grain_packing: bool = False,
    grain_packing_bin_count: int | None = None,
    shift: bool = True,
    worker_count: int = 1,
    worker_buffer_size: int = 1,
    drop_remainder: bool = True,
    tokenizer_cache_dir: str | None = None,
    max_steps_per_epoch: int | None = None,
    eod_token_id: int | None = None,
    batch_rampup_factors: dict[int, float] | None = None,
) -> MultiHostDataLoadIterator:
    """
    Pipeline for preprocessing array_records dataset.

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
        shuffle: Whether to shuffle the dataset.
        data_shuffle_seed: The shuffle seed.
        tokenizer_path: The path to the tokenizer.
        hf_access_token: The access token for HuggingFace.
        add_bos: Whether to add the beginning of sequence token.
        add_eos: Whether to add the end of sequence token.
        add_eod: Whether to add an end of document token.
        grain_packing: Whether to perform packing of the data. This is useful for datasets
            with a lot of padding, as batch elements will be packed together in a sequence
            to reduce the amount of padding. This can improve throughput efficiency. NOTE:
            if packing is enabled, the length of the iterator cannot be determined in advance
            and is likely incorrect in the iterator (will be set to maximum number of batches).
        grain_packing_bin_count: The number of packing bins to use. If not provided, the
            bin count will be set to the batch size. It can be beneficial to increase the packing
            bins to reduce padding.
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
        eod_token_id: The token ID to use for the end-of-document token. If tokenizer_path is
            provided, the tokenizer's EOD token ID is used. If neither the tokenizer nor the
            EOD token ID is provided, an error is raised.
        batch_rampup_factors: The batch rampup factors. If provided, the batch size will be
            ramped up according to the factors. The dictionary maps the step count to the
            scaling factor. See the `boundaries_and_scales` doc in
            :func:`grain_batch_rampup.create_batch_rampup_schedule` for more details.

    Returns:
        MultiHostDataLoadIterator: The multi-host data loading iterator.
    """
    assert global_batch_size % global_mesh.size == 0, "Batch size should be divisible number of global devices."

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
        shift_target = False
        operations = [
            grain_transforms.ParseArrayRecords(column_name=data_column_name),
            grain_transforms.HFTokenize(
                create_tokenizer_fn=create_tokenizer_fn,
                column_name=data_column_name,
                max_length=max_target_length,
                add_eod=add_eod,
                eod_token_id=eod_token_id,
            ),
            grain_transforms.HFNormalizeFeatures("input_ids"),
        ]
        LOGGER.info(f"Dataset size: {len(dataset)}")
    else:
        shift_target = False
        operations = [
            grain_transforms.ParseTokenizedArrayRecords(column_name="input_ids"),
            grain_transforms.AddEODToken(eod_token_id=eod_token_id, add_eod=add_eod, max_length=max_target_length),
            grain_transforms.HFNormalizeFeatures("input_ids"),
        ]

    if not drop_remainder:
        if grain_packing:
            LOGGER.warning(
                "drop_remainder is set to False, but grain_packing is enabled. This is currently incompatible. "
                "With online-packing, we cannot determine the number of batches beforehand."
            )
        else:
            dataset = pad_arrayrecord_dataset(dataset, global_batch_size, data_column_name)

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

    multihost_gen = make_grain_llm_iterator(
        dataloading_host_index,
        dataloading_host_count,
        global_mesh,
        dataset,
        global_batch_size,
        max_target_length,
        shuffle,
        data_shuffle_seed,
        num_epochs=None,  # Infinite epochs
        operations=operations,
        grain_packing=grain_packing,
        grain_packing_bin_count=grain_packing_bin_count,
        shift=shift,
        shift_target=shift_target,
        eod_token_id=eod_token_id,
        worker_count=worker_count,
        worker_buffer_size=worker_buffer_size,
        drop_remainder=True,  # remainder is padded up if not dropped
        batch_rampup_factors=batch_rampup_factors,
    )

    # Return multi-host jax.Array prep iterator
    return multihost_gen


def make_grain_iterator(
    config: GrainArrayRecordsDataConfig,
    global_mesh: Mesh,
    process_indices: list[int],
    dataloading_host_index: int | None = None,
    dataloading_host_count: int | None = None,
) -> MultiHostDataLoadIterator:
    """Load, preprocess dataset and return iterator for pure grain ArrayRecords dataset.

    Args:
        config: GrainArrayRecordsDataConfig object with dataset configuration.
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
    split_path = config.data_path / config.split
    LOGGER.info(f"Loading {config.split} data from local path {split_path}.")
    assert split_path.exists(), f"{config.split} data path {split_path} does not exist."
    dataset = load_array_record_dataset(dataset_path=split_path)

    iterator = preprocessing_pipeline(
        dataloading_host_index=dataloading_host_index,
        dataloading_host_count=dataloading_host_count,
        global_mesh=global_mesh,
        dataset=dataset,
        data_column_name=config.data_column,
        tokenize=config.tokenize_data,
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
        worker_count=config.worker_count,
        worker_buffer_size=config.worker_buffer_size,
        drop_remainder=config.drop_remainder,
        tokenizer_cache_dir=config.hf_cache_dir,
        max_steps_per_epoch=config.max_steps_per_epoch,
        batch_rampup_factors=config.batch_rampup_factors,
    )
    return iterator


def load_array_record_dataset(dataset_path: Path | str, file_extension=".arecord") -> grain.ArrayRecordDataSource:
    """Take all files located at dataset_path and load it as grain.ArrayRecordDataSource.

    Assumes that the filenames are multiple shards where the shard idx is in the filename, e.g. train_000001.arecord'.
    We load the files in the order of the shard idx.

    Args:
        dataset_path: Path to the dataset folder, which contains .arecord files.
        file_extension: The file extension of the dataset files. Default is '.arecord'.

    Returns:
        grain.ArrayRecordDataSource: The dataset as grain.ArrayRecordDataSource.
    """
    if isinstance(dataset_path, Path):
        dataset_path = dataset_path.absolute().as_posix()
    data_file_pattern = f"{dataset_path}/*{file_extension}"
    data_files = glob.glob(data_file_pattern)
    # sort the files by the shard idx in the filename, glob instead gives 00010 before 00002.
    escaped_extension = re.escape(file_extension)
    sorted_files = sorted(data_files, key=lambda x: int(re.search(r"_(\d+)" + escaped_extension + r"$", x).group(1)))
    dataset = grain.ArrayRecordDataSource(sorted_files)
    return dataset


class PaddedDataset(grain.RandomAccessDataSource):
    """Dataset wrapper to pad the dataset to be a multiple of the global batch
    size."""

    def __init__(self, dataset: grain.ArrayRecordDataSource, full_dataset_length: int, column_name: str):
        """Initializes the PaddedDataset.

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
        """Returns padding if the record key is out of bounds, otherwise
        returns the dataset record."""
        if record_key >= len(self.dataset):
            return b""  # empty bytestring for padding (padding here refers to dataset padding, i.e. an empty sequence)
        else:
            return self.dataset[record_key]

    def __repr__(self):
        return (
            f"PaddedDataset({self.dataset}, full_dataset_length={self.full_dataset_length}, "
            f"column_name={self.column_name})"
        )


def pad_arrayrecord_dataset(
    dataset: grain.ArrayRecordDataSource,
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
