import glob
import logging
import re
from pathlib import Path
from typing import Any

import grain.python as grain
import jax
from jax.sharding import Mesh

from xlstm_jax.dataset import grain_transforms
from xlstm_jax.dataset.hf_data_processing import load_tokenizer

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
    shift: bool = True,
    worker_count: int = 1,
    worker_buffer_size: int = 1,
    drop_remainder: bool = True,
    tokenizer_cache_dir: str | None = None,
    max_steps_per_epoch: int | None = None,
    eod_token_id: int | None = None,
) -> MultiHostDataLoadIterator:
    """Pipeline for preprocessing array_records dataset.

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

    Returns:
        MultiHostDataLoadIterator: The multi-host data loading iterator.
    """
    #  TODO: address the following assertions in a future PR after merging lazy API (issue #178)
    assert tokenize is True, "Only tokenized data is supported for now."
    assert drop_remainder is True, "Only drop remainder is supported for now."
    assert max_steps_per_epoch is None, "max_steps_per_epoch is not supported for now."

    assert global_batch_size % global_mesh.size == 0, "Batch size should be divisible number of global devices."

    # Load tokenizer if provided and set eod_token_id.
    if tokenizer_path is not None:
        tokenizer = load_tokenizer(tokenizer_path, add_bos, add_eos, hf_access_token, tokenizer_cache_dir)
        if eod_token_id is not None and eod_token_id != tokenizer.eos_token_id:
            LOGGER.warning(
                "EOD token ID has been set explicitly and differs from the tokenizer's EOD token ID: "
                f"tokenizer={tokenizer.eos_token_id}, provided={eod_token_id}. Using provided EOD token ID."
            )
        else:
            eod_token_id = tokenizer.eos_token_id
            LOGGER.info(f"Using EOD token ID: {eod_token_id}.")
    else:
        tokenizer = None

    if tokenize:
        assert tokenizer_path is not None, "Tokenizer path must be provided if tokenize is True."

    # Create initial operations before those from make_grain_llm_iterator.
    shift_target = False
    operations = [
        grain_transforms.ParseArrayRecords(column_name=data_column_name),
        grain_transforms.HFTokenize(
            tokenizer=tokenizer,
            column_name=data_column_name,
            max_length=max_target_length,
            add_eod=add_eod,
            eod_token_id=eod_token_id,
        ),
        grain_transforms.HFNormalizeFeatures("input_ids"),
    ]
    LOGGER.info(f"Dataset size: {len(dataset)}")

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
        shift=shift,
        shift_target=shift_target,
        eod_token_id=eod_token_id,
        worker_count=worker_count,
        worker_buffer_size=worker_buffer_size,
        drop_remainder=True,  # remainder is padded up if not dropped
    )

    # Return multi-host jax.Array prep iterator
    return multihost_gen


def make_grain_iterator(
    config: GrainArrayRecordsDataConfig,
    global_mesh: Mesh,
    process_indices: list[int],
    dataloading_host_index: int | None = None,
    dataloading_host_count: int | None = None,
):
    """Load, preprocess dataset and return iterators for pure grain ArrayRecords datasets.

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
    train_path = config.data_path / config.train_split
    LOGGER.info(f"Loading training data from local path {train_path}.")
    assert train_path.exists(), f"Training data path {train_path} does not exist."
    train_ds = load_array_record_dataset(dataset_path=train_path)

    train_iter = preprocessing_pipeline(
        dataloading_host_index=dataloading_host_index,
        dataloading_host_count=dataloading_host_count,
        global_mesh=global_mesh,
        dataset=train_ds,
        data_column_name=config.train_data_column,
        tokenize=config.tokenize_train_data,
        tokenizer_path=config.tokenizer_path,
        global_batch_size=config.global_batch_size,
        max_target_length=config.max_target_length,
        shuffle=config.shuffle_train_data,
        data_shuffle_seed=config.data_shuffle_seed,
        add_bos=config.add_bos,
        add_eos=config.add_eos,
        add_eod=config.add_eod,
        grain_packing=config.grain_packing,
        drop_remainder=True,
        tokenizer_cache_dir=config.hf_cache_dir,
    )

    # Load evaluation data from disk.
    eval_path = config.data_path / config.eval_split
    LOGGER.info(f"Loading evaluation data from local path {eval_path}.")
    assert eval_path.exists(), f"Evaluation data path {eval_path} does not exist."
    eval_ds = load_array_record_dataset(dataset_path=eval_path)

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
        data_column_name=config.train_data_column,
        tokenize=config.tokenize_train_data,
        tokenizer_path=config.tokenizer_path,
        global_batch_size=eval_batch_size,
        max_target_length=config.max_target_length,
        shuffle=False,
        data_shuffle_seed=config.data_shuffle_seed,
        add_bos=config.add_bos,
        add_eos=config.add_eos,
        add_eod=config.add_eod,
        grain_packing=config.grain_packing,
        drop_remainder=True,  # TODO: we currently do not support False. But we want it False for eval.
        tokenizer_cache_dir=config.hf_cache_dir,
        max_steps_per_epoch=config.eval_max_steps_per_epoch,
    )

    return train_iter, eval_iter


def load_array_record_dataset(dataset_path: Path | str, file_extension=".arecord"):
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
