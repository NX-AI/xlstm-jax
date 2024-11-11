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
from .hf_tokenizer import load_tokenizer
from .multihost_dataloading import MultiHostDataLoadIterator

LOGGER = logging.getLogger(__name__)


class PaddedDataset(grain.RandomAccessDataSource):
    """Dataset wrapper to pad the dataset to be a multiple of the global batch
    size."""

    def __init__(self, dataset: datasets.Dataset, full_dataset_length: int, column_name: str):
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


def group_texts(examples: dict[str, Sequence[Any]], block_size: int, eod_token: int = None) -> dict[str, Any]:
    """Groups texts together in a chunk of block_size.

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
    tokenizer: transformers.AutoTokenizer | None = None,
) -> tuple[datasets.Dataset, transformers.AutoTokenizer]:
    """Tokenizes the dataset.

    Args:
        dataset: The dataset to tokenize.
        tokenizer: The tokenizer to use.
        add_bos: Whether to add the beginning of sequence token.
        add_eos: Whether to add the end of sequence token.
        column_name: The column name in the dataset.
        hf_access_token: The access token for HuggingFace.
        num_proc: The number of processes to use in the preprocessing maps.
        cache_dir: The cache directory for the tokenizer.
        tokenizer: The tokenizer to use. If provided, the loading of the tokenizer is skipped. If not provided, the
            tokenizer is loaded from the `tokenizer_path`.

    Returns:
        The tokenized dataset and the tokenizer.
    """
    if tokenizer is None:
        tokenizer = load_tokenizer(tokenizer_path, add_bos, add_eos, hf_access_token, cache_dir)

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
    apply_group_texts: bool = True,
    tokenizer: transformers.AutoTokenizer | None = None,
) -> datasets.Dataset:
    """Preprocesses the HuggingFace dataset.

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
        apply_group_texts: Whether to group the texts.
        tokenizer: The tokenizer to use. If provided, the loading of the tokenizer is skipped. If not provided, the
            tokenizer is loaded from the `tokenizer_path`.

    Returns:
        The preprocessed dataset.
    """
    dataset, tokenizer = tokenize_dataset(
        dataset=dataset,
        tokenizer_path=tokenizer_path,
        add_bos=add_bos,
        add_eos=add_eos,
        column_name=column_name,
        hf_access_token=hf_access_token,
        num_proc=num_proc,
        cache_dir=tokenizer_cache_dir,
        tokenizer=tokenizer,
    )
    dataset = dataset.select_columns(["input_ids"])
    if apply_group_texts:
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
    tokenizer_path: str | None = None,
    hf_access_token: str | None = None,
    hf_num_map_processes: int | None = None,
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
    """Pipeline for preprocessing HF dataset.

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
        hf_num_map_processes: The number of processes to use in the preprocessing maps. If `None`,
            no multiprocessing is used.
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
            ramped up according to the schedule. The dictionary maps the step count to the
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

    # if tokenize=True, assume that dataset may be pre-processed (tokenized + grouped) already.
    if tokenize:
        # We can tokenize either via HF map (offline) or using grain (online).
        assert tokenizer_path is not None, "Tokenizer path must be provided if tokenize is True."

        if grain_packing:
            dataset = dataset.select_columns([data_column_name])
            shift_target = False
            operations = [
                grain_transforms.HFTokenize(
                    create_tokenizer_fn=create_tokenizer_fn,
                    column_name=data_column_name,
                    max_length=max_target_length,
                    add_eod=add_eod,
                    eod_token_id=eod_token_id,
                ),
                grain_transforms.HFNormalizeFeatures("input_ids"),
            ]
        else:
            dataset = preprocess_hf_dataset(
                dataset=dataset,
                tokenizer_path=tokenizer_path,
                column_name=data_column_name,
                max_target_length=max_target_length,
                add_bos=add_bos,
                add_eos=add_eos,
                add_eod=add_eod,
                hf_access_token=hf_access_token,
                num_proc=hf_num_map_processes,
                tokenizer_cache_dir=tokenizer_cache_dir,
                apply_group_texts=True,
                tokenizer=tokenizer,
            )
            operations = [grain_transforms.HFNormalizeFeatures(data_column_name)]
            shift_target = True
    else:
        dataset = dataset.select_columns([data_column_name])
        operations = [grain_transforms.HFNormalizeFeatures(data_column_name)]
        shift_target = True
        assert (
            eod_token_id is not None
        ), "EOD token ID must be provided if tokenize is False and no tokenizer is provided."

    if not drop_remainder:
        if grain_packing:
            LOGGER.warning(
                "drop_remainder is set to False, but grain_packing is enabled. This is currently incompatible, "
                "since PaddedDataset is applied to a datasets instance. Not using drop_remainder!"
            )
            # PaddedDataset assumes that dataset instance is tokenized and grouped, i.e. the format
            # {str: List[int]} is expected, where the list of integers are tokens.
            # However, when grain_packing is enabled, the dataset instance will be unprocessed, i.e. a raw string.
            # In order to support this for packing, we would need to implement this operation in grain.
        else:
            dataset = pad_hf_dataset(dataset, global_batch_size, data_column_name)

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


def make_hf_hub_iterator(
    config: HFHubDataConfig,
    global_mesh: Mesh,
    process_indices: list[int],
    dataloading_host_index: int | None = None,
    dataloading_host_count: int | None = None,
) -> MultiHostDataLoadIterator:
    """Load, preprocess dataset and return data-loading iterator for huggingface datasets.

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
        data-loading iterator (for training or evaluation).
    """
    if dataloading_host_index is None:
        dataloading_host_index = process_indices.index(jax.process_index())
    if dataloading_host_count is None:
        dataloading_host_count = len(process_indices)

    LOGGER.info(f"Loading {config.split} data of path {config.hf_path}.")
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

    # Create data-loading iterator.
    iterator = preprocessing_pipeline(
        dataloading_host_index=dataloading_host_index,
        dataloading_host_count=dataloading_host_count,
        global_mesh=global_mesh,
        dataset=dataset,
        data_column_name=config.data_column,
        tokenize=config.tokenize_data,
        tokenizer_path=config.tokenizer_path,
        hf_access_token=config.hf_access_token,
        hf_num_map_processes=config.hf_num_map_processes,
        global_batch_size=config.global_batch_size,
        max_target_length=config.max_target_length,
        shuffle=config.shuffle_data,
        data_shuffle_seed=config.data_shuffle_seed,
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


def make_hf_local_iterator(
    config: HFLocalDataConfig,
    global_mesh: Mesh,
    process_indices: list[int],
    dataloading_host_index: int | None = None,
    dataloading_host_count: int | None = None,
) -> MultiHostDataLoadIterator:
    """Load a preprocessed dataset from disk and return data-loading iterator for huggingface datasets.

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
        data-loading iterator (for training or evaluation).
    """
    if dataloading_host_index is None:
        dataloading_host_index = process_indices.index(jax.process_index())
    if dataloading_host_count is None:
        dataloading_host_count = len(process_indices)

    # Load dataset from disk.
    split_path = config.data_path / config.split
    LOGGER.info(f"Loading {config.split} data from local path {split_path}.")
    assert split_path.exists(), f"Training data path {split_path} does not exist."
    dataset = datasets.load_from_disk(split_path.absolute().as_posix())

    # Create data-loading iterator.
    iterator = preprocessing_pipeline(
        dataloading_host_index=dataloading_host_index,
        dataloading_host_count=dataloading_host_count,
        global_mesh=global_mesh,
        dataset=dataset,
        data_column_name=config.data_column,
        tokenize=False,
        global_batch_size=config.global_batch_size,
        max_target_length=config.max_target_length,
        shuffle=config.shuffle_data,
        data_shuffle_seed=config.data_shuffle_seed,
        worker_count=config.worker_count,
        worker_buffer_size=config.worker_buffer_size,
        drop_remainder=config.drop_remainder,
        max_steps_per_epoch=config.max_steps_per_epoch,
        eod_token_id=config.eod_token_id,
        batch_rampup_factors=config.batch_rampup_factors,
    )
    return iterator
