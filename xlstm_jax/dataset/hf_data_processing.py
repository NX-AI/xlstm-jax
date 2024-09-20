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

from collections.abc import Sequence
from functools import partial
from itertools import chain
from typing import Any

import datasets
import jax
import transformers
from jax.sharding import Mesh

from xlstm_jax.dataset import grain_transforms

from .configs import HFDataConfig
from .grain_iterator import make_grain_llm_iterator
from .multihost_dataloading import MultiHostDataLoadIterator


def group_texts(examples: dict[str, Sequence[Any]], block_size: int) -> dict[str, Any]:
    """
    Groups texts together in a chunk of block_size.

    This reduces the padding in the pre-training data by saving slices of data in a single
    sequence. Alternative to this is an online packing algorithm, which may lead to small
    padding overheads.

    Args:
        examples: The data elements that should be grouped. The data elements should be
            batched, i.e., a list of lists.
        block_size: The size of the block to group the texts in.

    Returns:
        dict: The grouped data elements.
    """
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    # Add a seq_ids column to keep track of the original sequence.
    concatenated_examples["seq_ids"] = list(
        chain(*[[i] * (len(examples["input_ids"][i])) for i in range(len(examples["input_ids"]))])
    )
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
) -> MultiHostDataLoadIterator:
    """
    Pipeline for preprocessing HF dataset.

    Args:
        dataloading_host_index: The index of the dataloading host. Will be used to select the
            correct shard of the dataset. In JAX, this is equivalent to `jax.process_index()`.
        dataloading_host_count: The number of dataloading hosts. Will be used to determine the
            shard size. In JAX, this is equivalent to `jax.process_count()`.
        global_mesh: The global mesh to shard the data over.
        dataset: The dataset to load. Should provide a __getitem__ method to access elements.
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
        hf_num_map_processes: The number of processes to use in the preprocessing maps. If None,
            no multi-processing is used.
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

    Returns:
        MultiHostDataLoadIterator: The multi-host data loading iterator.
    """

    assert global_batch_size % global_mesh.size == 0, "Batch size should be divisible number of global devices."
    assert not grain_packing, "We are currently not using grain packing."

    if tokenize:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_path,
            add_bos_token=add_bos,
            add_eos_token=add_eos,
            clean_up_tokenization_spaces=False,  # See https://github.com/huggingface/transformers/issues/31884
            legacy=False,
            token=hf_access_token,
        )

        dataset = dataset.map(
            grain_transforms.tokenization,
            batched=True,
            fn_kwargs={"hf_tokenizer": tokenizer, "max_length": None, "column_name": data_column_name},
            num_proc=hf_num_map_processes,
            desc="Tokenizing",
        )
        dataset = dataset.select_columns(["input_ids"])

        # TextGrouper and subsequent map is taken from xlstm-dev:
        # concatenate sequences
        # text_grouper = TextGrouper(context_length=max_target_length, output_column=None)
        # dataset = dataset.map(
        #     text_grouper, batched=True, batch_size=10 * max_target_length
        # )  # increase batch_size to reduce removed tokens
        dataset = dataset.map(
            partial(group_texts, block_size=max_target_length),
            batched=True,
            batch_size=10 * max_target_length,
            desc="Grouping texts",
        )

        dataset = dataset.select_columns(["input_ids"]).rename_column("input_ids", data_column_name)
    else:
        dataset = dataset.select_columns([data_column_name])

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
        drop_remainder=drop_remainder,
    )

    # Return multi-host jax.Array prep iterator
    return multihost_gen


def make_hf_iterator(
    config: HFDataConfig,
    global_mesh: Mesh,
    process_indices: list[int],
    dataloading_host_index: int | None = None,
    dataloading_host_count: int | None = None,
) -> tuple[MultiHostDataLoadIterator, MultiHostDataLoadIterator]:
    """
    Load, preprocess dataset and return iterators for huggingface datasets.

    Args:
        config: HFDataConfig object with dataset configuration.
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
    train_ds = datasets.load_dataset(
        config.hf_path,
        data_dir=config.hf_data_dir,
        data_files=config.hf_train_files,
        cache_dir=config.hf_cache_dir,
        split="train",
        streaming=False,
        token=config.hf_access_token,
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
        shuffle=True,
        data_shuffle_seed=config.data_shuffle_seed,
        add_bos=config.add_bos,
        add_eos=config.add_eos,
        drop_remainder=True,
    )

    eval_ds = datasets.load_dataset(
        config.hf_path,
        data_dir=config.hf_data_dir,
        data_files=config.hf_eval_files,
        cache_dir=config.hf_cache_dir,
        split=config.hf_eval_split,
        streaming=False,
        token=config.hf_access_token,
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
        num_epochs=1,
        shuffle=False,
        data_shuffle_seed=config.data_shuffle_seed,
        add_bos=config.add_bos,
        add_eos=config.add_eos,
        drop_remainder=True,
    )
    # We also drop the remainder for evals as in multi-host settings, we need to have the same batch size for all hosts.

    return train_iter, eval_iter
