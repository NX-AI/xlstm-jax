from time import time

import datasets
import transformers
from datasets import Dataset
from jax.sharding import Mesh
from tabulate import tabulate as python_tabulate

from xlstm_jax.dataset import HFHubDataConfig, create_data_iterator
from xlstm_jax.dataset.batch import LLMBatch
from xlstm_jax.dataset.multihost_dataloading import MultiHostDataLoadIterator
from xlstm_jax.distributed.mesh_utils import initialize_mesh
from xlstm_jax.models.configs import ParallelConfig
from xlstm_jax.trainer.base.param_utils import flatten_dict


def compute_packed_batch_statistics(
    batch_size_per_device: int = 4, context_length: int = 2048, max_batches: int = 20
) -> dict[str, dict[str, float]]:
    """Computes the fraction of padding and eod tokens in a given number of
    batches.

    Note that the mesh is currently hard-coded in the _setup_data helper function and needs to be adjusted if required.
    The statistics should not depend on the number of parallel processes though.

    Args:
        batch_size_per_device: The (local) batch size per device.
        context_length: The context/sequence length.
        max_batches: The number of batches to load for computing the stats.

    Returns:
        Dict (train, validation) of dicts with the fraction of padding and eod tokens in the dataset.
    """
    mesh, train_ds, eval_ds, train_iterator, eval_iterator, tokenizer = _setup_data(
        batch_size_per_device=batch_size_per_device,
        context_length=context_length,
    )
    loaded_batches = {
        "train": _load_n_batches(train_iterator, max_batches=max_batches),
        "validation": _load_n_batches(eval_iterator, max_batches=max_batches),
    }

    padding_token_id = 0  # hard-coded to the default value 0 for now

    # Compute the fraction of padding and EOD tokens in the dataset.
    stats = {"train": {}, "validation": {}}
    for split, batches in loaded_batches.items():
        n_padded = sum([(batch.inputs == padding_token_id).sum() for batch in batches])
        n_eod = sum([(batch.inputs == tokenizer.eos_token_id).sum() for batch in batches])
        n_total = sum([batch.inputs.size for batch in batches])
        stats[split]["frac_padded"] = n_padded / n_total
        stats[split]["frac_eod"] = n_eod / n_total

    return stats


def compute_and_tabulate_packed_batch_statistics() -> str:
    """Computes and tabulates the fraction of padding and eod tokens for a few
    combinations of different batch size and context length.

    Returns:
        Tabulated string with the statistics.
    """
    # batch sizes are local batch sizes since we use just a single device.
    batch_size_and_context_length_combinations = [
        (2, 2048),
        (4, 2048),
        (8, 2048),
        (2, 8192),
        (4, 8192),
    ]
    # create a table of statistics for different batch sizes and context lengths
    summary = {}
    start = time()
    for batch_size, context_length in batch_size_and_context_length_combinations:
        stats = compute_packed_batch_statistics(batch_size_per_device=batch_size, context_length=context_length)
        stats = flatten_dict(stats)
        if not summary:
            summary = {key: [] for key in stats}
            summary["batch_size"] = []
            summary["context_length"] = []

        summary["batch_size"].append(batch_size)
        summary["context_length"].append(context_length)
        for key, value in stats.items():
            summary[key].append(value)

    stop = time()
    table_str = python_tabulate(summary, headers="keys", intfmt="_", floatfmt=".4f")
    print(f"Total time: {stop - start:.2f} seconds")
    print(table_str)
    return table_str


def _load_n_batches(iterator: MultiHostDataLoadIterator, max_batches: int = 20) -> list[LLMBatch]:
    """Loads a given number of batches from an iterator.

    Args:
        iterator: MultiHostDataLoadIterator from which to load batches.
        max_batches: Maximum number of batches to load.

    Returns:
        List of loaded batches.
    """
    batches = []
    for batch in iterator:
        if len(batches) >= max_batches:
            break
        batches.append(batch)
    return batches


def _setup_data(
    batch_size_per_device: int = 2, context_length: int = 128
) -> tuple[
    ParallelConfig,
    Mesh,
    Dataset,
    Dataset,
    MultiHostDataLoadIterator,
    MultiHostDataLoadIterator,
    transformers.AutoTokenizer,
]:
    """Helper function to get data iterators, datasets, mesh, and tokenizer for
    testing.

    Note: We have a lot of code duplication compared to the helper function in
    tests/test_hf_data_processing. But here we later want to check
    different datasets. So keep it separate for now.

    Args:
        batch_size_per_device: The (local) batch size per device.
        context_length: The context/sequence length.

    Returns:
        parallel config, mesh, train_ds, eval_ds, train_iterator, eval_iterator, tokenizer.
    """

    # Define data configuration.
    parallel = ParallelConfig(
        data_axis_name="dp",
        fsdp_axis_name="fsdp",
        model_axis_name="tp",
        pipeline_axis_name="pp",
        fsdp_modules=("Embed", "LMHead", "mLSTMBlock"),
        fsdp_min_weight_size=2**8,
        fsdp_axis_size=1,
        model_axis_size=1,
        data_axis_size=-1,
    )

    # Initialize mesh.
    mesh = initialize_mesh(parallel_config=parallel)

    data_config = HFHubDataConfig(
        global_batch_size=batch_size_per_device * mesh.shape[parallel.data_axis_name],
        max_target_length=context_length,
        hf_path="DKYoon/SlimPajama-6B",
        hf_cache_dir="/nfs-gpu/xlstm/data/hf_cache",
        hf_eval_split="validation",
        train_data_column="text",
        eval_data_column="text",
        tokenize_train_data=True,
        tokenize_eval_data=True,
        tokenizer_path="gpt2",
        data_shuffle_seed=42,
        add_bos=False,
        add_eos=False,
        add_eod=True,
        grain_packing=True,
        hf_num_data_processes=32,
    )

    # Load the train and validation datasets.
    train_ds = datasets.load_dataset(
        data_config.hf_path,
        data_dir=data_config.hf_data_dir,
        data_files=data_config.hf_train_files,
        cache_dir=data_config.hf_cache_dir,
        split="train",
        streaming=False,
        token=data_config.hf_access_token,
        num_proc=data_config.hf_num_data_processes,
    )
    eval_ds = datasets.load_dataset(
        data_config.hf_path,
        data_dir=data_config.hf_data_dir,
        data_files=data_config.hf_train_files,
        cache_dir=data_config.hf_cache_dir,
        split="validation",
        streaming=False,
        token=data_config.hf_access_token,
        num_proc=data_config.hf_num_data_processes,
    )

    # Define iterators
    train_iterator, eval_iterator = create_data_iterator(config=data_config, mesh=mesh)

    # Get the tokenizer that is used for the train_iterator and eval_iterator. We need it for testing.
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        data_config.tokenizer_path,
        clean_up_tokenization_spaces=False,  # See https://github.com/huggingface/transformers/issues/31884
        legacy=False,
        token=data_config.hf_access_token,
        use_fast=True,
        add_bos=data_config.add_bos,
        add_eos=data_config.add_eos,
        cache_dir=data_config.hf_cache_dir,
    )
    return parallel, mesh, train_ds, eval_ds, train_iterator, eval_iterator, tokenizer


if __name__ == "__main__":
    compute_and_tabulate_packed_batch_statistics()
