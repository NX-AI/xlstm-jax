import os

# We disable GPU usage for this script, as it is not needed.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

# flake8: noqa E402
import argparse
import enum
import itertools
from collections import defaultdict
from pathlib import Path
from time import time

from tabulate import tabulate as python_tabulate

from xlstm_jax.dataset import GrainArrayRecordsDataConfig, HFHubDataConfig, create_data_iterator
from xlstm_jax.dataset.hf_data_processing import load_tokenizer
from xlstm_jax.dataset.multihost_dataloading import MultiHostDataLoadIterator
from xlstm_jax.distributed.mesh_utils import initialize_mesh
from xlstm_jax.models.configs import ParallelConfig
from xlstm_jax.trainer.base.param_utils import flatten_dict


class DatasetNameToHFPath(enum.StrEnum):
    SlimPajama6B = "DKYoon/SlimPajama-6B"
    SlimPajama627B = "cerebras/SlimPajama-627B"
    DCLM = "mlfoundations/dclm-baseline-1.0-parquet"


class DatasetNameToArrayRecordsPath(enum.StrEnum):
    SlimPajama6B = "/nfs-gpu/xlstm/data/array_records/DKYoon_SlimPajama-6B"
    SlimPajama627B = "/nfs-gpu/xlstm/data/array_records/cerebras_SlimPajama-627B"
    DCLM = "/nfs-gpu/xlstm/data/array_records/mlfoundations_dclm-baseline-1.0-parquet"


def compute_and_tabulate_packed_batch_statistics(
    dataset_name: str, use_array_records: bool = True, max_batches: int = 50
) -> str:
    """Computes and tabulates the fraction of padding and eod tokens for a few
    combinations of different batch size and context length.

    Args:
        dataset_name: The name of the dataset to load. Must be one of the keys in DatasetNameToHFPath.
        use_array_records: Whether to use array records or huggingface dataset for data loading.
        max_batches: Maximum number of batches to load for computing the stats.

    Returns:
        Tabulated string with the statistics for different batch sizes and context lengths.
    """
    print(f"Computing statis using {'array records' if use_array_records else 'huggingface'} dataset {dataset_name}.")

    parallel = ParallelConfig()
    mesh = initialize_mesh(parallel_config=parallel)

    train_data_config, _ = _create_data_config(
        dataset_name=dataset_name,
        use_array_records=use_array_records,
        context_length=-1,
        global_batch_size=-1,
    )

    tokenizer = load_tokenizer(
        tokenizer_path=train_data_config.tokenizer_path,
        add_bos=train_data_config.add_bos,
        add_eos=train_data_config.add_eos,
        hf_access_token=train_data_config.hf_access_token,
        cache_dir=train_data_config.hf_cache_dir,
    )

    # batch sizes are per-device, since we use just a single device. Will use mesh to get global_batch_size.
    per_device_batch_size_and_context_lengths = [
        (8, 2048),
        (2, 8192),
        (4, 8192),
        (8, 8192),
        (16, 8192),
        (32, 8192),
        (128, 8192),
    ]
    # create a table of statistics for different batch sizes and context lengths
    summary = defaultdict(list)
    start = time()
    for batch_size, context_length in per_device_batch_size_and_context_lengths:
        print(f"Computing statistics for batch size {batch_size} and context length {context_length}...")
        _train_data_config, _ = _create_data_config(
            dataset_name=dataset_name,
            use_array_records=use_array_records,
            context_length=context_length,
            global_batch_size=batch_size * mesh.shape[parallel.data_axis_name],
        )
        iterator = create_data_iterator(config=_train_data_config, mesh=mesh)

        stats = compute_packed_batch_statistics(
            iterator=iterator,
            eod_token_id=tokenizer.eos_token_id,  # eos is used for eod.
            max_batches=max_batches,
        )
        stats = flatten_dict(stats)

        summary["batch_size"].append(batch_size)
        summary["context_length"].append(context_length)
        for key, value in stats.items():
            summary[key].append(value)

    stop = time()
    print(f"Iterator loading and stats computation time: {stop - start:.2f} seconds")
    table_str = python_tabulate(summary, headers="keys", intfmt="_", floatfmt=".4f")
    print(table_str)
    return table_str


def compute_packed_batch_statistics(
    iterator: MultiHostDataLoadIterator,
    eod_token_id: int,
    padding_token_id: int = 0,
    max_batches: int = 50,
) -> dict[str, float]:
    """Computes the fraction of padding and eod tokens in a given number of
    batches.

    Args:
        iterator: The iterator to load the data from.
        max_batches: The number of batches to load for computing the stats.
        eod_token_id: The end-of-document token id.
        padding_token_id: The padding token id.

    Returns:
        dict containing statistics: fraction of padding tokens and eod tokens in the loaded batches.
    """
    batches = list(itertools.islice(iterator, max_batches))
    n_padded = sum([(batch.inputs == padding_token_id).sum().item() for batch in batches])
    n_eod = sum([(batch.inputs == eod_token_id).sum().item() for batch in batches])
    n_total = sum([batch.inputs.size for batch in batches])
    stats = dict(
        frac_padded=n_padded / n_total,
        frac_eod=n_eod / n_total,
    )
    return stats


def _create_data_config(
    dataset_name: str,
    use_array_records: bool,
    context_length: int,
    global_batch_size: int,
    tokenizer_path: str = "gpt2",
) -> tuple[HFHubDataConfig, HFHubDataConfig]:
    """Helper function to create a data config object, setting several default values.

    Args:
        dataset_name: The name of the dataset to load. Must be one of the keys in DatasetNameToHFPath.
        use_array_records: Whether to use array records or huggingface dataset for data loading.
        context_length: The context length.
        global_batch_size: The global batch size.
        tokenizer_path: The path to the tokenizer.

    Returns:
        tuple of train and eval HFHubDataConfig objects with dataset configurations.
    """
    shared_config = dict(
        global_batch_size=global_batch_size,
        max_target_length=context_length,
        hf_cache_dir="/nfs-gpu/xlstm/data/hf_cache",
        data_column="text",
        tokenize_data=True,
        tokenizer_path=tokenizer_path,
        add_bos=False,
        add_eos=False,
        add_eod=True,
        grain_packing=True,
    )
    if use_array_records:
        train_config, eval_config = GrainArrayRecordsDataConfig.create_train_eval_configs(
            data_path=Path(DatasetNameToArrayRecordsPath[dataset_name]),
            **shared_config,
        )
    else:
        train_config, eval_config = HFHubDataConfig.create_train_eval_configs(
            hf_path=Path(DatasetNameToHFPath[dataset_name]).as_posix(),
            hf_num_data_processes=min(len(os.sched_getaffinity(0)), 100),
            **shared_config,
        )
    return train_config, eval_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute stats of packing operations for a huggingface dataset.")
    parser.add_argument("--dataset_name", type=str, choices=DatasetNameToHFPath, default="DCLM")
    parser.add_argument("--max_batches", type=int, default=50)
    parser.add_argument("--use_array_records", type=bool, default=True)
    args = parser.parse_args()
    compute_and_tabulate_packed_batch_statistics(
        dataset_name=args.dataset_name,
        use_array_records=args.use_array_records,
        max_batches=args.max_batches,
    )
