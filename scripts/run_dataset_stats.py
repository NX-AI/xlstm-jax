import os

# We disable GPU usage for this script, as it is not needed.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

# flake8: noqa E402
import argparse
import enum
import itertools
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from time import time
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate as python_tabulate

from xlstm_jax.dataset import (
    GrainArrayRecordsDataConfig,
    HFHubDataConfig,
    LLMBatch,
    create_data_iterator,
    load_tokenizer,
)
from xlstm_jax.dataset.multihost_dataloading import MultiHostDataLoadIterator
from xlstm_jax.distributed.mesh_utils import initialize_mesh
from xlstm_jax.models.configs import ParallelConfig


class DatasetNameToHFPath(enum.StrEnum):
    SlimPajama6B = "DKYoon/SlimPajama-6B"
    SlimPajama627B = "cerebras/SlimPajama-627B"
    DCLM = "mlfoundations/dclm-baseline-1.0-parquet"
    FinewebEdu = "HuggingFaceFW/fineweb-edu"


class DatasetNameToArrayRecordsPath(enum.StrEnum):
    SlimPajama6B = "/nfs-gpu/xlstm/data/array_records/DKYoon_SlimPajama-6B"
    SlimPajama627B = "/nfs-gpu/xlstm/data/array_records/cerebras_SlimPajama-627B"
    DCLM = "/nfs-gpu/xlstm/data/array_records/mlfoundations_dclm-baseline-1.0-parquet-split"
    FinewebEdu = "/nfs-gpu/xlstm/data/array_records/HuggingFaceFW_fineweb-edu"


def compute_and_tabulate_packed_batch_statistics(
    dataset_name: str, tokenizer_path: str, use_array_records: bool = True, max_batches: int = 50
) -> str:
    """Computes and tabulates the fraction of padding and eod tokens for a few
    combinations of different batch size and context length.

    Args:
        dataset_name: The name of the dataset to load. Must be one of the keys in DatasetNameToHFPath.
        tokenizer_path: The path to/name of the tokenizer.
        use_array_records: Whether to use array records or huggingface dataset for data loading.
        max_batches: Maximum number of batches to load for computing the stats.

    Returns:
        Tabulated string with the statistics for different batch sizes and context lengths.
    """
    print(
        f"Computing statistics using {'array records' if use_array_records else 'huggingface'} dataset {dataset_name}."
    )

    parallel = ParallelConfig()
    mesh = initialize_mesh(parallel_config=parallel)

    train_data_config, _ = _create_data_config(
        dataset_name=dataset_name,
        use_array_records=use_array_records,
        context_length=-1,
        global_batch_size=-1,
        tokenizer_path=tokenizer_path,
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
        (128, 2048),
        (128, 8192),
        (32, 32768),
        (32, 65536),
        (32, 131072),
    ]

    # Create a folder to save the statistics.
    base_folder = Path(f"logs/datastats_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    base_folder.mkdir(parents=True, exist_ok=True)

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
            tokenizer_path=train_data_config.tokenizer_path,
        )
        iterator = create_data_iterator(config=_train_data_config, mesh=mesh)

        stats = compute_packed_batch_statistics(
            iterator=iterator,
            eod_token_id=tokenizer.eos_token_id,  # eos is used for eod.
            max_batches=max_batches,
        )

        summary["batch_size"].append(batch_size)
        summary["context_length"].append(context_length)
        for key, value in stats.items():
            if isinstance(value, (float, int)):
                summary[key].append(value)
            elif isinstance(value, dict) and isinstance(value.get("data"), np.ndarray):
                print(f"Plotting stats for {key}...")
                plot_data_stats(
                    value, title=key, file_path=base_folder / f"{key}_bs{batch_size}_ctx{context_length}.png"
                )
            else:
                print(f"Skipping {key} of type {type(value)}.")

    stop = time()
    print(f"Iterator loading and stats computation time: {stop - start:.2f} seconds")
    scalar_summary = {k: v for k, v in summary.items() if isinstance(v[0], (float, int))}
    table_str = python_tabulate(scalar_summary, headers="keys", intfmt="_", floatfmt=".4f")
    print(table_str)

    # Save summary to file.
    with open(base_folder / "summary.txt", "w") as f:
        f.write(table_str)
    with open(base_folder / "summary.json", "w") as f:
        json.dump(summary, f, indent=4)
    return table_str


def plot_data_stats(value_dict: dict[str, Any], file_path: Path, title: str = "") -> None:
    """Plots a histogram of the values.

    Args:
        value_dict: The values to plot. Should be dictionary with key "data" containing the values. Can be used to
            further customize the plot, with keys "type", "bins", "title", "xlabel", "ylabel", "xlim", "ylim". The
            "type" key should be either "hist" or "line", describing the type of plot.
        file_path: The path to save the plot.
        title: Default title and xlabel to use if not provided in value_dict.
    """
    data = value_dict["data"]
    plt.figure()
    if value_dict["type"] == "hist":
        plt.hist(data, bins=value_dict.get("bins", 128), density=True)
    elif value_dict["type"] == "line":
        plt.plot(data)
    else:
        raise ValueError(f"Unknown plot type {value_dict['type']}.")
    plt.title(value_dict.get("title", title))
    plt.ylabel(value_dict.get("ylabel", "Frequency"))
    plt.xlabel(value_dict.get("xlabel", title))
    plt.xlim(value_dict.get("xlim", None))
    plt.ylim(value_dict.get("ylim", None))
    plt.tight_layout()
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()
    np.savez(file_path.with_suffix(".npz"), data=data)


def compute_packed_batch_statistics(
    iterator: MultiHostDataLoadIterator,
    eod_token_id: int,
    max_batches: int = 50,
) -> dict[str, float | np.ndarray]:
    """Computes the fraction of padding and eod tokens in a given number of
    batches.

    Args:
        iterator: The iterator to load the data from.
        eod_token_id: The end-of-document token id.
        max_batches: The number of batches to load for computing the stats.

    Returns:
        dict containing statistics about the packed batches.
    """
    # Load batches.
    batches: list[LLMBatch] = list(itertools.islice(iterator, max_batches))
    batches = jax.device_get(batches)
    print("Loaded", len(batches), "batches.")
    # Attributes.
    num_batches = len(batches)
    batch_size, context_length = batches[0].inputs.shape
    # Padded and EOD tokens.
    n_padded = sum([(batch.inputs_segmentation == 0).sum().item() for batch in batches])
    n_eod = sum([((batch.inputs == eod_token_id) * (batch.inputs_segmentation != 0)).sum().item() for batch in batches])
    n_total = num_batches * batch_size * context_length
    # Num documents per sequence.
    n_docs = sum([jnp.max(batch.targets_segmentation, axis=-1).sum().item() for batch in batches])
    n_docs_per_seq = n_docs / (batch_size * num_batches)
    # Document lengths.
    n_full_seqs = sum(
        [
            ((batch.targets_segmentation[:, -1] != 0) * (batch.targets[:, -1] != eod_token_id)).sum().item()
            for batch in batches
        ]
    )
    doc_lens = [
        batch.targets_position[np.logical_and(batch.targets == eod_token_id, batch.targets_segmentation != 0)].flatten()
        for batch in batches
    ]
    doc_lens.append(np.full(n_full_seqs, context_length))  # Full sequences have no EOD token.
    doc_lens = np.concatenate(doc_lens)
    assert n_docs == doc_lens.shape[0], (
        "Two ways of calculating number of documents got different results, likely some error in segmentations or "
        f"incorrectly used EOD tokens: n_docs={n_docs} != doc_lens.shape[0]={doc_lens.shape[0]}."
    )
    # Accumulative token count over document sizes.
    doc_lens_hist = np.bincount(doc_lens, minlength=context_length).astype(np.int64)
    acc_token_count = np.cumsum(doc_lens_hist * np.arange(doc_lens_hist.shape[0]))
    full_token_count = acc_token_count[-1].item()
    frac_acc_token_count = acc_token_count.astype(np.float64) / full_token_count
    # Tokens lost if we truncate at each document length.
    inv_doc_cumsum = np.cumsum(doc_lens_hist[::-1])[::-1]
    inv_doc_cumsum = np.pad(inv_doc_cumsum[1:], (0, 1))
    tokens_kept = acc_token_count + inv_doc_cumsum * np.arange(inv_doc_cumsum.shape[0])
    tokens_lost = full_token_count - tokens_kept
    frac_tokens_lost = tokens_lost / full_token_count
    # Token entropy.
    max_token = max([batch.targets.max().item() for batch in batches])
    token_counts = np.zeros(max_token + 1, dtype=np.float32)
    for batch in batches:
        token_counts += np.bincount(batch.targets.flatten(), minlength=max_token + 1)
    token_counts = token_counts / token_counts.sum()
    token_entropy = -np.sum(token_counts * np.log(token_counts + 1e-12)).item()
    # Compute final stats.
    stats = dict(
        frac_padded=n_padded / n_total,
        frac_eod=n_eod / n_total,
        frac_full_seqs=n_full_seqs / (num_batches * batch_size),
        frac_full_docs=n_full_seqs / n_docs,
        n_docs_per_seq=n_docs_per_seq,
        token_entropy=token_entropy,
        doc_lens=dict(
            type="hist",
            data=doc_lens,
            bins=128,
            xlim=(0, context_length),
            title="Document Lengths",
            xlabel="Document Length",
        ),
        acc_token_count=dict(
            type="line",
            data=frac_acc_token_count,
            xlim=(0, context_length),
            ylim=(0, 1),
            title="Accumulative Token Count",
            xlabel="Document Length",
        ),
        frac_tokens_lost=dict(
            type="line",
            data=frac_tokens_lost,
            xlim=(0, context_length),
            ylim=(0, 1),
            title="Fraction of Tokens Lost at Document Length",
            xlabel="Document Length",
        ),
    )
    return stats


def _create_data_config(
    dataset_name: str,
    use_array_records: bool,
    context_length: int,
    global_batch_size: int,
    tokenizer_path: str,
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
        worker_count=0,  # Preprocessing can be done in main process.
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
    parser.add_argument("--dataset_name", type=str, choices=list(e.name for e in DatasetNameToHFPath), default="DCLM")
    parser.add_argument("--tokenizer_path", type=str, default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--max_batches", type=int, default=50)
    parser.add_argument("--use_array_records", type=bool, default=True)
    args = parser.parse_args()
    compute_and_tabulate_packed_batch_statistics(
        dataset_name=args.dataset_name,
        tokenizer_path=args.tokenizer_path,
        use_array_records=args.use_array_records,
        max_batches=args.max_batches,
    )
