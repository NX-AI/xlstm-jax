import argparse
import logging
import math
import os
import sys
from collections.abc import Sequence
from multiprocessing import Pool
from pathlib import Path

import datasets
import numpy as np
from array_record.python.array_record_module import ArrayRecordWriter

from xlstm_jax.dataset import HFHubDataConfig

LOGGER = logging.getLogger(__name__)


def write_array_record(
    config: HFHubDataConfig,
    split: str,
    out_path: Path,
    process_idx: int,
    shard_start_idx: int,
    example_start_idx: int,
    example_end_idx: int,
    shard_size: int,
):
    """Write the dataset split to an array record file.

    Args:
        config: The configuration for the HuggingFace dataset.
        split: The dataset split to write. Must be one of ("train", "validation", "test"), not e.g. "train[0:10]".
        out_path: The output path (folder) for the array record files.
        process_idx: The index of the process writing the array record file. Used for logging only.
        shard_start_idx: The index of the first shard to write, used to determine the shard index for the file name.
        example_start_idx: The index of the first example to write.
        example_end_idx: The index of the last example to write.
        shard_size: The number of examples in each shard.
    """
    assert split in ("train", "validation", "test"), f"Invalid split: {split}"
    process_split = f"{split}[{example_start_idx}:{example_end_idx}]"  # chunk for this process

    LOGGER.info(f"[Process {process_idx}] Loading {process_split} dataset.")
    ds = datasets.load_dataset(
        config.hf_path,
        data_dir=config.hf_data_dir,
        data_files=config.hf_train_files,
        cache_dir=config.hf_cache_dir,
        split=process_split,
        streaming=False,
        token=config.hf_access_token,
        num_proc=None,  # no multiprocessing inside process.
    )

    LOGGER.info(f"[Process {process_idx}] Finished loading {process_split} dataset.")
    writer = None
    for ex_idx, example in enumerate(ds):
        shard_idx = shard_start_idx + ex_idx // shard_size
        if writer is None:
            LOGGER.info(f"[Process {process_idx}] Writing shard {shard_idx}.")
            shard_path = (out_path / f"{split}_{shard_idx:06d}.arecord").absolute().as_posix()
            assert not os.path.exists(shard_path), f"Shard file already exists: {shard_path}"  # Do not overwrite
            # "group_size:1", because __get_item__ of ArrayRecordDataSource runs into the following logging error:
            # https://github.com/google/array_record/blob/main/python/
            # The API is of C++ ArrayRecordWriter is unfortunately not documented (afaik). But here is an example:
            # https://github.com/google/array_record/blob/main/python/array_record_module_test.py#L135C48-L135C63
            writer = ArrayRecordWriter(shard_path, "group_size:1")
        writer.write(str.encode(example["text"]))
        if (ex_idx + 1) % shard_size == 0:
            LOGGER.info(f"[Process {process_idx}] Finished shard {shard_idx}.")
            writer.close()
            writer = None
    if writer is not None:
        writer.close()
    LOGGER.info(f"[Process {process_idx}] Finished writing {process_split} dataset.")


def convert_dataset(
    hf_path: str,
    hf_data_name: str | None = None,
    hf_data_dir: str | None = None,
    splits: Sequence[str] = ("validation", "train"),
    num_processes: int | None = None,
    shard_size: int = 250_000,
    base_out_path: Path = Path("/nfs-gpu/xlstm/data/array_records"),
    hf_cache_dir: Path = Path("/nfs-gpu/xlstm/data/hf_cache"),
):
    """Convert dataset from HuggingFace to ArrayRecord.

    Args:
        hf_path: Huggingface dataset path.
        hf_data_name: Huggingface dataset name.
        hf_data_dir: Huggingface dataset directory.
        splits: The dataset splits to preprocess.
        num_processes: Number of workers to use for the convert/map preprocessing.
        shard_size: Number of examples in each shard.
        base_out_path: Base output directory for saving the preprocessed dataset.
        hf_cache_dir: Huggingface cache directory.
    """
    LOGGER.info(f"Converting to array_records with {num_processes} workers")
    # Dataset Configuration
    config = HFHubDataConfig(
        global_batch_size=1,
        max_target_length=-1,
        hf_path=hf_path,
        hf_data_dir=hf_data_dir,
        hf_cache_dir=hf_cache_dir,
        train_data_column="text",
        eval_data_column="text",
        tokenizer_path="gpt2",
        add_bos=True,
        add_eos=False,
        add_eod=True,
    )
    LOGGER.info(f"Dataset configuration: {config}")

    for split in splits:
        # Load dataset from hub/cache in order to get the dataset size.
        LOGGER.info(f"Loading {split} dataset.")

        ds = datasets.load_dataset(
            config.hf_path,
            name=hf_data_name,
            data_dir=config.hf_data_dir,
            data_files=config.hf_train_files,
            cache_dir=config.hf_cache_dir,
            split=split,
            streaming=False,
            token=config.hf_access_token,
            num_proc=None,  # multiprocessing does not give speedup for some reason.
        )
        dataset_size = len(ds)
        del ds  # free memory before reloading dataset chunks in multiple processes

        out_path = base_out_path / config.hf_path.replace("/", "_")
        if hf_data_dir is not None:
            out_path = out_path / hf_data_dir
        out_path = out_path / split
        # Allow creating the directory if it does not exist. But assert that it is empty to avoid overwriting data.
        out_path.mkdir(parents=True, exist_ok=True)
        assert os.listdir(out_path) == [], f"Output directory ({out_path}) is not empty: {os.listdir(out_path)}"

        # Write the dataset to an array record file.
        LOGGER.info(f"Writing {split} dataset to array record.")
        if num_processes is None or num_processes <= 1:
            write_array_record(
                config=config,
                split=split,
                out_path=out_path,
                process_idx=0,
                shard_start_idx=0,
                example_start_idx=0,
                example_end_idx=dataset_size,
                shard_size=shard_size,
            )
        else:
            # We split the dataset into num_processes chunks (start and end idx) and convert each chunk in parallel.
            # Since the number of shards is in general not divisible by the number of processes, we distribute the
            # residual shards among the first 'num_residuals' processes.
            num_shards = math.ceil(dataset_size / shard_size)
            num_processes = num_processes if num_processes <= num_shards else num_shards  # avoid empty processes
            min_shards_per_process = num_shards // num_processes
            num_residuals = num_shards % num_processes

            num_shards_per_process = np.full(num_processes, min_shards_per_process, dtype=np.int32)
            num_shards_per_process[:num_residuals] += 1
            assert num_shards_per_process.sum() == num_shards, "shards per process does not sum to total shards."
            shard_start_indices = np.cumsum(num_shards_per_process) - num_shards_per_process
            example_start_indices = shard_size * shard_start_indices
            example_end_indices = example_start_indices + shard_size * num_shards_per_process
            example_end_indices[-1] = dataset_size
            with Pool(num_processes) as pool:
                pool.starmap(
                    write_array_record,
                    [
                        (
                            config,
                            split,
                            out_path,
                            process_idx,
                            shard_start_indices[process_idx],
                            example_start_indices[process_idx],
                            example_end_indices[process_idx],
                            shard_size,
                        )
                        for process_idx in range(num_processes)
                    ],
                )

        LOGGER.info(f"Finished conversion of {split} dataset.")


if __name__ == "__main__":
    # Set up logging.
    LOG_FORMAT = "[%(asctime)s][%(name)s:%(lineno)d][%(levelname)s]{rank} - %(message)s"
    stdout_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(
        handlers=[stdout_handler],
        level="INFO",
        format=LOG_FORMAT.format(rank=""),
        force=True,
    )

    # Processing parameters.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        # "--hf_path", type=str, default="DKYoon/SlimPajama-6B", help="Huggingface dataset path."
        "--hf_path",
        type=str,
        default="mlfoundations/dclm-baseline-1.0-parquet",
        help="Huggingface dataset path.",
    )
    parser.add_argument("--hf_data_dir", type=str, default=None, help="Huggingface dataset directory.")
    parser.add_argument("--splits", type=str, nargs="+", default=["train"], help="Dataset splits to convert.")
    parser.add_argument("--num_processes", type=int, default=None, help="Number of workers used to convert.")
    args = parser.parse_args()

    # Convert dataset to ArrayRecords.
    convert_dataset(
        hf_path=args.hf_path,
        hf_data_name=None,
        hf_data_dir=args.hf_data_dir,
        splits=args.splits,
        num_processes=args.num_processes if args.num_processes is not None else min(len(os.sched_getaffinity(0)), 128),
    )
