#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import argparse
import enum
import logging
import math
import os
import sys
from multiprocessing import Pool
from pathlib import Path
from time import time

import numpy as np
from array_record.python.array_record_module import ArrayRecordWriter

from xlstm_jax.dataset.grain_data_processing import load_array_record_dataset

LOGGER = logging.getLogger(__name__)


class DatasetNameToArrayRecordsPath(enum.StrEnum):
    DCLM = "/nfs-gpu/xlstm/data/array_records/mlfoundations_dclm-baseline-1.0-parquet"


class DatasetNameToSize(enum.IntEnum):
    """The number of examples in each dataset. Required because loading dataset in main and sub-processes gets stuck"""

    DCLM = 2_949_254_346


SPLIT_FILES_FOLDER = "split_indices"


def split_array_records_dataset(
    dataset_name: DatasetNameToArrayRecordsPath,
    num_processes: int = None,
    source_split: str = "train",
    num_eval_examples: int = 500_000,
    shard_size: int = 250_000,
    random_seed: int = 42,
    exists_ok: bool = False,
):
    """Split array records dataset into train and eval datasets.

    Args:
        dataset_name: The name of the dataset to split.
        num_processes: The number of processes to use for writing the splits.
        source_split: The split to load from the source dataset.
        num_eval_examples: The number of examples to use for the validation split. The rest is for training
        shard_size: The maximum number of examples to write into a single shard.
        random_seed: The random seed to use for the split.
        exists_ok: If True, allow the creation of the split indices folder and split folders if they do not exist.
    """
    # ***** Load source Dataset *****
    # Set paths
    source_base_path = Path(DatasetNameToArrayRecordsPath[dataset_name])
    source_split_path = source_base_path / source_split
    target_base_path = source_base_path.parent / f"{source_base_path.name}-split"
    target_base_path_for_split_files = target_base_path / SPLIT_FILES_FOLDER

    # Create directories. Sub-folder containing the split indices is allowed to create only if exists_ok=True.
    target_base_path.mkdir(parents=True, exist_ok=True)
    target_base_path_for_split_files.mkdir(parents=True, exist_ok=exists_ok)

    # ***** Save split indices *****
    LOGGER.info(f"Creating random split with {num_eval_examples} validation samples using random seed {random_seed}.")
    # Create random example indices. Note: indices make up 22GB for DCLM.
    num_examples = DatasetNameToSize[dataset_name]
    indices = np.arange(num_examples, dtype=np.int64)
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices = indices[num_eval_examples:]
    validation_indices = indices[:num_eval_examples]
    del indices  # free memory. We will reuse this variable name below.
    train_indices.sort()
    validation_indices.sort()

    # Write indices to .npy files. (We would already have raised an error if exists_ok=False and folder exists)
    LOGGER.info("Saving (train, validation) split indices.")
    np.save(target_base_path_for_split_files / "train_indices.npy", train_indices)
    np.save(target_base_path_for_split_files / "validation_indices.npy", validation_indices)

    # Write train, validation splits using ArrayRecordWriter in multiple processes.
    LOGGER.info(f"Writing (train, validation) splits to {target_base_path}.")
    split_names = ["validation", "train"]
    split_indices = [validation_indices, train_indices]

    for split, indices in zip(split_names, split_indices):
        out_path = target_base_path / split
        out_path.mkdir(parents=True, exist_ok=exists_ok)
        LOGGER.info(f"Writing {split} dataset to array record.")
        if num_processes is None or num_processes <= 1:
            write_array_record(
                dataset_path=source_split_path,
                indices=indices,
                split=split,
                out_path=out_path,
                process_idx=0,
                shard_start_idx=0,
                shard_size=shard_size,
            )
        else:
            # We split the indices into num_processes chunks (start and end idx) and convert each chunk in parallel.
            # Since the number of shards is in general not divisible by the number of processes, we distribute the
            # residual shards among the first 'num_residuals' processes.
            num_shards = math.ceil(len(indices) / shard_size)
            num_split_processes = num_processes if num_processes <= num_shards else num_shards  # avoid empty processes
            min_shards_per_process = num_shards // num_split_processes
            num_residuals = num_shards % num_split_processes

            num_shards_per_process = np.full(num_split_processes, min_shards_per_process, dtype=np.int32)
            num_shards_per_process[:num_residuals] += 1
            assert num_shards_per_process.sum() == num_shards, "shards per process does not sum to total shards."
            shard_start_indices = np.cumsum(num_shards_per_process) - num_shards_per_process

            # Start and end indices of the numpy array holding the random/shuffled example indices.
            iter_start_indices = shard_size * shard_start_indices
            iter_end_indices = iter_start_indices + shard_size * num_shards_per_process
            iter_end_indices[-1] = len(indices)

            with Pool(num_split_processes) as pool:
                pool.starmap(
                    write_array_record,
                    [
                        (
                            source_split_path,
                            indices[iter_start_indices[process_idx] : iter_end_indices[process_idx]],
                            split,
                            out_path,
                            process_idx,
                            shard_start_indices[process_idx],
                            shard_size,
                        )
                        for process_idx in range(num_split_processes)
                    ],
                )

        LOGGER.info(f"Finished conversion of {split} dataset.")


def write_array_record(
    dataset_path: Path,
    indices: np.ndarray | list[int],
    split: str,
    out_path: Path,
    process_idx: int,
    shard_start_idx: int,
    shard_size: int,
):
    """Write the dataset split to an array record file.

    Args:
        dataset_path: The path to the dataset to split.
        indices: The indices to select from the dataset.
        split: The dataset split to write.
        out_path: The output path (folder) for the array record files.
        process_idx: The index of the process writing the array record file. Used for logging only.
        shard_start_idx: The index of the first shard to write, used to determine the shard index for the file name.
        shard_size: The number of examples in each shard.
    """

    dataset = load_array_record_dataset(dataset_path=dataset_path)
    writer = None
    for iter_idx, example_idx in enumerate(indices):
        shard_idx = shard_start_idx + iter_idx // shard_size
        example = dataset[example_idx]

        if writer is None:
            LOGGER.info(f"[Process {process_idx}] Writing shard {shard_idx}.")
            shard_path = (out_path / f"{split}_{shard_idx:06d}.arecord").absolute().as_posix()
            assert not os.path.exists(shard_path), f"Shard file already exists: {shard_path}"  # Do not overwrite
            writer = ArrayRecordWriter(shard_path, "group_size:1")
        writer.write(example)
        if (iter_idx + 1) % shard_size == 0:
            LOGGER.info(f"[Process {process_idx}] Finished shard {shard_idx}.")
            writer.close()
            writer = None
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    # Set up logging.
    LOG_FORMAT = "[%(asctime)s][%(name)s:%(lineno)d][%(levelname)s]{rank} - %(message)s"
    stdout_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(handlers=[stdout_handler], level="INFO", format=LOG_FORMAT.format(rank=""), force=True)

    # Parameters.
    parser = argparse.ArgumentParser(description="Split array records dataset into train and validation.")
    parser.add_argument(
        "--dataset_name", type=str, choices=list(e.name for e in DatasetNameToArrayRecordsPath), default="DCLM"
    )
    parser.add_argument("--num_processes", type=int, default=100, help="Number of workers to write splits.")
    parser.add_argument("--num_eval_examples", type=int, default=500_000, help="Size of the validation split.")
    parser.add_argument("--exists_ok", action="store_true", help="Allow creation of folders if they do not exist.")
    args = parser.parse_args()

    start = time()
    split_array_records_dataset(
        dataset_name=args.dataset_name,
        num_processes=args.num_processes,
        num_eval_examples=args.num_eval_examples,
        exists_ok=args.exists_ok,
    )
    stop = time()
    LOGGER.info(f"Done splitting dataset took {stop - start:.2f} seconds.")
