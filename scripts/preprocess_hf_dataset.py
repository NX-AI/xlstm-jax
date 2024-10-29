import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import datasets
import transformers

from xlstm_jax.dataset import HFHubDataConfig
from xlstm_jax.dataset.hf_data_processing import preprocess_hf_dataset

LOGGER = logging.getLogger(__name__)


def preprocess_dataset(
    hf_path: str,
    hf_data_dir: str | None = None,
    context_length: int = 2048,
    splits: Sequence[str] = ("validation", "train"),
    num_map_processes: int | None = None,
    num_data_processes: int | None = None,
    base_out_path: Path = Path("/nfs-gpu/xlstm/data/hf_datasets"),
):
    """Preprocess dataset from HuggingFace.

    Tokenizes and groups the text data into sequences of a fixed length. Afterwards, the dataset is saved to disk
    in the arrow format. Note that the preprocessing maps generate caches for the dataset splits and saves them to
    disk, to ensure that breaking the preprocessing in between does not need a full restart. However, the cache
    files are not deleted after the preprocessing is finished, and may need to be deleted manually to save disk
    space (go to HF cache directory of the dataset and remove all "cache-*" and "tmp*" files).

    Args:
        hf_path: Huggingface dataset path.
        hf_data_dir: Huggingface dataset directory.
        context_length: Maximum length of the target sequence.
        splits: List of dataset splits to preprocess.
        num_map_processes: Number of workers to use for the map preprocessing. If None, no multiprocessing is used.
        num_data_processes: Number of workers to use for loading the dataset. If None, no multiprocessing is used.
        base_out_path: Base output path for saving the preprocessed dataset. The dataset will be saved to
            `base_out_path/hf_path/hf_data_dir/ctx{context_length}/{split}`.
    """
    LOGGER.info(f"Preprocessing dataset {hf_path}.")
    # Dataset Configuration
    config = HFHubDataConfig(
        global_batch_size=1,
        max_target_length=context_length,
        hf_path=hf_path,
        hf_data_dir=hf_data_dir,
        hf_cache_dir="/nfs-gpu/xlstm/data/hf_cache",
        hf_num_map_processes=num_map_processes,
        hf_num_data_processes=num_data_processes,
        data_column="text",
        tokenizer_path="gpt2",
        add_bos=True,
        add_eos=False,
        add_eod=True,
        shuffle_data=False,  # unused
    )
    LOGGER.info(f"Dataset configuration: {config}")

    for split in splits:
        # Load dataset from hub/cache.
        LOGGER.info(f"Loading {split} dataset.")
        dataset = datasets.load_dataset(
            config.hf_path,
            data_dir=config.hf_data_dir,
            data_files=config.hf_data_files,
            cache_dir=config.hf_cache_dir,
            split=split,
            streaming=False,
            token=config.hf_access_token,
            num_proc=config.hf_num_data_processes,
        )

        # Preprocess dataset.
        LOGGER.info(f"Preprocessing {split} dataset.")
        dataset = preprocess_hf_dataset(
            dataset,
            tokenizer_path=config.tokenizer_path,
            column_name=config.data_column,
            max_target_length=config.max_target_length,
            add_bos=config.add_bos,
            add_eos=config.add_eos,
            add_eod=config.add_eod,
            hf_access_token=config.hf_access_token,
            num_proc=config.hf_num_map_processes,
            tokenizer_cache_dir=config.hf_cache_dir,
        )

        # Create output path.
        out_path = base_out_path / config.hf_path.replace("/", "_")
        if hf_data_dir is not None:
            out_path = out_path / hf_data_dir
        out_path = out_path / f"ctx{context_length}"
        out_path.mkdir(parents=True, exist_ok=True)
        LOGGER.info(f"Saving to {out_path}.")

        # Save dataset to disk.
        dataset.save_to_disk(out_path / split, num_proc=min(64, num_map_processes))
        LOGGER.info("Finished saving train dataset.")

    # Load dataset from disk.
    LOGGER.info("Loading dataset from disk.")
    test_ds = datasets.load_from_disk(out_path / splits[0])

    # Print dataset information.
    LOGGER.info(f"Dataset columns: {test_ds.column_names}")
    LOGGER.info(f"First element: {test_ds[0]}")
    LOGGER.info(f"Length of dataset: {len(test_ds)}")
    LOGGER.info(f"Dataset size: {test_ds.dataset_size}")

    # Print example text.
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.tokenizer_path,
        clean_up_tokenization_spaces=False,  # See https://github.com/huggingface/transformers/issues/31884
        legacy=False,
        token=config.hf_access_token,
        use_fast=True,
        add_bos=config.add_bos,
        add_eos=config.add_eos,
        cache_dir=config.hf_cache_dir,
    )
    LOGGER.info(f"Example text: {tokenizer.decode(dataset[0]['text'])}")
    LOGGER.info("Finished preprocessing.")


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

    # Hyperparams.
    context_length = 2048
    hf_path = "cerebras/SlimPajama-627B"
    hf_data_dir = None
    workers = len(os.sched_getaffinity(0))
    splits = ["validation", "train"]

    # Preprocess dataset.
    preprocess_dataset(
        hf_path=hf_path,
        hf_data_dir=hf_data_dir,
        context_length=context_length,
        splits=splits,
        num_map_processes=workers,
    )
