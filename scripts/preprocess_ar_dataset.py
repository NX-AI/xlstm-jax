import argparse
import enum
import logging
import os
import sys
from pathlib import Path

import numpy as np
import tqdm
from array_record.python.array_record_module import ArrayRecordWriter

from xlstm_jax.dataset.configs import GrainArrayRecordsDataConfig
from xlstm_jax.dataset.grain_transforms import ParseTokenizedArrayRecords
from xlstm_jax.dataset.input_pipeline_interface import create_data_iterator
from xlstm_jax.distributed.mesh_utils import initialize_mesh
from xlstm_jax.models.configs import ParallelConfig

os.environ["JAX_PLATFORMS"] = "cpu"


LOGGER = logging.getLogger(__name__)


class DatasetNameToArrayRecordsPath(enum.StrEnum):
    SlimPajama6B = "/nfs-gpu/xlstm/data/array_records/DKYoon_SlimPajama-6B"
    SlimPajama627B = "/nfs-gpu/xlstm/data/array_records/cerebras_SlimPajama-627B"
    DCLM = "/nfs-gpu/xlstm/data/array_records/mlfoundations_dclm-baseline-1.0-parquet-split"


def preprocess_validation_set(
    dataset_name: str,
    grain_packing_batch_size: int,
    context_length: int = 8192,
    tokenizer_path: str = "gpt2",
    allow_overwrite: bool = False,
    eod_token_id: int = 0,
):
    """Preprocess the validation ArrayRecords dataset via tokenization and packing.

    Args:
        dataset_name: Name of the dataset to preprocess. Must be one of the keys of `DatasetNameToArrayRecordsPath`.
        grain_packing_batch_size: Batch size for grain packing.
        context_length: sequence length.
        tokenizer_path: Path to the tokenizer.
        allow_overwrite: Whether existing file may be overwritten.
        eod_token_id: ID of the EOD token.
    """
    # Define paths
    data_path = Path(DatasetNameToArrayRecordsPath[dataset_name])  # path from which we load the dataset
    out_path = Path(f"{data_path}-preprocessed_tok-{tokenizer_path.replace('/', '_')}_cl-{context_length}")
    out_path.mkdir(parents=True, exist_ok=True)

    # Define data configuration.
    parallel = ParallelConfig(
        data_axis_name="dp",
        fsdp_axis_name="fsdp",
        model_axis_name="tp",
        pipeline_axis_name="pp",
        fsdp_modules=["Embed", "LMHead", "mLSTMBlock"],
        fsdp_min_weight_size=2**8,
        fsdp_axis_size=1,
        model_axis_size=1,
        data_axis_size=-1,
    )

    # Initialize mesh.
    mesh = initialize_mesh(init_distributed_on_slurm=False, parallel_config=parallel)
    assert mesh.shape[parallel.data_axis_name] == 1, "This script is only for single-device."

    # Create validation set config
    _, eval_data_config = GrainArrayRecordsDataConfig.create_train_eval_configs(
        global_batch_size=grain_packing_batch_size,  # mesh.shape[parallel.data_axis_name] == 1. -> no multiplication.
        max_target_length=context_length,
        data_path=data_path,
        data_column="text",
        tokenize_data=True,
        tokenizer_path=tokenizer_path,
        add_bos=False,
        add_eos=False,
        add_eod=True,
        worker_count=0,
        grain_packing=True,
        grain_packing_bin_count=None,  # use the local batch size (which here is the global batch size)
    )

    # Create validation iterator
    eval_iterator = create_data_iterator(config=eval_data_config, mesh=mesh)

    # Write to disk. Since it is only the eval set, we just write into one file with a single process for simplicity.
    split_path = out_path / eval_data_config.split
    split_path.mkdir(parents=True, exist_ok=True)
    file_path = (split_path / f"{eval_data_config.split}_000000.arecord").as_posix()
    if not allow_overwrite:
        assert not Path(file_path).exists(), f"File {file_path} already exists. Set allow_overwrite=True to overwrite."

    LOGGER.info(f"Writing validation set to {out_path}...")
    writer = ArrayRecordWriter(file_path, "group_size:1")
    for batch in tqdm.tqdm(eval_iterator):
        for idx_example_in_batch in tqdm.tqdm(range(len(batch.targets)), leave=False):
            # Extract targets as examples (incl. padding from packing), because they are not shifted
            example_with_padding = batch.targets[idx_example_in_batch]
            # Remove padding and EOD token.
            is_padding = batch.targets_segmentation[idx_example_in_batch] == 0
            num_padding = int(np.sum(is_padding))
            if num_padding > 0:
                # only the end may contain paddings, but assert to be sure.
                assert is_padding[-num_padding] != 0, f"Position -{num_padding} should be padding."
                assert is_padding[-num_padding - 1] == 0, f"Position -{num_padding - 1} should not be padding."
                example = example_with_padding[: -num_padding - 1]  # remove padding and EOD token
            else:
                # It can happen that sequences were packed perfectly without padding and the last sequence ends with EOD
                if example_with_padding[-1] == eod_token_id:
                    example = example_with_padding[:-1]  # remove EOD token. Will get added in pipeline.
                else:
                    example = example_with_padding  # there is no EOD token, sequence was longer than context_length.
            # create numpy bytestring from the 'pure' token sequence (without padding and EOD token).
            preprocessed_example = ParseTokenizedArrayRecords.sequence_to_bytestring(example)
            writer.write(preprocessed_example)
    writer.close()
    LOGGER.info(f"Finished writing validation set to {out_path}.")


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

    parser = argparse.ArgumentParser("Preprocess a validation set.")
    parser.add_argument(
        "--dataset_name", type=str, choices=list(e.name for e in DatasetNameToArrayRecordsPath), default="DCLM"
    )
    parser.add_argument("--context_length", type=int, default=8192, help="Context length.")
    parser.add_argument("--tokenizer_path", type=str, default="EleutherAI/gpt-neox-20b", help="Path to the tokenizer.")
    parser.add_argument("--grain_packing_batch_size", type=int, default=1024, help="Batch size for grain packing.")
    parser.add_argument("--allow_overwrite", type=str, default=True, help="Whether existing file may be overwritten")
    args = parser.parse_args()

    if args.tokenizer_path == "gpt2":
        eod_token_id = 50256
    elif args.tokenizer_path == "EleutherAI/gpt-neox-20b":
        eod_token_id = 0
    else:
        raise NotImplementedError(f"Tokenizer {args.tokenizer_path} not implemented. Please add the EOD token ID.")

    preprocess_validation_set(
        dataset_name=args.dataset_name,
        context_length=args.context_length,
        tokenizer_path=args.tokenizer_path,
        grain_packing_batch_size=args.grain_packing_batch_size,
        allow_overwrite=args.allow_overwrite,
        eod_token_id=eod_token_id,
    )
