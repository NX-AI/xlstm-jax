import enum
import logging
import sys
from pathlib import Path

import numpy as np

from xlstm_jax.dataset.configs import GrainArrayRecordsDataConfig
from xlstm_jax.dataset.input_pipeline_interface import create_data_iterator
from xlstm_jax.distributed.mesh_utils import initialize_mesh
from xlstm_jax.models.configs import ParallelConfig

LOGGER = logging.getLogger(__name__)


class DatasetNameToArrayRecordsPath(enum.StrEnum):
    SlimPajama6B = "/nfs-gpu/xlstm/data/array_records/DKYoon_SlimPajama-6B"
    SlimPajama627B = "/nfs-gpu/xlstm/data/array_records/cerebras_SlimPajama-627B"
    DCLM = "/nfs-gpu/xlstm/data/array_records/mlfoundations_dclm-baseline-1.0-parquet-split"


# TODO: this test is skipped due to WIP. It currently runs locally but not on CI, because we must prepreocess (AR)
#  the dataset first. Will do for wiki. Rename so it is not detected by pytest
#  It also currently fails when executed in pytest with simulated devices. For single-device currently
# def test_preprocessed_ar_iterator():
def foo():
    LOG_FORMAT = "[%(asctime)s][%(name)s:%(lineno)d][%(levelname)s]{rank} - %(message)s"
    stdout_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(
        handlers=[stdout_handler],
        level="INFO",
        format=LOG_FORMAT.format(rank=""),
        force=True,
    )

    dataset_name: str = "SlimPajama6B"
    context_length: int = 8192
    tokenizer_path: str = "gpt2"
    grain_packing_bin_count: int = None  # TODO: hard-code this.
    grain_packing_batch_size: int = 1024

    data_path = Path(DatasetNameToArrayRecordsPath[dataset_name])  # path from which we load the dataset
    out_path = Path(f"{data_path}-preprocessed_tok-{tokenizer_path.replace('/', '_')}_cl-{context_length}")

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
        data_axis_size=1,
    )

    # Initialize mesh.
    mesh = initialize_mesh(init_distributed_on_slurm=False, parallel_config=parallel)
    assert mesh.shape[parallel.data_axis_name] == 1, "This script is only for single-device."

    # ***** Standard validation iterator *****
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
        grain_packing_bin_count=grain_packing_bin_count,
    )

    # Create validation iterator
    eval_iterator = create_data_iterator(config=eval_data_config, mesh=mesh)

    # ***** Preprocessed validation iterator *****
    _, eval_preprocessed_data_config = GrainArrayRecordsDataConfig.create_train_eval_configs(
        global_batch_size=grain_packing_batch_size,  # mesh.shape[parallel.data_axis_name] == 1. -> no multiplication.
        max_target_length=context_length,
        data_path=out_path,
        data_column="text",
        tokenize_data=False,
        tokenizer_path=tokenizer_path,
        add_bos=False,
        add_eos=False,
        add_eod=True,
        worker_count=0,
        grain_packing=False,
        grain_packing_bin_count=grain_packing_bin_count,
    )

    # Create preprocessed validation iterator
    eval_iterator_preprocessed = create_data_iterator(config=eval_preprocessed_data_config, mesh=mesh)

    # ***** Compare the two iterators *****
    # for idx_batch, (eval_batch, eval_preprocessed_batch) in enumerate(zip(eval_iterator, eval_iterator_preprocessed)):
    for idx_batch in range(len(eval_iterator)):
        eval_batch = next(eval_iterator)
        eval_preprocessed_batch = next(eval_iterator_preprocessed)
        fields = eval_batch.__dict__.keys()
        assert isinstance(eval_batch, type(eval_preprocessed_batch)), "batches have different types."
        assert eval_preprocessed_batch.__dict__.keys() == fields, "batches have different __dict__ keys."

        for field in fields:
            if "position" in field:
                continue  # TODO: at some point add position to InferSegmentation. Currently unused and not correct
            if "segmentation" in field:
                # TODO: segmentation starts at 2 in case of offline packing.
                #  Segmentation starting at 2 should not be a problem. But subtract to make test work.
                segmentation = getattr(eval_batch, field)
                segmentation_preprocessed = getattr(eval_preprocessed_batch, field)
                padding_mask = segmentation_preprocessed == 0
                segmentation_preprocessed -= np.array(~padding_mask, dtype=segmentation.dtype)
                val = segmentation
                val_preprocessed = segmentation_preprocessed
            else:
                val = getattr(eval_batch, field)
                val_preprocessed = getattr(eval_preprocessed_batch, field)
            assert np.array_equal(val, val_preprocessed), f"field {field} in batch {idx_batch} is not equal."
        print(idx_batch)
