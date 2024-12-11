#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import pytest
from jax.sharding import PartitionSpec as P

from xlstm_jax.dataset import create_data_iterator
from xlstm_jax.dataset.batch import LLMBatch
from xlstm_jax.dataset.configs import SyntheticDataConfig
from xlstm_jax.distributed.mesh_utils import initialize_mesh
from xlstm_jax.models.configs import ParallelConfig


@pytest.mark.parametrize("num_train_batches,num_val_batches", [(8, 5), (10, 15), (12, 12)])
@pytest.mark.parametrize("global_batch_size", [16, 32, 256])
@pytest.mark.parametrize("tp_size,fsdp_size", [(1, 1), (2, 2), (1, 4), (4, 1)])
def test_synthetic_dataloader(
    num_train_batches: int, num_val_batches: int, global_batch_size: int, tp_size: int, fsdp_size: int
):
    """Test data loading with synthetic data."""
    if pytest.num_devices < tp_size * fsdp_size:
        pytest.skip("Test requires more devices than available.")
    parallel = ParallelConfig(
        data_axis_name="dp",
        fsdp_axis_name="fsdp",
        model_axis_name="tp",
        pipeline_axis_name="pp",
        fsdp_modules=("Embed", "LMHead", "mLSTMBlock"),
        fsdp_min_weight_size=2**8,
        fsdp_axis_size=fsdp_size,
        model_axis_size=tp_size,
        data_axis_size=-1,
    )

    # Initialize mesh.
    mesh = initialize_mesh(init_distributed_on_slurm=False, parallel_config=parallel)

    # Specify the configuration for the synthetic data.
    max_target_length = 128
    train_config, val_config = SyntheticDataConfig.create_train_eval_configs(
        train_kwargs={"num_batches": num_train_batches},
        eval_kwargs={"num_batches": num_val_batches},
        global_batch_size=global_batch_size,
        max_target_length=max_target_length,
        data_shuffle_seed=42,
    )

    # Define data iterator. The synthesic data iterator will not return an evaluation iterator.
    train_iterator = create_data_iterator(config=train_config, mesh=mesh)
    val_iterator = create_data_iterator(config=val_config, mesh=mesh)

    assert (
        len(train_iterator) == num_train_batches
    ), f"Expected {num_train_batches} train batches, got {len(train_iterator)}."
    assert len(val_iterator) == num_val_batches, f"Expected {num_val_batches} val batches, got {len(val_iterator)}."

    # Make sure that batches are LLMBatches of the correct shape.
    for dataloader in [train_iterator, val_iterator]:
        for epoch in range(3):
            count = 0
            for batch in dataloader:
                count += 1
                assert isinstance(batch, LLMBatch), f"Expected LLMBatch, got {type(batch)}."
                assert batch.inputs.shape == (
                    global_batch_size,
                    max_target_length,
                ), f"Expected shape {(global_batch_size, max_target_length)}, got {batch.inputs.shape}."
                assert batch.inputs.sharding.spec == P(
                    mesh.axis_names
                ), f"Expected sharding {P(mesh.axis_names)}, got {batch.inputs.sharding.spec}."
            assert count == len(dataloader), f"Expected {len(dataloader)} batches, got {count} in epoch {epoch}."
