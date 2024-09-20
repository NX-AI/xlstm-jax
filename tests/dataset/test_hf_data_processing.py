from pathlib import Path

import numpy as np
import pytest
from jax.sharding import PartitionSpec as P

from xlstm_jax.dataset import HFDataConfig, create_data_iterator
from xlstm_jax.distributed.mesh_utils import initialize_mesh
from xlstm_jax.models.configs import ParallelConfig


@pytest.mark.skipif(not pytest.grain_available, reason="Grain is not available.")
def test_hf_dataset(tmp_path: Path):
    """Test data loading with huggingface datasets."""

    parallel = ParallelConfig(
        data_axis_name="dp",
        fsdp_axis_name="fsdp",
        model_axis_name="tp",
        pipeline_axis_name="pp",
        fsdp_modules=("Embed", "LMHead", "mLSTMBlock"),
        fsdp_min_weight_size=2**8,
        fsdp_axis_size=2,
        model_axis_size=2,
        data_axis_size=-1,
    )

    # Initialize mesh.
    mesh = initialize_mesh(parallel_config=parallel)

    # Define data configuration.
    global_batch_size = 64
    context_length = 128
    data_config = HFDataConfig(
        num_train_epochs=2,
        global_batch_size=global_batch_size,
        max_target_length=context_length,
        hf_path="Salesforce/wikitext",
        hf_data_dir="wikitext-2-v1",
        hf_eval_split="validation",
        hf_cache_dir=tmp_path / "hf_cache",
        train_data_column="text",
        eval_data_column="text",
        tokenize_train_data=True,
        tokenize_eval_data=True,
        tokenizer_path="gpt2",
        data_shuffle_seed=42,
        add_bos=True,
        add_eos=True,
    )

    # Define iterators
    train_iterator, eval_iterator = create_data_iterator(config=data_config, mesh=mesh)

    # Get batch from training iterator and make sure that the batch size is correct.
    train_batch = next(train_iterator)
    assert train_batch.inputs.shape == (global_batch_size, context_length)
    assert train_batch.inputs.sharding.spec == P(mesh.axis_names)

    # Get batches from evaluation iterator and make sure that the batch size is correct,
    # and that the batches are the same for each epoch.
    eval_batches = list()
    for epoch_idx in range(2):
        for batch in eval_iterator:
            assert batch.inputs.shape == (global_batch_size, context_length)
            assert batch.inputs.sharding.spec == P(mesh.axis_names)
            if epoch_idx == 0:
                eval_batches.append(batch)
            else:
                ep0_batch = eval_batches.pop(0)
                np.testing.assert_allclose(batch.inputs, ep0_batch.inputs)
                np.testing.assert_allclose(batch.targets, ep0_batch.targets)
