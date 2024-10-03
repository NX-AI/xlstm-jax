from pathlib import Path

import numpy as np
import pytest
from jax.sharding import PartitionSpec as P

from xlstm_jax.dataset import HFHubDataConfig, create_data_iterator
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
    data_config = HFHubDataConfig(
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
    last_eval_batch = None
    eval_dataset_size = len(eval_iterator)
    for epoch_idx in range(2):
        step = 0
        for batch_idx, batch in enumerate(eval_iterator):
            step += 1
            assert batch.inputs.shape == (global_batch_size, context_length)
            assert batch.inputs.sharding.spec == P(mesh.axis_names)
            if epoch_idx == 0:
                eval_batches.append(batch)
            else:
                ep0_batch = eval_batches[batch_idx]
                np.testing.assert_allclose(batch.inputs, ep0_batch.inputs)
                np.testing.assert_allclose(batch.targets, ep0_batch.targets)
        assert step == eval_dataset_size, "Should have the same number of steps for each epoch."
    assert len(eval_batches) == eval_dataset_size, "Should have the same number of batches for each epoch."
    last_eval_batch = eval_batches[-1]
    # For all attributes in the batch, we should have zeros.
    for attr_name in [
        "inputs",
        "targets",
        "inputs_segmentation",
        "targets_segmentation",
        "inputs_position",
        "targets_position",
    ]:
        val = getattr(last_eval_batch, attr_name)[-1]
        assert np.all(val == 0), f"Last batch should have all zeros for {attr_name}, but got {val}."
    assert not np.all(
        last_eval_batch.inputs[0] == 0
    ), "In the last batch, there should be at least one non-padded element."

    # Test max steps per epoch and a continuous validation iterator.
    eval_max_steps_per_epoch = len(eval_iterator) // 2
    data_config = HFHubDataConfig(
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
        eval_max_steps_per_epoch=eval_max_steps_per_epoch,
    )

    # Define iterators
    _, eval_iterator = create_data_iterator(config=data_config, mesh=mesh)
    assert len(eval_iterator) == eval_max_steps_per_epoch, "Eval dataset have the specified number of steps."
    for epoch_idx in range(3):
        batch_idx = 0
        for batch in eval_iterator:
            np.testing.assert_allclose(
                batch.inputs,
                eval_batches[batch_idx].inputs,
                err_msg=f"Failed at batch idx: {batch_idx}, epoch idx: {epoch_idx}",
            )
            np.testing.assert_allclose(
                batch.targets,
                eval_batches[batch_idx].targets,
                err_msg=f"Failed at batch idx: {batch_idx}, epoch idx: {epoch_idx}",
            )
            batch_idx += 1
        assert (
            batch_idx == eval_max_steps_per_epoch
        ), f"Should have the same number of steps for each epoch, failed at {epoch_idx}."
