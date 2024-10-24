import os
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xlstm_jax.dataset import LLMBatch
from xlstm_jax.models import ModelConfig
from xlstm_jax.models.configs import ParallelConfig
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.block import mLSTMBlockConfig
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.layer import mLSTMCellConfig, mLSTMLayerConfig
from xlstm_jax.models.xlstm_parallel.xlstm_lm_model import xLSTMLMModel, xLSTMLMModelConfig
from xlstm_jax.trainer.callbacks.checkpointing import ModelCheckpointConfig
from xlstm_jax.trainer.llm.trainer import LLMTrainer, LLMTrainerConfig
from xlstm_jax.trainer.logger import LoggerConfig
from xlstm_jax.trainer.optimizer import OptimizerConfig, SchedulerConfig


@pytest.mark.parametrize("tp_size,fsdp_size", [(1, 1), (2, 2), (1, 8), (8, 1)])
def test_llm_trainer(llm_toy_model: Any, tmp_path: Path, tp_size: int, fsdp_size: int):
    """
    Tests training a simple model with LLM loss under different mesh configs.

    Also reproduces the checkpointing test from the checkpointing test file for this new trainer.
    """
    LLMToyModel = llm_toy_model
    if pytest.num_devices < tp_size * fsdp_size:
        pytest.skip("Test requires more devices than available.")
    batch_size = 8
    context_length = 16
    model_config = ModelConfig(
        model_class=LLMToyModel,
        parallel=ParallelConfig(
            data_axis_size=-1,
            model_axis_size=tp_size,
            fsdp_axis_size=fsdp_size,
            fsdp_min_weight_size=pytest.num_devices,
        ),
    )
    optimizer_config = OptimizerConfig(
        name="adam",
        scheduler=SchedulerConfig(
            name="constant",
            lr=1e-4,
        ),
    )
    trainer = LLMTrainer(
        LLMTrainerConfig(
            callbacks=(
                ModelCheckpointConfig(
                    monitor="perplexity",
                    max_to_keep=4,
                    save_optimizer_state=True,
                    enable_async_checkpointing=True,
                ),
            ),
            logger=LoggerConfig(log_path=tmp_path),
            check_val_every_n_epoch=1,
        ),
        model_config,
        optimizer_config,
        batch=LLMBatch.get_dtype_struct(batch_size=batch_size, max_length=context_length),
    )

    def data_gen_fn(idx: int) -> LLMBatch:
        inputs = jax.random.randint(jax.random.PRNGKey(idx), (batch_size, context_length), minval=0, maxval=50)
        targets = jnp.mod(inputs + 1, 50)
        return LLMBatch.from_inputs(inputs=inputs, targets=targets)

    train_loader = [data_gen_fn(idx) for idx in range(100)]
    val_loader = train_loader[:20]
    final_metrics = trainer.train_model(
        train_loader,
        val_loader,
        num_epochs=5,
    )
    assert final_metrics is not None
    assert all(f"val_epoch_{i}" in final_metrics for i in range(1, 6))
    assert "perplexity" in final_metrics["val_epoch_1"]
    assert (
        final_metrics["val_epoch_5"]["perplexity"] < final_metrics["val_epoch_4"]["perplexity"]
    ), "Validation perplexity should decrease over epochs."
    # Check that checkpoints have been created.
    assert os.path.exists(tmp_path)
    assert os.path.exists(tmp_path / "checkpoints")
    assert len(os.listdir(tmp_path / "checkpoints/")) == 4
    # Check that validation performance can be reproduced with checkpoint.
    trainer.load_model(step_idx=200)
    new_metrics = trainer.eval_model(val_loader, "val", epoch_idx=2)
    assert "perplexity" in new_metrics
    assert "loss" in new_metrics
    assert (
        new_metrics["perplexity"] == final_metrics["val_epoch_2"]["perplexity"]
    ), f"Perplexity should match the loaded model: {new_metrics} versus {final_metrics['val_epoch_2']}"
    assert new_metrics["loss"] == final_metrics["val_epoch_2"]["loss"], "Loss should match the loaded model."
    # Check that we can continue training from this checkpoint as before.
    new_final_metrics = trainer.train_model(
        train_loader,
        val_loader,
        num_epochs=5,
    )
    assert new_final_metrics is not None
    assert new_final_metrics["val_epoch_5"]["perplexity"] == final_metrics["val_epoch_5"]["perplexity"], (
        "Perplexity should match the loaded model, but got "
        f"{new_final_metrics['val_epoch_5']['perplexity']} versus {final_metrics['val_epoch_5']['perplexity']}."
    )
    # Check loading from pretrained model.
    new_trainer = LLMTrainer(
        LLMTrainerConfig(
            callbacks=(
                ModelCheckpointConfig(
                    monitor="perplexity",
                    max_to_keep=4,
                    save_optimizer_state=True,
                    enable_async_checkpointing=True,
                ),
            ),
            logger=LoggerConfig(log_path=tmp_path / "new_trainer_subdir"),
            check_val_every_n_epoch=1,
        ),
        model_config,
        optimizer_config,
        batch=LLMBatch.get_dtype_struct(batch_size=batch_size, max_length=context_length),
    )
    new_trainer.load_pretrained_model(tmp_path, step_idx=200, train_loader=train_loader, val_loader=val_loader)
    new_final_metrics = new_trainer.train_model(
        train_loader,
        val_loader,
        num_epochs=5,
    )
    assert new_final_metrics is not None
    assert new_final_metrics["val_epoch_5"]["perplexity"] == final_metrics["val_epoch_5"]["perplexity"], (
        "Perplexity should match the loaded pretrained model, but got "
        f"{new_final_metrics['val_epoch_5']['perplexity']} versus {final_metrics['val_epoch_5']['perplexity']}."
    )


def test_llm_padding(llm_toy_model: Any, tmp_path: Path):
    """Tests whether the padding works correctly in the LLM Trainer."""
    LLMToyModel = llm_toy_model
    batch_size = 8
    context_length = 16
    trainer = LLMTrainer(
        LLMTrainerConfig(
            callbacks=(
                ModelCheckpointConfig(
                    monitor="perplexity",
                    max_to_keep=4,
                    save_optimizer_state=True,
                    enable_async_checkpointing=True,
                ),
            ),
            logger=LoggerConfig(log_path=tmp_path),
            check_val_every_n_epoch=1,
        ),
        ModelConfig(
            model_class=LLMToyModel,
            parallel=ParallelConfig(
                data_axis_size=-1,
                model_axis_size=1,
                fsdp_axis_size=1,
                fsdp_min_weight_size=pytest.num_devices,
            ),
        ),
        OptimizerConfig(
            name="adam",
            scheduler=SchedulerConfig(
                name="constant",
                lr=1e-4,
            ),
        ),
        batch=LLMBatch.get_dtype_struct(batch_size=batch_size, max_length=context_length),
    )

    def data_gen_fn(idx: int) -> LLMBatch:
        inputs = jax.random.randint(jax.random.PRNGKey(idx), (batch_size, context_length), minval=0, maxval=50)
        targets = jnp.mod(inputs + 1, 50)
        return LLMBatch.from_inputs(inputs=inputs, targets=targets)

    def pad_batch(batch: LLMBatch) -> LLMBatch:
        return jax.tree.map(
            lambda x: jnp.pad(
                x, ((0, batch_size - x.shape[0]), (0, context_length - x.shape[1])), mode="constant", constant_values=0
            ),
            batch,
        )

    train_loader = [data_gen_fn(idx) for idx in range(100)]
    val_loader = train_loader[:20]
    val_metrics = trainer.eval_model(val_loader, "val", epoch_idx=0)
    batch_padded_val_loader = []
    for batch in val_loader:
        # Divide by 3 for non-uniform split.
        batch_padded_val_loader.append(pad_batch(batch[: batch_size // 3]))
        batch_padded_val_loader.append(pad_batch(batch[batch_size // 3 :]))
    batch_pad_val_metrics = trainer.eval_model(batch_padded_val_loader, "val", epoch_idx=0)
    # We do not check the perplexity as it is calculated per batch and thus can differ.
    np.testing.assert_allclose(
        val_metrics["loss"],
        batch_pad_val_metrics["loss"],
        rtol=1e-5,
        atol=1e-5,
        err_msg="Loss should match for batch-padded and non-padded batches.",
    )
    np.testing.assert_allclose(
        val_metrics["accuracy"],
        batch_pad_val_metrics["accuracy"],
        rtol=1e-5,
        atol=1e-5,
        err_msg="Loss should match for batch-padded and non-padded batches.",
    )


@pytest.mark.parametrize("tp_size,fsdp_size", [(1, 1), (1, 8), (2, 2), (4, 1)])
def test_xlstm_training(tmp_path: Path, tp_size: int, fsdp_size: int):
    """
    Tests training a xLSTM model with debug configuration.

    Also reproduces the checkpointing test from the checkpointing test file for this new trainer.
    """
    if pytest.num_devices < tp_size * fsdp_size:
        pytest.skip("Test requires more devices than available.")
    # General hyperparameters.
    batch_size = 8
    context_length = 16
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
    # Define model config as before.
    xlstm_config = xLSTMLMModelConfig(
        vocab_size=20,
        embedding_dim=128,
        num_blocks=2,
        context_length=context_length,
        tie_weights=False,
        add_embedding_dropout=True,
        add_post_blocks_norm=True,
        parallel=parallel,
        dtype=jnp.float32,
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                proj_factor=2.0,
                conv1d_kernel_size=4,
                num_heads=4,
                dropout=0.2,
                embedding_dim=128,
                context_length=context_length,
                mlstm_cell=mLSTMCellConfig(
                    gate_linear_headwise=True,
                    gate_soft_cap=30.0,
                    reset_at_document_boundaries=True,
                ),
            )
        ),
    )

    trainer = LLMTrainer(
        LLMTrainerConfig(
            callbacks=(
                ModelCheckpointConfig(
                    monitor="perplexity",
                    max_to_keep=4,
                    save_optimizer_state=True,
                    enable_async_checkpointing=True,
                ),
            ),
            logger=LoggerConfig(log_path=tmp_path),
        ),
        ModelConfig(
            model_class=xLSTMLMModel,
            parallel=parallel,
            model_config=xlstm_config,
        ),
        OptimizerConfig(
            name="adam",
            scheduler=SchedulerConfig(
                name="constant",
                lr=1e-4,
            ),
        ),
        batch=LLMBatch.get_dtype_struct(batch_size, xlstm_config.context_length),
    )

    def data_gen_fn(idx: int) -> LLMBatch:
        inputs = jax.random.randint(
            jax.random.PRNGKey(idx),
            (batch_size, xlstm_config.context_length),
            minval=0,
            maxval=xlstm_config.vocab_size,
        )
        targets = jnp.mod(inputs + 1, xlstm_config.vocab_size)
        return LLMBatch.from_inputs(inputs=inputs, targets=targets)

    train_loader = [data_gen_fn(idx) for idx in range(50)]
    val_loader = train_loader[:10]
    final_metrics = trainer.train_model(
        train_loader,
        val_loader,
        num_epochs=5,
    )
    # Check that metrics are present as expected.
    assert final_metrics is not None
    assert all(f"val_epoch_{i}" in final_metrics for i in range(1, 6))
    assert "perplexity" in final_metrics["val_epoch_1"]
    assert (
        final_metrics["val_epoch_5"]["perplexity"] < final_metrics["val_epoch_4"]["perplexity"]
    ), "Validation perplexity should decrease over epochs."
    # Check that checkpoints have been created.
    assert os.path.exists(tmp_path)
    assert os.path.exists(tmp_path / "checkpoints")
    assert len(os.listdir(tmp_path / "checkpoints/")) == 4
    # Check that validation performance can be reproduced with checkpoint.
    trainer.load_model(step_idx=150)
    new_metrics = trainer.eval_model(val_loader, "val", epoch_idx=3)
    assert "perplexity" in new_metrics
    assert "loss" in new_metrics
    assert (
        new_metrics["perplexity"] == final_metrics["val_epoch_3"]["perplexity"]
    ), f"Perplexity should match the loaded model: {new_metrics} versus {final_metrics['val_epoch_3']}"
    assert new_metrics["loss"] == final_metrics["val_epoch_3"]["loss"], "Loss should match the loaded model."
    # Check that we can continue training from this checkpoint as before.
    new_final_metrics = trainer.train_model(
        train_loader,
        val_loader,
        num_epochs=5,
    )
    assert new_final_metrics is not None
    assert new_final_metrics["val_epoch_5"]["perplexity"] == final_metrics["val_epoch_5"]["perplexity"], (
        "Perplexity should match the loaded pretrained model, but got "
        f"{new_final_metrics['val_epoch_5']['perplexity']} versus {final_metrics['val_epoch_5']['perplexity']}."
    )
