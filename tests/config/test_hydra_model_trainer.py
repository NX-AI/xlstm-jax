from pathlib import Path

import jax
import numpy as np
import pytest
from hydra import compose, initialize

from xlstm_jax.define_hydra_schemas import register_configs
from xlstm_jax.main_train import main_train


@pytest.mark.parametrize("model_name", ["llamaDebug", "mLSTMDebug"])
def test_hydra_trainer(tmp_path: Path, model_name: str):
    """Sets up the config via Hydra and calls regular main_train function."""
    fsdp_axis_size = 1
    model_axis_size = 1

    register_configs()

    context_length = 32
    batch_size_per_device = 1
    global_batch_size = batch_size_per_device * pytest.num_devices
    with initialize(version_base="1.3", config_path="../../configs"):
        cfg = compose(
            config_name="config",
            overrides=[
                f"batch_size_per_device={batch_size_per_device}",
                f"context_length={context_length}",
                f"log_path={tmp_path}",
                "num_epochs=1",
                "parallel.data_axis_name=dp",
                "parallel.fsdp_axis_name=fsdp",
                "parallel.model_axis_name=tp",
                "parallel.pipeline_axis_name=pp",
                "parallel.fsdp_modules=[Embed, LMHead, mLSTMBlock]",
                f"parallel.fsdp_min_weight_size={pytest.num_devices}",
                f"parallel.fsdp_axis_size={fsdp_axis_size}",
                f"parallel.model_axis_size={model_axis_size}",
                "parallel.data_axis_size=-1",
                f"data.global_batch_size={global_batch_size}",
                f"data.max_target_length={context_length}",
                "data.data_shuffle_seed=42",
                "data.num_batches=50",
                f"model.name={model_name}",
                "model.vocab_size=20",
                "model.embedding_dim=128",
                "model.num_blocks=1",
                f"model.context_length={context_length}",
                "model.tie_weights=False",
                "model.add_embedding_dropout=True",
                "model.add_post_blocks_norm=True",
                "model.dtype=float32",
                "model.num_heads=4",
                "model.head_dim=8",
                "model.gate_dtype=float32",
                "model.backend=parallel_stabilized",
                "checkpointing.monitor=perplexity",
                "checkpointing.max_to_keep=1",
                "checkpointing.save_optimizer_state=True",
                "checkpointing.enable_async_checkpointing=True",
                "scheduler.lr=1e-3",
                "scheduler.end_lr_factor=0.1",
                "scheduler.warmup_steps=20",
                "scheduler.cooldown_steps=10",
                "optimizer.beta2=0.95",
                "logger.loggers_to_use=[file_logger, tb_logger]",
                "task_name=unit_test",
                "hydra.job.num=0",
            ],
            return_hydra_config=True,
        )

        # Call main train function. We need to set the output_dir, log_path and cmd_logging_name
        # manually because Hydra does not fill the hydra configuration when composing the config
        # compared to using the @hydra.main decorator. Is only relevant for the unit test.
        cfg.hydra.runtime.output_dir = tmp_path
        cfg.logger.log_path = tmp_path
        cfg.logger.cmd_logging_name = "unit_test"
        final_metrics = main_train(cfg=cfg)

    assert final_metrics is not None, "Output metrics were None."
    assert (
        "val_epoch_1" in final_metrics
    ), f"Missing validation epoch 1 key in final metrics, got instead {final_metrics}."
    assert not any(np.isnan(jax.tree.leaves(final_metrics))), f"Found NaNs in final metrics: {final_metrics}."
    val_acc = final_metrics["val_epoch_1"]["accuracy"]
    assert val_acc == 1.0, f"Expected accuracy to be 100%, but got {val_acc:.2%}."
