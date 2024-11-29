from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from xlstm_jax.common_types import Metrics, PyTree
from xlstm_jax.dataset import Batch
from xlstm_jax.distributed import split_array_over_mesh
from xlstm_jax.models import ModelConfig
from xlstm_jax.models.configs import ParallelConfig
from xlstm_jax.trainer import TrainerConfig, TrainerModule
from xlstm_jax.trainer.callbacks import Callback
from xlstm_jax.trainer.callbacks.extended_evaluation import ExtendedEvaluation, ExtendedEvaluationConfig
from xlstm_jax.trainer.logger import FileLoggerConfig, LoggerConfig
from xlstm_jax.trainer.optimizer import OptimizerConfig, SchedulerConfig


class ToyExtendedEvaluation(ExtendedEvaluation):
    def eval_function(self, params: Any, apply_fn: Any, batch: Batch, rng: jax.Array) -> tuple[Metrics, PyTree]:
        """Add additional L3 metric as example."""
        # Apply the model to the inputs. This is where the forward pass happens.
        preds, mutable_vars = apply_fn(
            {"params": params},
            batch.inputs,
            train=False,
            mutable="intermediates",
        )
        # Select the labels per device. This is only needed if we want to split the output over the mesh.
        # Otherwise, the TP and PP devices may share the same outputs.
        labels = batch.targets
        labels = split_array_over_mesh(labels, axis_name=self.trainer.pipeline_axis_name, split_axis=0)
        labels = split_array_over_mesh(labels, axis_name=self.trainer.model_axis_name, split_axis=0)
        assert (
            preds.shape == labels.shape
        ), f"Predictions and labels shapes do not match: {preds.shape} vs {labels.shape}"
        # Compute the loss and metrics.
        batch_size = np.prod(labels.shape)
        l1_dist = jnp.abs((preds - labels) ** 3)
        # Collect metrics and return loss.
        step_metrics = {
            "l3_dist": {"value": l1_dist.sum(), "count": batch_size},
        }
        return step_metrics, mutable_vars

    def on_filtered_validation_epoch_end(self, eval_metrics: Metrics, epoch_idx: int, step_idx: int):
        """Rerun the extended metrics on the validation dataset and log to standard logger."""
        val_iter = iter(self.data_module.val_dataloader)
        metrics = self.eval_model(val_iter, mode="val", epoch_idx=epoch_idx)
        self.trainer.logger.log_host_metrics(metrics, step=step_idx, mode="val")


class ToyExtendedEvaluationConfig(ExtendedEvaluationConfig):
    """Configuration for the ToyCallback."""

    def create(self, trainer: TrainerModule, data_module: Any | None = None) -> Callback:
        return ToyExtendedEvaluation(self, trainer, data_module)


def test_extended_evaluation(mse_trainer: Any, toy_model: Any, tmp_path: Path):
    """
    Tests callback if filtered methods are running at expected epochs and steps.
    """
    MSETrainer = mse_trainer
    ToyModel = toy_model

    log_path = tmp_path / "test_extended_evaluation"
    fl_dir = "file_logs"
    trainer = MSETrainer(
        TrainerConfig(
            callbacks=(ToyExtendedEvaluationConfig(),),
            logger=LoggerConfig(
                log_path=log_path,
                log_tools=[
                    FileLoggerConfig(log_dir=fl_dir),
                ],
                log_every_n_steps=10,
            ),
            check_val_every_n_epoch=1,
        ),
        ModelConfig(
            model_class=ToyModel,
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
                name="exponential_decay",
                lr=1e-3,
                decay_steps=500,
                warmup_steps=50,
                cooldown_steps=50,
                end_lr_factor=0.1,
            ),
        ),
        batch=Batch(
            inputs=jax.ShapeDtypeStruct((8, 64), jnp.float32),
            targets=jax.ShapeDtypeStruct((8, 1), jnp.float32),
        ),
    )

    def data_gen_fn(idx: int) -> Batch:
        inputs = jax.random.normal(jax.random.PRNGKey(idx), (8, 64))
        labels = inputs[:, 0:1]
        return Batch(inputs=inputs, targets=labels)

    train_loader = [data_gen_fn(idx) for idx in range(13)]
    val_loader = train_loader[:7]
    _ = trainer.train_model(
        train_loader,
        val_loader,
        num_train_steps=200,
    )

    assert log_path.exists()
    assert (log_path / "output.log").exists()
    assert (log_path / fl_dir).exists(), f"Expected file logging directory {log_path / fl_dir} to exist"
    assert (
        log_path / fl_dir / "metrics_val.csv"
    ).exists(), f"Expected metrics file {log_path / fl_dir / 'metrics_val.csv'} to exist"
    df = pd.read_csv(log_path / fl_dir / "metrics_val.csv")

    assert "l3_dist" in df.columns, f"Expected 'l3_dist' column in DataFrame {df.columns}"
