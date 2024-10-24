from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import pandas as pd
import pytest

from xlstm_jax.dataset import Batch
from xlstm_jax.models import ModelConfig
from xlstm_jax.models.configs import ParallelConfig
from xlstm_jax.trainer import TrainerConfig
from xlstm_jax.trainer.logger import FileLoggerConfig, LoggerConfig
from xlstm_jax.trainer.optimizer import OptimizerConfig, SchedulerConfig


@pytest.mark.parametrize("tp_size,fsdp_size", [(1, 1), (2, 2), (1, 8), (8, 1)])
def test_mse_trainer(mse_trainer: Any, toy_model: Any, tp_size: int, fsdp_size: int):
    """Tests training a simple model with MSE loss under different mesh configs."""
    MSETrainer = mse_trainer
    ToyModel = toy_model
    if pytest.num_devices < tp_size * fsdp_size:
        pytest.skip("Test requires more devices than available.")
    trainer = MSETrainer(
        TrainerConfig(
            check_val_every_n_epoch=1,
            check_val_every_n_steps=100,
        ),
        ModelConfig(
            model_class=ToyModel,
            parallel=ParallelConfig(
                data_axis_size=-1,
                model_axis_size=tp_size,
                fsdp_axis_size=fsdp_size,
                fsdp_min_weight_size=pytest.num_devices,
            ),
        ),
        OptimizerConfig(
            name="adam",
            scheduler=SchedulerConfig(
                name="constant",
                lr=1e-3,
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

    train_loader = [data_gen_fn(idx) for idx in range(250)]
    val_loader = train_loader[:20]
    final_metrics = trainer.train_model(
        train_loader,
        val_loader,
        num_epochs=2,
    )
    assert final_metrics is not None
    epoch_keys = [f"val_epoch_{i}" for i in range(1, 3)]
    step_keys = [f"val_step_{i}" for i in range(100, 501, 100)]
    assert all(
        k in final_metrics for k in epoch_keys
    ), f"Validation metrics should be logged at the end of each epoch, instead got keys: {final_metrics.keys()}."
    assert all(
        k in final_metrics for k in step_keys
    ), f"Validation metrics should be logged at the end of each step, instead got keys: {final_metrics.keys()}."
    assert set(final_metrics[epoch_keys[0]].keys()) == {
        "loss",
        "l1_dist",
        "epoch_time",
    }, f"Keys should be the same as specified in the loss function, but got {final_metrics[epoch_keys[0]].keys()}."
    assert set(final_metrics[epoch_keys[0]].keys()) == set(
        final_metrics[epoch_keys[1]].keys()
    ), f"Keys should be the same for all validation metrics, but got {final_metrics[epoch_keys[1]].keys()}."
    assert set(final_metrics[epoch_keys[0]].keys()) == set(
        final_metrics[step_keys[1]].keys()
    ), f"Keys should be the same for all validation metrics, but got {final_metrics[step_keys[1]].keys()}."
    assert (
        final_metrics[epoch_keys[0]]["loss"] < 0.1
    ), f"Validation loss should be less than 0.1, but got {final_metrics[epoch_keys[0]]['loss']}."
    assert (
        final_metrics[epoch_keys[1]]["loss"] < 0.1
    ), f"Validation loss should be less than 0.1, but got {final_metrics[epoch_keys[1]]['loss']}."
    assert (
        final_metrics[epoch_keys[1]]["loss"] < final_metrics[epoch_keys[0]]["loss"]
    ), "Validation loss should decrease over epochs."
    new_metrics = trainer.eval_model(val_loader, "eval", epoch_idx=2)
    assert new_metrics is not None
    assert new_metrics["loss"] == final_metrics[epoch_keys[1]]["loss"], "Loss should be the same."
    assert new_metrics["l1_dist"] == final_metrics[epoch_keys[1]]["l1_dist"], "L1 distance should be the same."


def test_nan_checks(mse_trainer: Any, toy_model: Any, tmp_path: Path):
    """Tests training a simple model with MSE loss under different mesh configs."""
    MSETrainer = mse_trainer
    ToyModel = toy_model
    log_path = tmp_path / "logs"
    fl_dir = "file_logs"
    trainer = MSETrainer(
        TrainerConfig(
            check_val_every_n_epoch=1,
            check_val_every_n_steps=100,
            check_for_nan=True,
            logger=LoggerConfig(
                log_every_n_steps=20,
                log_path=log_path,
                log_tools=[FileLoggerConfig(log_dir=fl_dir)],
            ),
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
                name="constant",
                lr=1e-3,
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
        if idx == 210:
            labels = jnp.nan * labels
        return Batch(inputs=inputs, targets=labels)

    train_loader = [data_gen_fn(idx) for idx in range(250)]
    val_loader = train_loader[:20]
    final_metrics = trainer.train_model(
        train_loader,
        val_loader,
        num_epochs=2,
    )
    assert final_metrics is not None
    assert not any(
        key.startswith("val_epoch") for key in final_metrics.keys()
    ), "No validation metrics should be logged at an epoch, NaN should have been detected beforehand."
    assert set(final_metrics.keys()) == {
        "val_step_100",
        "val_step_200",
    }, f"Only two validations should have been logged, but found keys: {final_metrics.keys()}."

    assert log_path.exists()
    assert (log_path / "output.log").exists()
    assert (log_path / fl_dir).exists(), f"Expected file logging directory {log_path / fl_dir} to exist"
    assert (
        log_path / fl_dir / "metrics_train.csv"
    ).exists(), f"Expected metrics file {log_path / fl_dir / f'metrics_train.csv'} to exist"
    df = pd.read_csv(log_path / fl_dir / "metrics_train.csv")
    # Logs till 220 (11 steps), plus one epoch time.
    assert df.shape[0] == 12, f"Expected 12 columns in the metrics file, but got {df.shape[0]}."
    assert "loss_mean" in df.columns, "Expected 'loss_mean' column in the metrics file."
    # Check that last loss is NaN.
    assert jnp.isnan(df["loss_mean"].values[-2]), "Expected last loss to be NaN."


@pytest.mark.parametrize("gradient_accumulate_scan", [False, True])
@pytest.mark.parametrize("gradient_accumulate_steps", [1, 2])
def test_log_intermediates(
    mse_trainer: Any, toy_model: Any, tmp_path: Path, gradient_accumulate_scan: bool, gradient_accumulate_steps: int
):
    """Tests logging intermediates during the training."""
    MSETrainer = mse_trainer
    ToyModel = toy_model
    log_path = tmp_path / "logs"
    fl_dir = "file_logs"
    trainer = MSETrainer(
        TrainerConfig(
            check_val_every_n_epoch=1,
            check_val_every_n_steps=100,
            check_for_nan=True,
            logger=LoggerConfig(
                log_every_n_steps=10,
                log_path=log_path,
                log_tools=[FileLoggerConfig(log_dir=fl_dir)],
            ),
            gradient_accumulate_scan=gradient_accumulate_scan,
            gradient_accumulate_steps=gradient_accumulate_steps,
            log_intermediates=True,
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
                name="constant",
                lr=1e-3,
            ),
        ),
        batch=Batch(
            inputs=jax.ShapeDtypeStruct((16, 64), jnp.float32),
            targets=jax.ShapeDtypeStruct((16, 1), jnp.float32),
        ),
    )

    def data_gen_fn(idx: int) -> Batch:
        inputs = jax.random.normal(jax.random.PRNGKey(idx), (16, 64))
        labels = inputs[:, 0:1]
        return Batch(inputs=inputs, targets=labels)

    train_loader = [data_gen_fn(idx) for idx in range(50)]
    val_loader = train_loader[:10]
    final_metrics = trainer.train_model(
        train_loader,
        val_loader,
        num_epochs=2,
    )
    assert final_metrics is not None

    assert log_path.exists()
    assert (log_path / "output.log").exists()
    assert (log_path / fl_dir).exists(), f"Expected file logging directory {log_path / fl_dir} to exist"
    assert (
        log_path / fl_dir / "metrics_train.csv"
    ).exists(), f"Expected metrics file {log_path / fl_dir / f'metrics_train.csv'} to exist"
    df = pd.read_csv(log_path / fl_dir / "metrics_train.csv")
    assert (
        "last_activations.0_0_mean" in df.columns
    ), f"Expected 'last_activations.0_0_mean' column in the metrics file, but got {df.columns}."
