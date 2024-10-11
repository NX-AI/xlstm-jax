import logging
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from xlstm_jax.dataset import Batch
from xlstm_jax.models import ModelConfig
from xlstm_jax.models.configs import ParallelConfig
from xlstm_jax.trainer import TrainerConfig
from xlstm_jax.trainer.logger import FileLoggerConfig, LoggerConfig
from xlstm_jax.trainer.optimizer import OptimizerConfig, SchedulerConfig

from ..helpers.mse_trainer import MSETrainer, ToyModel

LOGGER = logging.getLogger(__name__)


@pytest.mark.parametrize("fl_dir", ["file_logs", "loggings_files"])
@pytest.mark.parametrize("config_format", ["json", "yaml", "pickle"])
def test_file_logging_mse_trainer(tmp_path: Path, fl_dir: str, config_format: Literal["json", "yaml", "pickle"]):
    """Tests logging for example trainer."""
    log_path = tmp_path / "logs"
    trainer = MSETrainer(
        TrainerConfig(
            check_val_every_n_epoch=1,
            check_val_every_n_steps=50,
            logger=LoggerConfig(
                log_every_n_steps=10,
                log_path=log_path,
                log_tools=[FileLoggerConfig(log_dir=fl_dir, config_format=config_format)],
            ),
            default_train_log_modes=("mean", "single"),
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
        targets = inputs[:, 0:1]
        return Batch(inputs=inputs, targets=targets)

    train_loader = [data_gen_fn(idx) for idx in range(100)]
    val_loader = train_loader[:20]
    test_loader = train_loader[20:40]
    final_metrics = trainer.train_model(
        train_loader,
        val_loader,
        test_loader=test_loader,
        num_epochs=2,
    )

    # Check that the log directory exists and has the right files.
    assert log_path.exists()
    assert (log_path / "output.log").exists()
    assert (log_path / fl_dir).exists(), f"Expected file logging directory {log_path / fl_dir} to exist"
    config_postfix = "json" if config_format == "json" else "yaml" if config_format == "yaml" else "pkl"
    assert (
        log_path / fl_dir / f"config.{config_postfix}"
    ).exists(), f"Expected config file {log_path / fl_dir / f'config.{config_postfix}'} to exist"

    # Check that files for the pre-training logs exist.
    for pre_modes in ["dataset", "model", "parallel"]:
        assert (
            log_path / fl_dir / f"metrics_{pre_modes}.csv"
        ).exists(), f"Expected metrics file {log_path / fl_dir / f'metrics_{pre_modes}.csv'} to exist."

    # Check that files for the training logs exist and are in the right format.
    for mode in ["train", "val", "test"]:
        assert (
            log_path / fl_dir / f"metrics_{mode}.csv"
        ).exists(), f"Expected metrics file {log_path / fl_dir / f'metrics_{mode}.csv'} to exist"
        df = pd.read_csv(log_path / fl_dir / f"metrics_{mode}.csv")
        if mode == "train":
            # Check that the training metrics pandas dataframe has the right shape and columns.
            assert (
                df.shape[0] == 22
            ), f"Expected 22 rows in train metrics file, got {df.shape[0]}"  # 20 logs, 2 epoch times
            exp_columns = [
                "l1_dist_mean",
                "l1_dist_single",
                "loss_mean",
                "loss_single",
                "grad_norm_mean",
                "grad_norm_single",
                "param_norm",
                "epoch",
                "log_step",
                "log_epoch",
                "epoch_time",
                "step_time",
            ]
            assert df.shape[1] == len(
                exp_columns
            ), f"Expected {len(exp_columns)} columns in train metrics file, got {df.shape[1]}: {df.columns.tolist()}"
            assert set(df.columns.tolist()) == set(
                exp_columns
            ), f"Expected columns in train metrics file to be {exp_columns}, got {df.columns.tolist()}"
            # Logs are every 10 steps.
            assert set(df["log_step"].tolist()) == set(
                range(10, 201, 10)
            ), f"Expected log steps to be [20, 40, ..., 200], got {df['log_step'].tolist()}"
        else:
            # Check that the validation/test metrics pandas dataframe has the right shape and columns.
            if mode == "val":
                exp_rows = 4
                exp_steps = list(range(50, 201, 50))
            else:
                exp_rows = 1
                exp_steps = [200]
            exp_rows = 4 if mode == "val" else 1
            assert df.shape[0] == exp_rows, f"Expected {exp_rows} rows in {mode} metrics file, got {df.shape[0]}"
            exp_columns = ["l1_dist", "loss", "epoch_time", "log_step", "log_epoch"]
            assert df.shape[1] == len(
                exp_columns
            ), f"Expected {exp_columns} columns in val metrics file, got {df.shape[1]}"
            assert set(df.columns.tolist()) == set(
                exp_columns
            ), f"Expected columns in val metrics file to be {exp_columns}, got {df.columns.tolist()}"
            assert (
                df["log_step"].tolist() == exp_steps
            ), f"Expected log steps to be {exp_steps}, got {df['log_step'].tolist()}"
            # Check that metrics match the final metric values returned by trainer.
            for idx, step in enumerate(exp_steps):
                metrics = final_metrics[f"{mode}_step_{step}" if mode == "val" else mode]
                df_elem = df.iloc[idx]
                for key in metrics:
                    np.testing.assert_allclose(
                        df_elem[key],
                        metrics[key],
                        err_msg=f"Expected {key} to be {metrics[key]} at step {step}, got {df_elem[key]}",
                    )
