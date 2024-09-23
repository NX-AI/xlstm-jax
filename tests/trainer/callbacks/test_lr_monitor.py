from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from xlstm_jax.dataset import Batch
from xlstm_jax.models import ModelConfig
from xlstm_jax.models.configs import ParallelConfig
from xlstm_jax.trainer import TrainerConfig
from xlstm_jax.trainer.callbacks import LearningRateMonitorConfig
from xlstm_jax.trainer.logger import FileLoggerConfig, LoggerConfig
from xlstm_jax.trainer.optimizer import OptimizerConfig, SchedulerConfig, build_lr_scheduler

from ..helpers.mse_trainer import MSETrainer, ToyModel

SCHEDULERS = [
    SchedulerConfig(name="constant", lr=0.1),
    SchedulerConfig(name="exponential_decay", lr=0.1, end_lr=0.01, decay_steps=400, warmup_steps=20, cooldown_steps=20),
    SchedulerConfig(name="cosine_decay", lr=0.1, end_lr=0.01, decay_steps=400, warmup_steps=20, cooldown_steps=0),
    SchedulerConfig(name="linear", lr=0.1, end_lr=0.01, decay_steps=400, warmup_steps=0, cooldown_steps=20),
]


@pytest.mark.parametrize("scheduler_config", SCHEDULERS)
@pytest.mark.parametrize("step_freq", [20, 50])
def test_lr_monitor(tmp_path: Path, scheduler_config: SchedulerConfig, step_freq: int):
    """Tests logging the learning rate with the callback."""
    log_path = tmp_path / "test_lr_monitor" / f"scheduler_{scheduler_config.name}"
    fl_dir = "file_logs"
    trainer = MSETrainer(
        TrainerConfig(
            callbacks=(
                LearningRateMonitorConfig(
                    every_n_steps=step_freq,
                    every_n_epochs=-1,
                    main_process_only=True,
                ),
            ),
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
            scheduler=scheduler_config,
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

    train_loader = [data_gen_fn(idx) for idx in range(100)]
    val_loader = train_loader[:20]
    _ = trainer.train_model(
        train_loader,
        val_loader,
        num_epochs=4,
    )
    assert log_path.exists()
    assert (log_path / "output.log").exists()
    assert (log_path / fl_dir).exists(), f"Expected file logging directory {log_path / fl_dir} to exist"
    assert (
        log_path / fl_dir / "metrics_train.csv"
    ).exists(), f"Expected metrics file {log_path / fl_dir / 'metrics_train.csv'} to exist"
    df = pd.read_csv(log_path / fl_dir / "metrics_train.csv")
    assert "optimizer/lr" in df.columns, f"Expected 'lr' column in DataFrame {df.columns}"
    tracked_lrs = {step: lr for step, lr in zip(df["log_step"], df["optimizer/lr"]) if not jnp.isnan(lr)}
    assert tracked_lrs, "Expected at least one learning rate to be logged"
    assert all(
        key in tracked_lrs for key in range(step_freq, 400, step_freq)
    ), "Expected learning rate to be logged at every step_freq"
    lr_schedule = build_lr_scheduler(scheduler_config)
    for key in tracked_lrs:
        np.testing.assert_allclose(
            tracked_lrs[key],
            lr_schedule(key),
            atol=1e-6,
            err_msg=f"Expected learning rate to match scheduler at step {key}",
        )
