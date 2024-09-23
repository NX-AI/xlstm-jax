import logging
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from xlstm_jax.dataset import Batch
from xlstm_jax.models import ModelConfig
from xlstm_jax.models.configs import ParallelConfig
from xlstm_jax.trainer import TrainerConfig
from xlstm_jax.trainer.logger import Logger, LoggerConfig
from xlstm_jax.trainer.logger.base_logger import LoggerTool, LoggerToolsConfig
from xlstm_jax.trainer.optimizer import OptimizerConfig, SchedulerConfig

from ..helpers.mse_trainer import MSETrainer, ToyModel

LOGGER = logging.getLogger(__name__)


class ToyLoggerTool(LoggerTool):
    """Toy logger tool for testing."""

    def log_config(self, config: dict):
        """Log the config."""
        del config
        LOGGER.info("ToyLoggerTool: Logging config")

    def setup(self):
        """Setup the logger tool."""
        LOGGER.info("ToyLoggerTool: Setup")

    def log_metrics(self, metrics: dict, step: int, epoch: int, mode: str):
        """Log the metrics and check that they are correct."""
        LOGGER.info(f"ToyLoggerTool: Logging metrics {metrics} at epoch {epoch}, step {step}, mode {mode}")
        assert isinstance(metrics, dict)
        assert all(
            [isinstance(v, (float, int, str)) for v in metrics.values()]
        ), f"Values are not all float, int, or str, but found types: {[(k, type(v)) for k, v in metrics.items()]}"
        if mode == "train":
            if "epoch_time" in metrics:  # Train epoch logging.
                assert set(metrics.keys()) == {"epoch_time"}, f"Keys {metrics.keys()} do not match expected keys."
            else:  # Train step logging.
                assert "epoch" in metrics, "Epoch not in metrics."
                for key in metrics:
                    if key.endswith("_single"):
                        assert (
                            metrics[key] != metrics[key[: -len("_single")] + "_mean"]
                        ), f"Key {key} is not different from mean."
        elif mode in ["val", "test"]:  # Validation / test epoch logging.
            assert set(metrics.keys()) == {
                "l1_dist",
                "loss",
                "epoch_time",
            }, f"Keys {metrics.keys()} do not match expected keys."


@dataclass(kw_only=True, frozen=True)
class ToyLoggerToolConfig(LoggerToolsConfig):
    """Toy logger tool config for testing."""

    def create(self, logger: Logger) -> LoggerTool:
        """Create the toy logger tool."""
        del logger
        return ToyLoggerTool()


def test_base_logging_mse_trainer(tmp_path: Path):
    """Tests logging for example trainer."""
    log_path = tmp_path / "logs"
    trainer = MSETrainer(
        TrainerConfig(
            check_val_every_n_epoch=1,
            check_val_every_n_steps=100,
            logger=LoggerConfig(
                log_every_n_steps=20,
                log_path=log_path,
                log_tools=[ToyLoggerToolConfig()],
            ),
        ),
        ModelConfig(
            model_class=ToyModel,
            parallel=ParallelConfig(
                data_axis_size=-1,
                model_axis_size=2,
                fsdp_axis_size=2,
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
        """Data generator function."""
        inputs = jax.random.normal(jax.random.PRNGKey(idx), (8, 64))
        targets = inputs[:, 0:1]
        return Batch(inputs=inputs, targets=targets)

    train_loader = [data_gen_fn(idx) for idx in range(250)]
    val_loader = train_loader[:20]
    test_loader = train_loader[20:40]
    _ = trainer.train_model(
        train_loader,
        val_loader,
        test_loader=test_loader,
        num_epochs=2,
    )
    assert log_path.exists()
    assert (log_path / "output.log").exists()
    with open(log_path / "output.log") as f:
        lines = f.readlines()
    assert len(lines) > 0
    full_log = "".join(lines)
    # Check that setup is correctly called.
    assert "ToyLoggerTool: Setup" in full_log, "Setup not found in log."
    # Check that config is correctly logged.
    assert "ToyLoggerTool: Logging config" in full_log, "Config not found in log."
    # Check that logging is happening at every 20 steps.
    for step in range(20, 500, 20):
        epoch = step // 250 + 1
        assert (
            f"Logging at step {step} in epoch {epoch} in mode train" in full_log
        ), f"Train step {step} not found in log."
    # Check that logging is happening at every 100 steps for validation.
    for step in range(100, 500, 100):
        epoch = step // 250 + 1
        assert f"Starting epoch {epoch} at step {step} in mode val" in full_log, f"Val epoch {epoch} not found in log."
    # Check that logging is happening at every epoch.
    for step in range(0, 500, 250):
        epoch = step // 250 + 1
        assert (
            f"Starting epoch {epoch} at step {step} in mode train" in full_log
        ), f"Train epoch {epoch} not found in log."
    # Check that logging is happening for test mode.
    assert "Starting epoch 2 at step 500 in mode test." in full_log
    assert (
        "FrozenDict" not in full_log
    ), "FrozenDict should not be in log. Indicates we are printing device metrics somewhere."
