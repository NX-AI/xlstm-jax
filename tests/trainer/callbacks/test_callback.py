from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import pytest

from xlstm_jax.common_types import Metrics
from xlstm_jax.dataset import Batch
from xlstm_jax.models import ModelConfig
from xlstm_jax.models.configs import ParallelConfig
from xlstm_jax.trainer import TrainerConfig, TrainerModule
from xlstm_jax.trainer.callbacks import Callback, CallbackConfig
from xlstm_jax.trainer.logger import LoggerConfig
from xlstm_jax.trainer.optimizer import OptimizerConfig, SchedulerConfig


class ToyCallback(Callback):
    """Toy callback for testing purposes."""

    def on_filtered_training_epoch_start(self, epoch_idx: int):
        """Test function for training epoch start."""
        assert self._every_n_epochs > 0, "Train epoch start should not be called if every_n_epochs is not active."
        assert (
            epoch_idx % self._every_n_epochs == 0
        ), f"Train epoch start called at epoch idx {epoch_idx} but every {self._every_n_epochs} epochs."
        assert (not self._main_process_only) or (
            jax.process_index() == 0
        ), "Train epoch start should only be called in the main process."

    def on_filtered_training_epoch_end(self, train_metrics: Metrics, epoch_idx: int):
        """Test function for training epoch end."""
        del train_metrics
        assert self._every_n_epochs > 0, "Train epoch end should not be called if every_n_epochs is not active."
        assert (
            epoch_idx % self._every_n_epochs == 0
        ), f"Train epoch end called at epoch idx {epoch_idx} but every {self._every_n_epochs} epochs."
        assert (not self._main_process_only) or (
            jax.process_index() == 0
        ), "Train epoch end should only be called in the main process."

    def on_filtered_validation_epoch_start(self, epoch_idx: int, step_idx: int):
        """Test function for validation epoch start."""
        del step_idx
        assert self._every_n_epochs > 0, "Validation epoch start should not be called if every_n_epochs is not active."
        assert (
            epoch_idx % self._every_n_epochs == 0
        ), f"Validation epoch start called at epoch idx {epoch_idx} but every {self._every_n_epochs} epochs."
        assert (not self._main_process_only) or (
            jax.process_index() == 0
        ), "Validation epoch start should only be called in the main process."

    def on_filtered_validation_epoch_end(self, val_metrics: Metrics, epoch_idx: int, step_idx: int):
        """Test function for validation epoch end."""
        del val_metrics, step_idx
        assert self._every_n_epochs > 0, "Validation epoch end should not be called if every_n_epochs is not active."
        assert (
            epoch_idx % self._every_n_epochs == 0
        ), f"Validation epoch end called at epoch idx {epoch_idx} but every {self._every_n_epochs} epochs."
        assert (not self._main_process_only) or (
            jax.process_index() == 0
        ), "Validation epoch end should only be called in the main process."

    def on_filtered_training_step(self, step_metrics: Metrics, epoch_idx: int, step_idx: int):
        """Test function for training step."""
        del step_metrics, epoch_idx
        assert self._every_n_steps > 0, "Train step should not be called if every_n_steps is not active."
        assert (
            step_idx % self._every_n_steps == 0
        ), f"Train step called at step idx {step_idx} but every {self._every_n_steps} steps."
        assert (not self._main_process_only) or (
            jax.process_index() == 0
        ), "Train step should only be called in the main process."


class ToyCallbackConfig(CallbackConfig):
    """Configuration for the ToyCallback."""

    def create(self, trainer: TrainerModule, data_module: Any | None = None) -> Callback:
        return ToyCallback(self, trainer, data_module)


@pytest.mark.parametrize("every_n_epochs", [-1, 1, 2, 7])
@pytest.mark.parametrize("every_n_steps", [-1, 1, 2, 7])
@pytest.mark.parametrize("main_process_only", [True, False])
def test_callback_filtered_methods(
    mse_trainer: Any, toy_model: Any, tmp_path: Path, every_n_epochs: int, every_n_steps: int, main_process_only: bool
):
    """
    Tests callback if filtered methods are running at expected epochs and steps.
    """
    MSETrainer = mse_trainer
    ToyModel = toy_model

    log_path = (
        tmp_path
        / "test_callback_filtered_methods"
        / f"epoch_{every_n_epochs}_step_{every_n_epochs}_main_{main_process_only}"
    )
    trainer = MSETrainer(
        TrainerConfig(
            callbacks=(
                ToyCallbackConfig(
                    every_n_epochs=every_n_epochs,
                    every_n_steps=every_n_steps,
                    main_process_only=main_process_only,
                ),
            ),
            logger=LoggerConfig(log_path=log_path),
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
