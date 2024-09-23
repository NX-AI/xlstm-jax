import logging
from dataclasses import dataclass
from typing import Any

import jax

from xlstm_jax.trainer.callbacks.callback import Callback, CallbackConfig
from xlstm_jax.trainer.metrics import Metrics
from xlstm_jax.trainer.optimizer import build_lr_scheduler

LOGGER = logging.getLogger(__name__)


@dataclass(kw_only=True, frozen=True)
class LearningRateMonitorConfig(CallbackConfig):
    """Configuration for the LearningRateMonitor callback.

    Attributes:
        every_n_epochs: Log the learning rate every n epochs. Set to -1 to disable.
        every_n_steps: Log the learning rate every n steps. By default, logs every 50 steps.
        main_process_only: Log the learning rate only in the main process.
        log_lr_key: Key to use for logging the learning rate.
    """

    every_n_epochs: int = -1
    every_n_steps: int = 50
    main_process_only: bool = True
    log_lr_key: str = "optimizer/lr"

    def create(self, trainer: Any, data_module: Any = None) -> "LearningRateMonitor":
        """
        Creates the LearningRateMonitor callback.

        Args:
            trainer: Trainer object.
            data_module (optional): Data module object.

        Returns:
            LearningRateMonitor object.
        """
        return LearningRateMonitor(self, trainer, data_module)


class LearningRateMonitor(Callback):
    """Callback to monitor the learning rate."""

    def __init__(self, config: LearningRateMonitorConfig, trainer: Any, data_module: Any | None = None):
        super().__init__(config, trainer, data_module)
        self.lr_scheduler = jax.jit(
            build_lr_scheduler(self.trainer.optimizer_config.scheduler),
        )
        LOGGER.info("LearningRateMonitor initialized.")

    def on_filtered_training_step(self, step_metrics: Metrics, epoch_idx: int, step_idx: int):
        """
        Logs the learning rate after a step.

        Args:
            step_metrics: Metrics of the current step. Unused in this callback.
            epoch_idx: Index of the current epoch. Unused in this callback.
            step_idx: Index of the current step.
        """
        del step_metrics, epoch_idx
        self._log_lr(step_idx)

    def on_filtered_training_epoch_end(self, train_metrics: Metrics, epoch_idx: int):
        """
        Logs the learning rate after an epoch.

        Args:
            train_metrics: Metrics of the current epoch. Unused in this callback.
            epoch_idx: Index of the current epoch. Unused in this callback.
        """
        del train_metrics, epoch_idx
        self._log_lr(self.trainer.global_step)

    def _log_lr(self, step_idx: int):
        """
        Logs the learning rate.

        Args:
            step_idx: Index of the current step.
        """
        # Run on CPU to avoid unnecessary GPU usage.
        with jax.default_device(jax.local_devices(backend="cpu")[0]):
            lr: jax.Array = self.lr_scheduler(step_idx)
            lr = jax.device_get(lr)  # Transfer back to host, i.e. same CPU device.
        self.trainer.logger.log_host_metrics({self.config.log_lr_key: lr}, step=step_idx)
