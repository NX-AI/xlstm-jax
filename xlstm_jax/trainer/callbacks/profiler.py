import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax

from xlstm_jax.trainer.callbacks.callback import Callback, CallbackConfig
from xlstm_jax.trainer.data_module import DataloaderModule
from xlstm_jax.trainer.metrics import Metrics

LOGGER = logging.getLogger(__name__)


@dataclass(kw_only=True, frozen=False)
class JaxProfilerConfig(CallbackConfig):
    """Configuration for the JaxProfiler callback.

    Attributes:
        every_n_epochs: Unused in this callback.
        every_n_steps: Unused in this callback.
        main_process_only: If True, the profiler is only active in the main process.
            Otherwise, one profile per process is created.
        profile_every_n_minutes: Profile every n minutes. If set below 0, the profiler
            is only done once at the beginning.
        profile_first_step: The first step to start profiling.
        profile_n_steps: Number of steps to profile.
        profile_log_dir: Directory to save the profiler logs. By default set to
            "tensorboard", where also the TensorBoard logs are saved.
    """

    every_n_epochs: int = -1
    every_n_steps: int = -1
    main_process_only: bool = True
    profile_every_n_minutes: int = 60
    profile_first_step: int = 10
    profile_n_steps: int = 5
    profile_log_dir: str = "tensorboard"

    def create(self, trainer: Any, data_module: DataloaderModule | None = None) -> "JaxProfiler":
        """
        Creates the JaxProfiler callback.

        Args:
            trainer: Trainer object.
            data_module (optional): Data module object.

        Returns:
            JaxProfiler object.
        """
        return JaxProfiler(self, trainer, data_module)


class JaxProfiler(Callback):
    """Callback to profile model training steps."""

    def __init__(self, config: JaxProfilerConfig, trainer: Any, data_module: DataloaderModule | None = None):
        """
        Initialize the JaxProfiler callback.

        Args:
            config: The configuration for the JaxProfiler callback.
            trainer: The trainer object.
            data_module: The data module object.
        """
        super().__init__(config, trainer, data_module)
        assert self.trainer.log_path is not None, "Log directory must be set in the trainer if using the JaxProfiler."
        self.log_path: Path = self.trainer.log_path / self.config.profile_log_dir
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.profile_every_n_minutes = self.config.profile_every_n_minutes
        self.profile_first_step = self.config.profile_first_step
        self.profile_n_steps = self.config.profile_n_steps
        self.profiler_active = False
        self.profiler_last_time = None
        LOGGER.info("JaxProfiler initialized.")

    def on_training_start(self):
        """
        Called at the beginning of training.

        Starts tracking the time to determine when to start the profiler.
        """
        self.profiler_active = False
        self.profiler_last_time = time.time()

    def on_training_step(self, step_metrics: Metrics, epoch_idx: int, step_idx: int):
        """
        Called at the end of each training step.

        Starts the profiler if the current step is the first step or if the
        time since the last profiling is greater than the specified interval.
        If the profiler is active, it stops the profiler after the specified
        number of steps.

        Args:
            step_metrics: Dictionary of training metrics of the current step.
            epoch_idx: Index of the current epoch.
            step_idx: Index of the current step.
        """
        del step_metrics, epoch_idx
        if self.profiler_active:
            if step_idx >= self.profile_start_step + self.profile_n_steps:
                self.stop_trace()
        else:
            if (step_idx == self.profile_first_step) or (
                time.time() - self.profiler_last_time > self.profile_every_n_minutes * 60
            ):
                self.start_trace(step_idx)

    def on_training_epoch_end(self, train_metrics: Metrics, epoch_idx: int):
        """
        Called at the end of each training epoch.

        Stop the profiler if it is still active to prevent tracing
        non-training step operations.

        Args:
            train_metrics: Metrics of the current epoch.
            epoch_idx: Index of the current epoch.
        """
        del train_metrics, epoch_idx
        self.stop_trace()

    def on_validation_epoch_start(self, epoch_idx: int, step_idx: int):
        """
        Called at the beginning of validation.

        If profiler is active, stop it to prevent tracing all validation
        steps.

        Args:
            epoch_idx: Index of the current training epoch.
            step_idx: Index of the current training step.
        """
        del epoch_idx, step_idx
        self.stop_trace()

    def start_trace(self, step_idx: int):
        """
        Start the profiler trace.

        If the profiler is already active, a warning is logged.

        Args:
            step_idx: Index of the current training step.
        """
        if self.config.main_process_only and jax.process_index() != 0:
            LOGGER.info(f"Skipping profiling in non-main process {jax.process_index()}.")
            return
        if not self.profiler_active:
            LOGGER.info(f"Starting trace at step {step_idx}.")
            jax.profiler.start_trace(self.log_path)
            self.profiler_active = True
            self.profile_start_step = step_idx
        else:
            LOGGER.warning(f"Trace already active at step {step_idx}.")

    def stop_trace(self):
        """
        Stop the profiler trace.

        If the profiler is not active, nothing is done.
        """
        if self.profiler_active:
            LOGGER.info("Stopping trace.")
            # Wait until the parameters are all ready.
            jax.tree.map(lambda x: x.block_until_ready(), self.trainer.state.params)
            # Stop the profiler.
            jax.profiler.stop_trace()
            self.profiler_last_time = time.time()
            self.profiler_active = False
