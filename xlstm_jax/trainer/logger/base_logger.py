#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import jax
import numpy as np

from xlstm_jax.common_types import HostMetrics, Metrics
from xlstm_jax.configs import ConfigDict
from xlstm_jax.trainer.metrics import get_metrics

from .cmd_logging import setup_logging_multiprocess

LOGGER = logging.getLogger(__name__)


@dataclass(kw_only=True, frozen=False)
class LoggerConfig(ConfigDict):
    """
    Configuration for the logger.

    Attributes:
        log_every_n_steps: The frequency at which logs should be written.
        log_path: The path where the logs should be written. If None, we will not write logs to disk.
        log_tools: A list of LoggerToolsConfig objects that should be used to log the metrics. These tools will be
            created in the Logger class.
        cmd_logging_name: The name of the output file for command line logging without suffix. The suffix `.log` will
            be added automatically.
    """

    log_every_n_steps: int = 1
    log_path: Path | None = None
    log_tools: list["LoggerToolsConfig"] = field(default_factory=list)
    cmd_logging_name: str = "output"

    @property
    def log_dir(self) -> str:
        """Returns the log directory as a string."""
        return self.log_path.as_posix()


@dataclass(kw_only=True, frozen=False)
class LoggerToolsConfig(ConfigDict):
    """
    Base config class for logger tools.

    These are tools that can be used to log metrics, images, etc. They are created inside the Logger class.
    """

    def create(self, logger: "Logger") -> "LoggerTool":
        """Creates the logger tool."""
        raise NotImplementedError


class Logger:
    """Logger class to log metrics, images, etc."""

    def __init__(self, config: LoggerConfig, metric_postprocess_fn: Callable[[HostMetrics], HostMetrics] | None = None):
        """
        Base Logger Class.

        Args:
            config: The logger config.
        """
        self.config = config
        self.log_path = config.log_path
        self.metric_postprocess_fn = metric_postprocess_fn if metric_postprocess_fn is not None else lambda x: x
        if self.log_path is not None:
            self.log_path.mkdir(parents=True, exist_ok=True)
            setup_logging_multiprocess(logfile=self.log_path / f"{config.cmd_logging_name}.log")
        if jax.process_index() == 0:
            self.log_tools = [tool_config.create(self) for tool_config in config.log_tools]
        else:
            self.log_tools = []
        self.epoch = 0
        self.step = 0
        self.found_nans = False
        self.last_step = {}
        self.last_step_time = None
        self.epoch_start_time_stack = []
        self.mode_stack = []

    @property
    def mode(self) -> Literal["default", "train", "val", "test"]:
        """
        Current logging mode. Can be "default", "train", "val", or "test".

        Returns:
            str: The current logging mode.
        """
        if len(self.mode_stack) == 0:
            return "default"
        return self.mode_stack[-1]

    def log_config(self, config: ConfigDict | dict[str, ConfigDict]):
        """
        Logs the configuration.

        Args:
            config: The configuration to log. Can also be a dictionary of multiple configurations.
        """
        LOGGER.info("Logging config.")
        for tool in self.log_tools:
            tool.log_config(config)

    def on_training_start(self):
        """Set up the logger for training."""
        LOGGER.info("Starting training.")
        for tool in self.log_tools:
            tool.setup()

    def start_epoch(self, epoch: int, step: int, mode: Literal["train", "val", "test"] = "train"):
        """
        Starts a new epoch.

        To be called before starting a new training, eval or test epoch. Can also be called if one is still
        in another epoch. For instance, if the training epoch is interrupted by a validation epoch, the logger
        switches to the validation mode until a `end_epoch` is called. Then, the logger switches back to the
        training mode.

        Args:
            epoch: The index of the epoch.
            step: The index of the global training step.
            mode: The logging mode. Should be in {"train", "val", "test"}. Defaults to "train".
        """
        LOGGER.info(f"Starting epoch {epoch} at step {step} in mode {mode}.")
        self.epoch_start_time_stack.append(time.time())
        self.last_step = {"time": time.time(), "step": step}
        self.epoch = epoch
        self.mode_stack.append(mode)

    def log_step(self, metrics: Metrics, step: int) -> Metrics:
        """
        Log metrics for a single step.

        Args:
            metrics: The metrics to log. Should follow the structure of the metrics in the metrics.py file.
            step: The current step.

        Returns:
            If the metrics are logged in this step, the metrics will be updated to reset all metrics.
            If the metrics are not logged in this step, the metrics will be returned unchanged.
        """
        if step % self.config.log_every_n_steps == 0:
            LOGGER.info(f"Logging at step {step} in epoch {self.epoch} in mode {self.mode}.")

            # Get metrics to host and reset them.
            metrics, host_metrics = get_metrics(metrics, reset_metrics=True)
            host_metrics = self.metric_postprocess_fn(host_metrics)

            # Add the epoch and step time to the host metrics.
            host_metrics["epoch"] = self.epoch
            if self.last_step is not None and step != self.last_step["step"]:
                host_metrics["step_time"] = (time.time() - self.last_step["time"]) / (step - self.last_step["step"])
            self.last_step = {"time": time.time(), "step": step}

            # Check for NaNs in the metrics.
            self._check_for_nans(host_metrics, step)

            # Send the metrics to the logger tools.
            for tool in self.log_tools:
                tool.log_metrics(host_metrics, step, self.epoch, self.mode)
        elif self.last_step is None:
            # If the last step is None, set it to the current time and step.
            self.last_step = {"time": time.time(), "step": step}
        return metrics

    def _check_for_nans(self, host_metrics: HostMetrics, step: int | None = None):
        """
        Check if any of the metrics contain NaNs.

        If `NaN` are found, a warning is logged and the `found_nans` attribute is set to `True`.

        Args:
            host_metrics: The metrics to check.
            step: The step at which the metrics were logged. Used for logging if provided.
        """
        for key, value in host_metrics.items():
            if isinstance(value, float) and np.isnan(value).any():
                step_str = f" at step {step}" if step is not None else ""
                LOGGER.warning(f"NaN found in metric {key}{step_str} in epoch {self.epoch} (mode {self.mode}).")
                self.found_nans = True

    def log_host_metrics(self, host_metrics: HostMetrics, step: int, mode: str | None = None):
        """
        Logs a dictionary of metrics on the host.

        Can be used by callbacks to log additional metrics.

        Args:
            host_metrics: The metrics to log.
            step: The current step.
            mode: The mode / prefix with which to log the metrics. If None, the current mode is used.
        """
        mode = mode if mode is not None else self.mode
        for tool in self.log_tools:
            tool.log_metrics(host_metrics, step, self.epoch, mode)

    def end_epoch(
        self,
        metrics: Metrics,
        step: int,
    ) -> tuple[Metrics, HostMetrics]:
        """
        Ends the current epoch and logs the epoch metrics.

        If any other epoch is still running, the logger will switch back to that epoch.

        Args:
            metrics: The metrics that should be logged in this epoch.
            step: The current step.

        Returns:
            The originally passed metric dict and potentially any other metrics that should be passed
            to callbacks later on. Note that the metrics will not be reset.
        """
        if len(self.mode_stack) == 0:
            LOGGER.warning("end_epoch was called, but no epoch was started. Skipping end_epoch.")
            return metrics, {}

        # Get the mode and pop it from the stack.
        mode = self.mode_stack.pop()

        # Calculate the epoch time.
        epoch_start_time = self.epoch_start_time_stack.pop()
        epoch_time = time.time() - epoch_start_time
        if mode != "train":
            metrics, host_metrics = get_metrics(metrics, reset_metrics=False)
            host_metrics = self.metric_postprocess_fn(host_metrics)
            host_metrics["epoch_time"] = epoch_time
        else:
            host_metrics = {"epoch_time": epoch_time}

        # Check for NaNs in the metrics.
        self._check_for_nans(host_metrics)

        # Log the epoch metrics in the log tools.
        for tool in self.log_tools:
            tool.log_metrics(host_metrics, step, self.epoch, mode)

        # Reset step time. If it was validation, it effectively skips the validation time in the train step times.
        self.last_step = None

        LOGGER.info(f"Epoch in mode {mode} finished in {epoch_time:.2f} seconds.")
        return metrics, host_metrics

    def finalize(self, status: str):
        """
        Closes the logger.

        Args:
            status: The status of the training run (e.g. success, failure).
        """
        for tool in self.log_tools:
            tool.finalize(status)
        LOGGER.info(f"Finished logger with status: {status}.")


class LoggerTool:
    """Base class for logger tools."""

    def log_config(self, config: ConfigDict | dict[str, ConfigDict]):
        """
        Log the configuration to the tool.

        Args:
            config: The configuration to log.
        """

    def log_metrics(self, metrics: HostMetrics, step: int, epoch: int, mode: str):
        """
        Log the metrics to the tool.

        Args:
            metrics: The metrics to log.
            step: The current step.
            epoch: The current epoch.
            mode: The current mode (train, val, test).
        """
        raise NotImplementedError

    def finalize(self, status: str):
        """Finalize and close the tool."""
