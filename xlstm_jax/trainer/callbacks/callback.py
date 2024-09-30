from dataclasses import dataclass
from typing import Any

import jax

from xlstm_jax.configs import ConfigDict
from xlstm_jax.trainer.metrics import Metrics


@dataclass(kw_only=True, frozen=True)
class CallbackConfig(ConfigDict):
    """Base configuration of a callback."""

    every_n_epochs: int = 1
    """If the callback implements functions on a per-epoch basis (e.g. `on_training_epoch_start`,
    `on_training_epoch_end`, `on_validation_epoch_start`), this parameter specifies the frequency of calling these
    functions."""
    every_n_steps: int = -1
    """If the callback implements functions on a per-step basis (e.g. `on_training_step`), this parameter specifies the
    frequency of calling these functions."""
    main_process_only: bool = False
    """Whether to call the callback only in the main process."""

    def create(self, trainer: Any, data_module: Any = None) -> "Callback":
        """
        Creates the callback object.

        Args:
            trainer: Trainer object.
            data_module (optional): Data module object.

        Returns:
            Callback object.
        """
        del trainer, data_module
        raise NotImplementedError("Method `create` must be implemented by subclasses.")


class Callback:
    """
    Base class for callbacks.

    Callbacks are used to perform additional actions during training, validation, and testing. We provide a set of
    predefined functions, that can be overridden by subclasses to implement custom behavior. The predefined functions
    are called at the beginning and end of training, validation, and testing, as well as at the beginning and end of
    each epoch or step.

    Note: all counts of epoch index and step index are starting at 1 (i.e. the first epoch is 1 instead of 0).

    Args:
        config: Configuration dictionary.
        trainer: Trainer object.
        data_module (optional): Data module object.
    """

    def __init__(self, config: ConfigDict, trainer: Any, data_module: Any | None = None):
        self.config = config  # TODO: define CallbackConfig
        self.trainer = trainer
        self.data_module = data_module
        self._every_n_epochs = config.every_n_epochs
        self._every_n_steps = config.every_n_steps
        self._main_process_only = config.main_process_only
        self._active_on_epochs = (self._every_n_epochs > 0) and (
            not self._main_process_only or jax.process_index() == 0
        )
        self._active_on_steps = (self._every_n_steps > 0) and (not self._main_process_only or jax.process_index() == 0)

    def on_training_start(self):
        """Called at the beginning of training."""

    def on_training_end(self):
        """Called at the end of training."""

    def on_training_epoch_start(self, epoch_idx: int):
        """
        Called at the beginning of each training epoch.

        Args:
            epoch_idx: Index of the current epoch.
        """
        if self._active_on_epochs and epoch_idx % self._every_n_epochs == 0:
            self.on_filtered_training_epoch_start(epoch_idx)

    def on_filtered_training_epoch_start(self, epoch_idx: int):
        """
        Called at the beginning of each `every_n_epochs` training epoch. To be implemented by subclasses.

        Args:
            epoch_idx: Index of the current epoch.
        """

    def on_training_epoch_end(self, train_metrics: Metrics, epoch_idx: int):
        """
        Called at the end of each training epoch.

        Args:
            train_metrics: Dictionary of training metrics of the current epoch.
            epoch_idx: Index of the current epoch.
        """
        if self._active_on_epochs and epoch_idx % self._every_n_epochs == 0:
            self.on_filtered_training_epoch_end(train_metrics, epoch_idx)

    def on_filtered_training_epoch_end(self, train_metrics: Metrics, epoch_idx: int):
        """
        Called at the end of each `every_n_epochs` training epoch. To be implemented by subclasses.

        Args:
            train_metrics: Dictionary of training metrics of the current epoch.
            epoch_idx: Index of the current epoch.
        """

    def on_training_step(self, step_metrics: Metrics, epoch_idx: int, step_idx: int):
        """
        Called at the end of each training step.

        Args:
            step_metrics: Dictionary of training metrics of the current step.
            epoch_idx: Index of the current epoch.
            step_idx: Index of the current step.
        """
        if self._active_on_steps and step_idx % self._every_n_steps == 0:
            self.on_filtered_training_step(step_metrics, epoch_idx, step_idx)

    def on_filtered_training_step(self, step_metrics: Metrics, epoch_idx: int, step_idx: int):
        """
        Called at the end of each `every_n_steps` training step. To be implemented by subclasses.

        Args:
            step_metrics: Dictionary of training metrics of the current step.
            epoch_idx: Index of the current epo.
            step_idx: Index of the current step.
        """

    def on_validation_epoch_start(self, epoch_idx: int, step_idx: int):
        """
        Called at the beginning of validation.

        Args:
            epoch_idx: Index of the current training epoch.
            step_idx: Index of the current training step.
        """
        if self._active_on_epochs and epoch_idx % self._every_n_epochs == 0:
            self.on_filtered_validation_epoch_start(epoch_idx, step_idx)

    def on_filtered_validation_epoch_start(self, epoch_idx: int, step_idx: int):
        """
        Called at the beginning of `every_n_epochs` validation. To be implemented by subclasses.

        Args:
            epoch_idx: Index of the current training epoch.
            step_idx: Index of the current training step.
        """

    def on_validation_epoch_end(self, eval_metrics: Metrics, epoch_idx: int, step_idx: int):
        """
        Called at the end of each validation epoch.

        Args:
            eval_metrics: Dictionary of evaluation metrics of the current epoch.
            epoch_idx: Index of the current training epoch.
            step_idx: Index of the current training step.
        """
        if self._active_on_epochs and epoch_idx % self._every_n_epochs == 0:
            self.on_filtered_validation_epoch_end(eval_metrics, epoch_idx, step_idx)

    def on_filtered_validation_epoch_end(self, eval_metrics: Metrics, epoch_idx: int, step_idx: int):
        """
        Called at the end of each `every_n_epochs` validation epoch. To be implemented by subclasses.

        Args:
            eval_metrics: Dictionary of evaluation metrics of the current epoch.
            epoch_idx: Index of the current training epoch.
            step_idx: Index of the current training step.
        """

    def on_test_epoch_start(self, epoch_idx: int):
        """
        Called at the beginning of testing.

        To be implemented by subclasses.

        Args:
            epoch_idx: Index of the current epoch.
        """

    def on_test_epoch_end(self, test_metrics: Metrics, epoch_idx: int):
        """
        Called at the end of each test epoch. To be implemented by subclasses.

        Args:
            test_metrics: Dictionary of test metrics of the current epoch.
            epoch_idx: Index of the current epoch.
        """

    def set_dataset(self, data_module: Any):
        """
        Sets the data module.

        Args:
            data_module: Data module object.
        """
        self.data_module = data_module

    def finalize(self, status: str | None = None):
        """
        Called at the end of the whole training process.

        To be implemented by subclasses.

        Args:
            status: Status of the training process.
        """
