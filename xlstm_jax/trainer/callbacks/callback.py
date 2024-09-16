from dataclasses import dataclass
from typing import Any, Optional

from xlstm_jax.configs import ConfigDict
from xlstm_jax.trainer.metrics import Metrics

import jax


@dataclass(kw_only=True, frozen=True)
class CallbackConfig(ConfigDict):
    every_n_epochs: int = 1
    every_n_steps: int = -1
    main_process_only: bool = False


class Callback:
    """Base class for callbacks.

    Callbacks are used to perform additional actions during training, validation, and testing.
    We provide a set of predefined functions, that can be overridden by subclasses to implement
    custom behavior. The predefined functions are called at the beginning and end of training,
    validation, and testing, as well as at the beginning and end of each epoch or step.
    """

    def __init__(self, config: ConfigDict, trainer: Any, data_module: Any | None = None):
        """Base class for callbacks.

        Args:
            config: Configuration dictionary.
            trainer: Trainer object.
            data_module (optional): Data module object.
        """
        self.config = config
        self.trainer = trainer
        self.data_module = data_module
        self.every_n_epochs = config.every_n_epochs
        self.every_n_steps = config.every_n_steps
        self.main_process_only = config.main_process_only

    def on_training_start(self):
        """Called at the beginning of training."""
        pass

    def on_training_end(self):
        """Called at the end of training."""
        pass

    def on_training_epoch_start(self, epoch_idx: int):
        """Called at the beginning of each training epoch.

        Args:
            epoch_idx: Index of the current epoch.
        """
        if epoch_idx % self.every_n_epochs != 0 or (self.main_process_only and jax.process_index() != 0):
            return
        self.on_filtered_training_epoch_start(epoch_idx)

    def on_filtered_training_epoch_start(self, epoch_idx: int):
        """Called at the beginning of each `every_n_epochs` training epoch. To be implemented by
        subclasses.

        Args:
            epoch_idx: Index of the current epoch.
        """
        pass

    def on_training_epoch_end(self, train_metrics: Metrics, epoch_idx: int):
        """Called at the end of each training epoch.

        Args:
            train_metrics: Dictionary of training metrics of the current epoch.
            epoch_idx: Index of the current epoch.
        """
        if epoch_idx % self.every_n_epochs != 0 or (self.main_process_only and jax.process_index() != 0):
            return
        self.on_filtered_training_epoch_end(train_metrics, epoch_idx)

    def on_filtered_training_epoch_end(self, train_metrics: Metrics, epoch_idx: int):
        """Called at the end of each `every_n_epochs` training epoch. To be implemented by
        subclasses.

        Args:
            train_metrics: Dictionary of training metrics of the current epoch.
            epoch_idx: Index of the current epoch.
        """
        pass

    def on_training_step(self, step_metrics: Metrics, epoch_idx: int, step_idx: int):
        """Called at the end of each training step.

        Args:
            step_metrics: Dictionary of training metrics of the current step.
            epoch_idx: Index of the current epoch.
            step_idx: Index of the current step.
        """
        if step_idx % self.every_n_steps != 0 or (self.main_process_only and jax.process_index() != 0):
            return
        self.on_filtered_training_step(step_metrics, epoch_idx, step_idx)

    def on_filtered_training_step(self, step_metrics: Metrics, epoch_idx: int, step_idx: int):
        """Called at the end of each `every_n_steps` training step. To be implemented by
        subclasses.

        Args:
            step_metrics: Dictionary of training metrics of the current step.
            step_idx: Index of the current step.
        """
        pass

    def on_validation_epoch_start(self, epoch_idx: int):
        """Called at the beginning of validation.

        Args:
            epoch_idx: Index of the current epoch.
        """
        if epoch_idx % self.every_n_epochs != 0 or (self.main_process_only and jax.process_index() != 0):
            return
        self.on_filtered_validation_epoch_start(epoch_idx)

    def on_filtered_validation_epoch_start(self, epoch_idx: int):
        """Called at the beginning of `every_n_epochs` validation. To be implemented by subclasses.

        Args:
            epoch_idx: Index of the current epoch.
        """
        pass

    def on_validation_epoch_end(self, eval_metrics: Metrics, epoch_idx: int):
        """Called at the end of each validation epoch.

        Args:
            eval_metrics: Dictionary of evaluation metrics of the current epoch.
            epoch_idx: Index of the current epoch.
        """
        if epoch_idx % self.every_n_epochs != 0 or (self.main_process_only and jax.process_index() != 0):
            return
        self.on_filtered_validation_epoch_end(eval_metrics, epoch_idx)

    def on_filtered_validation_epoch_end(self, eval_metrics: Metrics, epoch_idx: int):
        """Called at the end of each `every_n_epochs` validation epoch. To be implemented by
        subclasses.

        Args:
            eval_metrics: Dictionary of evaluation metrics of the current epoch.
            epoch_idx: Index of the current epoch.
        """
        pass

    def on_test_epoch_start(self, epoch_idx: int):
        """Called at the beginning of testing.

        To be implemented by subclasses.

        Args:
            epoch_idx: Index of the current epoch.
        """
        pass

    def on_test_epoch_end(self, test_metrics: Metrics, epoch_idx: int):
        """Called at the end of each test epoch. To be implemented by subclasses.

        Args:
            test_metrics: Dictionary of test metrics of the current epoch.
            epoch_idx: Index of the current epoch.
        """
        pass

    def set_dataset(self, data_module: Any):
        """Sets the data module.

        Args:
            data_module: Data module object.
        """
        self.data_module = data_module

    def finalize(self, status: str | None = None):
        """Called at the end of the whole training process.

        To be implemented by subclasses.

        Args:
            status: Status of the training process.
        """
        pass
