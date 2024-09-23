from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import jax
import orbax.checkpoint as ocp
from absl import logging

from xlstm_jax.import_utils import class_to_name
from xlstm_jax.trainer.callbacks.callback import Callback, CallbackConfig
from xlstm_jax.trainer.metrics import Metrics


@dataclass(kw_only=True, frozen=True)
class ModelCheckpointConfig(CallbackConfig):
    """
    Configuration for the ModelCheckpoint callback.

    By default, the checkpoint saves the model parameters, training step, random number generator state, and metadata
    to the logging directory. The metadata includes the trainer, model, and optimizer configurations.

    Attributes:
        max_to_keep: Number of checkpoints to keep. If None, keeps all checkpoints. Otherwise, keeps the most recent
            `max_to_keep` checkpoints. If `monitor` is set, keeps the best `max_to_keep` checkpoints instead of the
            most recent.
        monitor: Metric to monitor for saving the model. Should be a key of the evaluation metrics. If None, checkpoints
            are sorted by recency.
        mode: One of {"min", "max"}. If "min", saves the model with the smallest value of the monitored metric. If
            "max", saves the model with the largest value of the monitored metric.
        save_optimizer_state: Whether to save the optimizer state.
        enable_async_checkpointing: Whether to enable asynchronous checkpointing. See orbax documentation for more
            information.
    """

    max_to_keep: int | None = 1
    monitor: str | None = None
    mode: Literal["min", "max"] = "min"
    save_optimizer_state: bool = True
    enable_async_checkpointing: bool = True

    def create(self, trainer: Any, data_module: Any = None) -> "ModelCheckpoint":
        """
        Creates the ModelCheckpoint callback.

        Args:
            trainer: Trainer object.
            data_module (optional): Data module object.

        Returns:
            ModelCheckpoint object.
        """
        return ModelCheckpoint(self, trainer, data_module)


class ModelCheckpoint(Callback):
    """Callback to save model parameters and mutable variables to the logging directory."""

    def __init__(self, config: ModelCheckpointConfig, trainer: Any, data_module: Any | None = None):
        """
        Initialize the Model Checkpoint callback.

        Sets up an orbax checkpoint manager to save model parameters, training step, random number generator state, and
        metadata to the logging directory.

        Args:
            config: The configuration for the ModelCheckpoint callback.
            trainer: The trainer object.
            data_module: The data module object.
        """
        super().__init__(config, trainer, data_module)
        assert self.trainer.log_path is not None, "Log directory must be set in the trainer if using ModelCheckpoint."
        self.log_path: Path = self.trainer.log_path

        options = ocp.CheckpointManagerOptions(
            max_to_keep=self.config.max_to_keep,
            best_fn=lambda m: m[self.config.monitor] if self.config.monitor is not None else None,
            best_mode=self.config.mode,
            step_prefix="checkpoint",
            cleanup_tmp_directories=True,
            create=True,
            enable_async_checkpointing=self.config.enable_async_checkpointing,
        )
        self.metadata = {
            "trainer": self.trainer.trainer_config.to_dict(),
            "model": self.trainer.model_config.to_dict(),
            "optimizer": self.trainer.optimizer_config.to_dict(),
        }
        self.metadata = jax.tree.map(class_to_name, self.metadata)
        item_handlers = {
            "step": ocp.ArrayCheckpointHandler(),
            "params": ocp.StandardCheckpointHandler(),
            "rng": ocp.ArrayCheckpointHandler(),
            "metadata": ocp.JsonCheckpointHandler(),
        }
        if self.trainer.state.mutable_variables is not None:
            item_handlers["mutable_variables"] = ocp.StandardCheckpointHandler()
        if self.config.save_optimizer_state:
            item_handlers["opt_state"] = ocp.StandardCheckpointHandler()
        self.manager = ocp.CheckpointManager(
            directory=(self.log_path / "checkpoints").absolute().as_posix(),
            item_names=tuple(item_handlers.keys()),
            item_handlers=item_handlers,
            options=options,
        )

    def on_filtered_validation_epoch_end(self, eval_metrics, epoch_idx: int, step_idx: int):
        """
        Saves the model at the end of the validation epoch.

        Args:
            eval_metrics: Dictionary of evaluation metrics. If a monitored metric is set, the model is saved based on
                the monitored metrics in this dictionary. If the monitored metric is not found, an error is raised.
                The metrics are saved along with the model.
            epoch_idx: Index of the current epoch.
            step_idx: Index of the current step.
        """
        del epoch_idx
        self.save_model(eval_metrics, step_idx)

    def save_model(self, eval_metrics: Metrics, step_idx: int):
        """
        Saves model state dict to the logging directory.

        Args:
            eval_metrics: Dictionary of evaluation metrics. If a monitored metric is set, the model is saved based on
                the monitored metrics in this dictionary. If the monitored metric is not found, an error is raised.
                The metrics are saved along with the model.
            step_idx: Index of the current step.
        """
        logging.info(f"Saving model at step {step_idx} with eval metrics {eval_metrics}.")
        if self.config.monitor is not None:
            assert (
                self.config.monitor in eval_metrics
            ), f"Metric '{self.config.monitor}' not found in available eval metrics: {', '.join(eval_metrics)}"
        save_items = {
            "step": ocp.args.ArraySave(self.trainer.state.step),
            "params": ocp.args.StandardSave(self.trainer.state.params),
            "rng": ocp.args.ArraySave(self.trainer.state.rng),
            "metadata": ocp.args.JsonSave(self.metadata),
        }
        if self.trainer.state.mutable_variables is not None:
            save_items["mutable_variables"] = ocp.args.StandardSave(self.trainer.state.mutable_variables)
        if self.config.save_optimizer_state:
            save_items["opt_state"] = ocp.args.StandardSave(self.trainer.state.opt_state)
        eval_metrics = {
            k: eval_metrics[k] for k in eval_metrics if isinstance(eval_metrics[k], (int, float, str, bool))
        }
        save_items = ocp.args.Composite(**save_items)
        self.manager.save(step_idx, args=save_items, metrics=eval_metrics)

    def load_model(self, step_idx: int = -1, load_best: bool = False):
        """
        Loads model parameters and variables from the logging directory.

        Args:
            step_idx: Index of the step to load. If -1, loads the latest step by default.
            load_best: If True and step_idx is -1, loads the best checkpoint
                based on the monitored metric instead of the latest checkpoint.

        Returns:
            Dictionary of loaded model parameters and additional variables.
        """
        logging.info(f"Loading model at step {step_idx}.")
        if step_idx == -1:
            if load_best:
                step_idx = self.manager.best_step()
            else:
                step_idx = self.manager.latest_step()
        args = {
            "step": ocp.args.ArrayRestore(self.trainer.state.step),
            "params": ocp.args.StandardRestore(self.trainer.state.params),
            "rng": ocp.args.ArrayRestore(self.trainer.state.rng),
            "metadata": ocp.args.JsonRestore(self.metadata),
        }
        if self.trainer.state.mutable_variables is not None:
            args["mutable_variables"] = ocp.args.StandardRestore(self.trainer.state.mutable_variables)
        if self.config.save_optimizer_state:
            args["opt_state"] = ocp.args.StandardRestore(self.trainer.state.opt_state)
        state_dict = self.manager.restore(
            step_idx,
            args=ocp.args.Composite(**args),
        )
        state_dict = {k: v for k, v in state_dict.items() if v is not None}
        return state_dict

    def finalize(self, status: str | None = None):
        """
        Closes the checkpoint manager.

        Args:
            status: The status of the training run (e.g. success, failure).
        """
        del status
        logging.info("Closing checkpoint manager")
        self.manager.wait_until_finished()
        self.manager.close()
