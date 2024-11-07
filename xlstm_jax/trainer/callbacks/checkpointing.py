import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import orbax.checkpoint as ocp

from xlstm_jax.import_utils import class_to_name
from xlstm_jax.trainer.base.param_utils import flatten_dict
from xlstm_jax.trainer.callbacks.callback import Callback, CallbackConfig
from xlstm_jax.trainer.data_module import DataloaderModule
from xlstm_jax.trainer.metrics import Metrics

LOGGER = logging.getLogger(__name__)


@dataclass(kw_only=True, frozen=False)
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
        save_dataloader_state: Whether to save the dataloader state.
        enable_async_checkpointing: Whether to enable asynchronous checkpointing. See orbax documentation for more
            information.
        log_path: Path to save the checkpoints as subfolder to. If None, saves to the logging directory of the trainer.
    """

    max_to_keep: int | None = 1
    monitor: str | None = None
    mode: str = "min"
    save_optimizer_state: bool = True
    save_dataloader_state: bool = True
    enable_async_checkpointing: bool = True
    log_path: Path | None = None

    def __post_init__(self):
        assert self.mode in ["min", "max"], "Mode must be one of {'min', 'max'}."

    def create(self, trainer: Any, data_module: DataloaderModule | None = None) -> "ModelCheckpoint":
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
    """
    Callback to save model parameters and mutable variables to the logging directory.

    Sets up an orbax checkpoint manager to save model parameters, training step, random number generator state, and
    metadata to the logging directory.

    Args:
        config: The configuration for the ModelCheckpoint callback.
        trainer: The trainer object. If the trainer has no optimizer attribute, the optimizer part will not be loaded.
        data_module: The data module object.
    """

    def __init__(self, config: ModelCheckpointConfig, trainer: Any, data_module: DataloaderModule | None = None):
        super().__init__(config, trainer, data_module)
        if self.config.log_path is not None:
            self.log_path = self.config.log_path
        else:
            assert (
                self.trainer.log_path is not None
            ), "Log directory must be set in the trainer if using ModelCheckpoint."
            self.log_path: Path = self.trainer.log_path
        self.checkpoint_path = self.log_path / "checkpoints"
        if not self.checkpoint_path.exists():
            self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.dataloader_path = self.log_path / "checkpoints_dataloaders"
        if self.config.save_dataloader_state:
            self.dataloader_path.mkdir(parents=True, exist_ok=True)

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
            directory=self.checkpoint_path.absolute().as_posix(),
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
        if self.config.save_dataloader_state:
            self.save_dataloader(step_idx)

    def save_model(self, eval_metrics: Metrics, step_idx: int):
        """
        Saves model state dict to the logging directory.

        Args:
            eval_metrics: Dictionary of evaluation metrics. If a monitored metric is set, the model is saved based on
                the monitored metrics in this dictionary. If the monitored metric is not found, an error is raised.
                The metrics are saved along with the model.
            step_idx: Index of the current step.
        """
        LOGGER.info(f"Saving model at step {step_idx} with eval metrics {eval_metrics}.")
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

    def save_dataloader(self, step_idx: int):
        """
        Saves the dataloader state to the logging directory.

        Args:
            step_idx: Index of the current step.
        """
        LOGGER.info(f"Saving dataloader state at step {step_idx}.")
        if self.data_module is None:
            LOGGER.warning("Data module not set. Skipping dataloader state save.")
            return
        checkpoint_path = self.dataloader_path / f"checkpoint_{step_idx}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Collect dataloaders.
        loaders = flatten_dict(
            {
                "train": self.data_module.train_dataloader,
                "val": self.data_module.val_dataloader,
                "test": self.data_module.test_dataloader,
            },
            separator="_",
        )

        # Save metadata for dataloaders.
        if jax.process_index() == 0:
            metadata = {
                "step": step_idx,
                "process_count": jax.process_count(),
                "primary_host": jax.process_index(),
                "train_exists": self.data_module.train_dataloader is not None,
                "val_exists": self.data_module.val_dataloader is not None,
                "test_exists": self.data_module.test_dataloader is not None,
                "loader_names": sorted(list(loaders.keys())),
            }
            metadata_path = checkpoint_path / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f)
            LOGGER.info(f"Saved metadata for dataloader state at step {step_idx}: {metadata}.")

        # Save state for each dataloader.
        for name, loader in loaders.items():
            if loader is None:
                # Test loader might be None.
                continue
            elif hasattr(loader, "get_state"):
                state_dict = loader.get_state()
                # Save with pickle to support custom objects.
                dir_path = checkpoint_path / name
                dir_path.mkdir(parents=True, exist_ok=True)
                file_path = dir_path / f"process_{jax.process_index()}.pkl"
                LOGGER.info(f"Saving dataloader state for {name} in process {jax.process_index()}.")
                with open(file_path, "wb") as f:
                    pickle.dump(state_dict, f)
            else:
                LOGGER.warning(f"No dataloader state found for {name}. Skipping save.")

    def load_model(self, step_idx: int = -1, load_best: bool = False) -> dict[str, Any]:
        """
        Loads model parameters and variables from the logging directory.

        Args:
            step_idx: Index of the step to load. If -1, loads the latest step by default.
            load_best: If True and step_idx is -1, loads the best checkpoint
                based on the monitored metric instead of the latest checkpoint.

        Returns:
            Dictionary of loaded model parameters and additional variables.
        """
        step_idx = self.resolve_step_idx(step_idx, load_best)
        LOGGER.info(f"Loading model at step {step_idx}.")
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

    def load_dataloader(self, step_idx: int = -1, load_best: bool = False) -> dict[str, Any]:
        """
        Loads the dataloader state from the logging directory.

        Args:
            step_idx: Index of the step to load. If -1, loads the latest step by default.
            load_best: If True and step_idx is -1, loads the best checkpoint
                based on the monitored metric instead of the latest checkpoint.

        Returns:
            Dictionary of loaded dataloader states.
        """
        step_idx = self.resolve_step_idx(step_idx, load_best)
        LOGGER.info(f"Loading dataloader state at step {step_idx}.")

        # Check that checkpoint path exists.
        checkpoint_path = self.dataloader_path / f"checkpoint_{step_idx}"
        if not checkpoint_path.exists():
            LOGGER.warning(f"No dataloader state found at step {step_idx}.")
            return {}

        # Check metadata and that process count matches.
        metadata_path = checkpoint_path / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)
        LOGGER.info(f"Loaded metadata for dataloader state at step {step_idx}: {metadata}.")
        if metadata["process_count"] != jax.process_count():
            LOGGER.warning(
                f"Process count mismatch. Expected {metadata['process_count']} but got {jax.process_count()}."
            )
            return {}

        # Load state for each dataloader.
        state_dict = {}
        for name in metadata["loader_names"]:
            dir_path = checkpoint_path / name
            if not dir_path.exists():
                LOGGER.info(f"No dataloader state found for {name}.")
                continue
            file_path = dir_path / f"process_{jax.process_index()}.pkl"
            if file_path.exists():
                with open(file_path, "rb") as f:
                    state_dict[name] = pickle.load(f)
            else:
                LOGGER.warning(f"No dataloader state found for {name} in process {jax.process_index()}.")
        return state_dict

    def resolve_step_idx(self, step_idx: int, load_best: bool) -> int:
        """
        Resolves the step index to load.

        Args:
            step_idx: Index of the step to load. If -1, loads the latest step by default.
            load_best: If True and step_idx is -1, loads the best checkpoint
                based on the monitored metric instead of the latest checkpoint.

        Returns:
            The resolved step index.
        """
        if step_idx != -1:
            return step_idx
        if load_best:
            step_idx = self.manager.best_step()
        else:
            step_idx = self.manager.latest_step()
        return step_idx

    def finalize(self, status: str | None = None):
        """
        Closes the checkpoint manager.

        Args:
            status: The status of the training run (e.g. success, failure).
        """
        del status
        LOGGER.info("Closing checkpoint manager")
        self.manager.wait_until_finished()
        self.manager.close()


def load_pretrained_model(
    checkpoint_path: Path,
    trainer: Any,
    step_idx: int = -1,
    load_optimizer: bool = True,
    load_best: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    """
    Loads a pretrained model from a checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint directory.
        trainer: Trainer object.
        data_module (optional): Data module object.
        step_idx: Index of the step to load. If -1, loads the latest step by default.
        load_optimizer: If True the optimizer state is loaded from the checkpoint.
        load_best: If True and step_idx is -1, loads the best checkpoint
            based on the monitored metric instead of the latest checkpoint.

    Returns:
        Dictionary of loaded model parameters and additional variables, as well as the dataloader state
        and the resolved step index that was loaded.
    """
    config = ModelCheckpointConfig(log_path=checkpoint_path, save_optimizer_state=load_optimizer)
    callback = ModelCheckpoint(config, trainer, None)
    step_idx = callback.resolve_step_idx(step_idx, load_best)
    state_dict = callback.load_model(step_idx)
    data_module_state = callback.load_dataloader(step_idx)
    callback.finalize()
    return (state_dict, data_module_state, step_idx)
