import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from xlstm_jax.import_utils import class_to_name
from xlstm_jax.trainer.callbacks.callback import Callback, CallbackConfig

import jax
import orbax.checkpoint as ocp
from absl import logging
from flax.training import orbax_utils


@dataclass(kw_only=True, frozen=True)
class ModelCheckpointConfig(CallbackConfig):
    monitor: str
    save_top_k: int = 1
    mode: str = "min"
    save_optimizer_state: bool = True
    enable_async_checkpointing: bool = True
    class_name: str = "ModelCheckpoint"


class ModelCheckpoint(Callback):
    """Callback to save model parameters and mutable variables to the logging directory."""

    def __init__(self, config: ModelCheckpointConfig, trainer: Any, data_module: Any | None = None):
        super().__init__(config, trainer, data_module)
        self.log_dir = self.trainer.log_dir

        options = ocp.CheckpointManagerOptions(
            max_to_keep=self.config.save_top_k,
            best_fn=lambda m: m[self.config.monitor],
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
            directory=os.path.abspath(os.path.join(self.log_dir, "checkpoints/")),
            item_names=tuple(item_handlers.keys()),
            item_handlers=item_handlers,
            options=options,
        )

    def on_filtered_validation_epoch_end(self, eval_metrics, epoch_idx):
        self.save_model(eval_metrics, epoch_idx)

    def save_model(self, eval_metrics, epoch_idx):
        """Saves model parameters and batch statistics to the logging directory.

        Args:
            eval_metrics: Dictionary of evaluation metrics.
            epoch_idx: Index of the current epoch.
        """
        logging.info(f"Saving model at epoch {epoch_idx} with eval metrics {eval_metrics}.")
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
        self.manager.save(epoch_idx, args=save_items, metrics=eval_metrics)

    def load_model(self, epoch_idx=-1):
        """Loads model parameters and variables from the logging directory.

        Args:
            epoch_idx: Index of the epoch to load. If -1, loads the best epoch.

        Returns:
            Dictionary of loaded model parameters and additional variables.
        """
        logging.info(f"Loading model at epoch {epoch_idx}.")
        if epoch_idx == -1:
            epoch_idx = self.manager.best_step()
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
            epoch_idx,
            args=ocp.args.Composite(**args),
        )
        state_dict = {k: v for k, v in state_dict.items() if v is not None}
        return state_dict

    def finalize(self, status: str | None = None):
        logging.info("Closing checkpoint manager")
        self.manager.wait_until_finished()
        self.manager.close()
