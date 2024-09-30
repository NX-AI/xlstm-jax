from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from xlstm_jax.dataset import LLMBatch
from xlstm_jax.distributed import fold_rng_over_axis, split_array_over_mesh
from xlstm_jax.trainer.base.trainer import TrainerConfig, TrainerModule
from xlstm_jax.trainer.metrics import HostMetrics, Metrics

PyTree = Any


@dataclass(kw_only=True, frozen=True)
class LLMTrainerConfig(TrainerConfig):
    """
    Configuration for the LLM trainer.

    See TrainerConfig for inherited attributes.

    Attributes:
        log_logit_stats: Whether to log statistics of the logits during training. Note that for large vocabularies, this
            can be expensive.
    """

    log_logit_stats: bool = False


class LLMTrainer(TrainerModule):
    """Trainer for Autoregressive Language Models."""

    def loss_function(
        self, params: Any, apply_fn: Any, batch: LLMBatch, rng: jax.Array, train: bool = True
    ) -> tuple[jax.Array, tuple[Metrics, PyTree]]:
        """
        Calculate the loss function for the model.

        Args:
            params: The model parameters.
            apply_fn: The model apply function.
            batch: The input batch.
            rng: The random number generator.
            train: Whether the model is in training mode. Defaults to True.

        Returns:
            The loss and a tuple of metrics and mutable variables.
        """
        # Since dropout masks vary across the batch dimension, we want each device to generate a
        # different mask. We can achieve this by folding the rng over the data axis, so that each
        # device gets a different rng and thus mask.
        dropout_rng = fold_rng_over_axis(
            rng, (self.data_axis_name, self.fsdp_axis_name, self.pipeline_axis_name, self.model_axis_name)
        )
        # Remaining computation is the same as before for single device.
        logits, mutable_variables = apply_fn(
            {"params": params},
            batch.inputs,
            train=train,
            rngs={"dropout": dropout_rng},
            mutable="intermediates",
        )
        # Select the targets per device.
        targets = batch.targets
        targets = split_array_over_mesh(targets, axis_name=self.pipeline_axis_name, split_axis=1)
        targets = split_array_over_mesh(targets, axis_name=self.model_axis_name, split_axis=1)
        assert (
            logits.shape[:-1] == targets.shape
        ), f"Logits and targets shapes do not match: {logits.shape} vs {targets.shape}"
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        correct_pred = jnp.equal(jnp.argmax(logits, axis=-1), targets)
        # Mask out padding tokens.
        targets_mask = batch.targets_segmentation != 0
        targets_mask = split_array_over_mesh(targets_mask, axis_name=self.pipeline_axis_name, split_axis=1)
        targets_mask = split_array_over_mesh(targets_mask, axis_name=self.model_axis_name, split_axis=1)
        loss = loss * targets_mask
        correct_pred = correct_pred * targets_mask
        num_targets = targets_mask.sum()
        avg_loss = jnp.where(num_targets > 0, loss.sum() / num_targets, 0.0)
        # Collect metrics and return loss.
        step_metrics = {
            "loss": {"value": loss.sum(), "count": num_targets},
            "accuracy": {"value": correct_pred.sum(), "count": num_targets},
        }
        # For training, we also log the norm, std, and max of the logits.
        if train and self.trainer_config.log_logit_stats:
            logits_norm = jnp.linalg.norm(logits, axis=-1)
            logits_std = jnp.std(logits, axis=-1)
            logits_max = jnp.max(logits)
            step_metrics.update(
                {
                    "logits_norm": {
                        "value": (logits_norm * targets_mask).sum(),
                        "count": num_targets,
                        "log_modes": ["mean"],
                    },
                    "logits_std": {
                        "value": (logits_std * targets_mask).sum(),
                        "count": num_targets,
                        "log_modes": ["mean"],
                    },
                    "logits_max": {"value": logits_max, "count": 1, "log_modes": ["max"]},
                }
            )
        return avg_loss, (step_metrics, mutable_variables)

    def get_metric_postprocess_fn(self) -> Callable[[HostMetrics], HostMetrics]:
        """
        Get function to post-process metrics with on host.

        Will be passed to logger. Adds perplexity to the metrics.

        Returns:
            The postprocess metric function.
        """

        def _postprocess_metrics(metrics: HostMetrics) -> HostMetrics:
            """Add perplexity to the metrics."""
            for key in ["loss", "loss_mean", "loss_single"]:
                if key in metrics:
                    metrics[f"perplexity{key[4:]}"] = np.exp(metrics[key])
            return metrics

        return _postprocess_metrics
