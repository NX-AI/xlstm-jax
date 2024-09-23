from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from xlstm_jax.dataset import LLMBatch
from xlstm_jax.distributed import fold_rng_over_axis, split_array_over_mesh
from xlstm_jax.trainer.base.trainer import TrainerModule
from xlstm_jax.trainer.metrics import HostMetrics, Metrics


class LLMTrainer(TrainerModule):
    """Trainer for Autoregressive Language Models."""

    def loss_function(
        self, params: Any, apply_fn: Any, batch: LLMBatch, rng: jax.Array, train: bool = True
    ) -> tuple[jax.Array, Metrics]:
        """
        Calculate the loss function for the model.

        Args:
            params (Any): The model parameters.
            apply_fn (Any): The model apply function.
            batch (LLMBatch): The input batch.
            rng (jax.Array): The random number generator.
            train (bool, optional): Whether the model is in training mode. Defaults to True.

        Returns:
            Tuple[jax.Array, Metrics]: The loss and the metrics.
        """
        # Since dropout masks vary across the batch dimension, we want each device to generate a
        # different mask. We can achieve this by folding the rng over the data axis, so that each
        # device gets a different rng and thus mask.
        dropout_rng = fold_rng_over_axis(
            rng, (self.data_axis_name, self.fsdp_axis_name, self.pipeline_axis_name, self.model_axis_name)
        )
        # Remaining computation is the same as before for single device.
        logits = apply_fn(
            {"params": params},
            batch.inputs,
            train=train,
            rngs={"dropout": dropout_rng},
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
        return avg_loss, step_metrics

    def get_metric_postprocess_fn(self) -> Callable[[HostMetrics], HostMetrics]:
        """
        Get function to post-process metrics with on host.

        Will be passed to logger. Adds perplexity to the metrics.

        Returns:
            Callable[[HostMetrics], HostMetrics]: The postprocess metric function.
        """

        def _postprocess_metrics(metrics: HostMetrics) -> HostMetrics:
            """Add perplexity to the metrics."""
            for key in ["loss", "loss_mean", "loss_single"]:
                if key in metrics:
                    metrics[f"perplexity{key[4:]}"] = np.exp(metrics[key])
            return metrics

        return _postprocess_metrics
