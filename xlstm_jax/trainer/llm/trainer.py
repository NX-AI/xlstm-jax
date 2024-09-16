from typing import Any, Tuple

from xlstm_jax.distributed.array_utils import fold_rng_over_axis, split_array_over_mesh
from xlstm_jax.distributed.single_gpu import Batch
from xlstm_jax.trainer.base.train_state import TrainState
from xlstm_jax.trainer.base.trainer import TrainerModule
from xlstm_jax.trainer.metrics import Metrics

import jax
import jax.numpy as jnp
import numpy as np
import optax


class LLMTrainer(TrainerModule):
    def loss_function(
        self, params: Any, apply_fn: Any, batch: Batch, rng: jax.Array, train: bool = True
    ) -> tuple[jax.Array, Metrics]:
        """Calculate the loss function for the model.

        Args:
            params (Any): The model parameters.
            apply_fn (Any): The model apply function.
            batch (Batch): The input batch.
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
        # Select the labels per device.
        labels = batch.labels
        labels = split_array_over_mesh(labels, axis_name=self.pipeline_axis_name, split_axis=1)
        labels = split_array_over_mesh(labels, axis_name=self.model_axis_name, split_axis=1)
        assert (
            logits.shape[:-1] == labels.shape
        ), f"Logits and labels shapes do not match: {logits.shape} vs {labels.shape}"
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        correct_pred = jnp.equal(jnp.argmax(logits, axis=-1), labels)
        batch_size = np.prod(labels.shape)
        perplexity = jnp.exp(loss.mean())
        # Collect metrics and return loss.
        step_metrics = {
            "loss": {"value": loss.sum(), "count": batch_size},
            "accuracy": {"value": correct_pred.sum(), "count": batch_size},
            "perplexity": {"value": perplexity, "count": 1},
        }
        loss = loss.mean()
        return loss, step_metrics
