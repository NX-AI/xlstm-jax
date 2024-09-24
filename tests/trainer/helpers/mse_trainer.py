from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn

from xlstm_jax.dataset import Batch
from xlstm_jax.distributed import ModelParallelismWrapper, TPDense, shard_module_params, split_array_over_mesh
from xlstm_jax.models import ModelConfig
from xlstm_jax.trainer.base.trainer import TrainerModule
from xlstm_jax.trainer.metrics import Metrics

PyTree = Any


class ToyModel(nn.Module):
    """
    Toy model for testing purposes.

    Contains one TP, one FSDP+TP, and one pure FSDP layer. Has no real use case and solely for testing purposes.
    """

    config: ModelConfig
    out_features: int = 1

    @nn.compact
    def __call__(self, x: jax.Array, train: bool = False, **kwargs) -> jax.Array:
        """Forward pass of the model."""
        tp_size = jax.lax.psum(1, self.config.parallel.model_axis_name)
        # Input layer with TP. All devices share the same input already (hence skip_communication),
        # but each will have a different output. We split the output features over the TP axis.
        x = TPDense(
            partial(nn.Dense, features=64 // tp_size),
            model_axis_name=self.config.parallel.model_axis_name,
            tp_mode="gather",
            skip_communication=True,
            name="in",
        )(x)
        # Example LayerNorm with model parallelism. Uses the model axis for parameter sharding and reduction of
        # statistics.
        x = ModelParallelismWrapper(
            module_fn=partial(nn.LayerNorm, axis_name=self.config.parallel.model_axis_name),
            model_axis_name=self.config.parallel.model_axis_name,
        )(x)
        x = nn.swish(x)
        # Intermediate layer with FSDP+TP. Each device has a different input and need to gather first, and the output
        # is split over the TP axis.
        dense_fn = partial(
            TPDense,
            dense_fn=partial(nn.Dense, features=64 // tp_size),
            model_axis_name=self.config.parallel.model_axis_name,
            tp_mode="gather",
            skip_communication=False,
        )
        dense_fn = shard_module_params(
            dense_fn,
            axis_name=self.config.parallel.fsdp_axis_name,
            min_weight_size=0,
        )
        x = dense_fn(name="middle")(x)
        x = nn.swish(x)
        # Intermediate features for testing logging.
        self.sow("intermediates", "last_activations", x.std())
        # For the output layer, we only use FSDP. We first gather all inputs, split them over the model and pipeline
        # axis over the batch dimension. Then, we apply the layer and calculate the outputs.
        x = jax.lax.all_gather(x, axis_name=self.config.parallel.model_axis_name, axis=-1, tiled=True)
        x = split_array_over_mesh(x, axis_name=self.config.parallel.pipeline_axis_name, split_axis=0)
        x = split_array_over_mesh(x, axis_name=self.config.parallel.model_axis_name, split_axis=0)
        x = shard_module_params(
            partial(nn.Dense, features=self.out_features),
            self.config.parallel.fsdp_axis_name,
            min_weight_size=self.config.parallel.fsdp_min_weight_size,
        )(name="out")(x)
        return x


class MSETrainer(TrainerModule):
    def loss_function(
        self, params: Any, apply_fn: Any, batch: Batch, rng: jax.Array, train: bool = True
    ) -> tuple[jax.Array, tuple[Metrics, PyTree]]:
        """Loss function to calculate gradients and metrics."""
        # Apply the model to the inputs. This is where the forward pass happens.
        preds, mutable_vars = apply_fn(
            {"params": params},
            batch.inputs,
            train=train,
            mutable="intermediates",
        )
        # Select the labels per device. This is only needed if we want to split the output over the mesh.
        # Otherwise, the TP and PP devices may share the same outputs.
        labels = batch.targets
        labels = split_array_over_mesh(labels, axis_name=self.pipeline_axis_name, split_axis=0)
        labels = split_array_over_mesh(labels, axis_name=self.model_axis_name, split_axis=0)
        assert (
            preds.shape == labels.shape
        ), f"Predictions and labels shapes do not match: {preds.shape} vs {labels.shape}"
        # Compute the loss and metrics.
        loss = optax.l2_loss(preds, labels)
        batch_size = np.prod(labels.shape)
        l1_dist = jnp.abs(preds - labels)
        # Collect metrics and return loss.
        step_metrics = {
            "loss": {"value": loss.sum(), "count": batch_size},
            "l1_dist": {"value": l1_dist.sum(), "count": batch_size},
        }
        loss = loss.mean()
        return loss, (step_metrics, mutable_vars)
