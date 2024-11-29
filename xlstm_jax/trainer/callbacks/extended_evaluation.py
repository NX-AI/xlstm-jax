import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from flax import core, linen as nn, struct
from flax.core import FrozenDict
from jax import random
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

from xlstm_jax.common_types import HostMetrics, ImmutableMetrics, Metrics, PyTree, TrainState
from xlstm_jax.dataset.batch import Batch
from xlstm_jax.trainer.callbacks.callback import Callback, CallbackConfig
from xlstm_jax.trainer.data_module import DataloaderModule
from xlstm_jax.trainer.metrics import (
    aggregate_metrics,
    get_metrics,
    update_metrics,
)

LOGGER = logging.getLogger(__name__)


class EvalState(struct.PyTreeNode):
    """
    EvalState with additional mutable variables and RNG.

    Args:
        step: Counter starts at 0 and is incremented by every call to ``.apply_gradients()``.
        apply_fn: Usually set to ``model.apply()``. Kept in this dataclass for convenience to have a shorter params
            list for the ``train_step()`` function in your training loop.
        params: The parameters to be updated by ``tx`` and used by ``apply_fn``.
    """

    step: int | jax.Array
    rng: jax.Array
    apply_fn: Callable = struct.field(pytree_node=False)
    params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)

    # A simple extension of EvalState to also include mutable variables
    # like batch statistics. If a model has no mutable vars, it is None.
    mutable_variables: Any = None

    @classmethod
    def from_train_state(cls, *, train_state: TrainState) -> "EvalState":
        return cls(
            step=train_state.step,
            apply_fn=train_state.apply_fn,
            params=train_state.params,
            mutable_variables=train_state.mutable_variables,
            rng=train_state.rng,
        )

    @classmethod
    def create(cls, *, apply_fn: Callable, params: core.FrozenDict[str, Any], **kwargs) -> "EvalState":
        return cls(apply_fn=apply_fn, params=params, **kwargs)


@dataclass(kw_only=True, frozen=False)
class ExtendedEvaluationConfig(CallbackConfig):
    """
    Configuration for additional Evaluations callback.

    """

    def create(self, trainer: Any, data_module: DataloaderModule | None = None) -> "ExtendedEvaluation":
        """
        Creates an Evaluation callback.

        Args:
            trainer: Trainer object.
            data_module: Data module object.
        """
        return ExtendedEvaluation(config=self, trainer=trainer, data_module=data_module)


def device_metrics_aggregation(trainer: Any, metrics: Metrics) -> Metrics:
    """
    Aggregates metrics beyond a single scalar value and a count.

    Also include `single_noreduce` metrics by concatenation.

    Args:
        trainer: Trainer (for aggregation axes)
        metrics: the sharded metrics

    Returns:
        The reduced/gathered metrics.
    """
    res = {
        metric: {
            comp_name: (
                jax.tree.map(
                    lambda x: x
                    if isinstance(x, str)
                    else (
                        jax.lax.all_gather(
                            x,
                            axis_name=(
                                trainer.data_axis_name,
                                trainer.fsdp_axis_name,
                            ),
                            tiled=True,
                        )
                    ),
                    comp,
                )
                if (
                    "log_modes" in metric_components
                    and (
                        "single_noreduce_wcount" in metric_components["log_modes"]
                        or (comp_name == "value" and "single_noreduce" in metric_components["log_modes"])
                    )
                )
                else jax.tree.map(
                    lambda x: x
                    if isinstance(x, str)
                    else jax.lax.psum(
                        x,
                        axis_name=(
                            trainer.data_axis_name,
                            trainer.fsdp_axis_name,
                        ),
                    ),
                    comp,
                )
            )
            for comp_name, comp in metric_components.items()
        }
        for metric, metric_components in metrics.items()
    }
    return res


class ExtendedEvaluation(Callback):
    """
    Callback that runs additional evaluations.

    Args:
        config: The configuration for the Evaluation callback.
        trainer: Trainer
        data_module: :class:`DataloaderModule`, containing train/val/test data loaders.
    """

    def __init__(self, config: ExtendedEvaluationConfig, trainer: Any, data_module: DataloaderModule | None = None):
        super().__init__(config, trainer, data_module)
        self.config = config
        self.trainer = trainer
        self.exmp_batch = self.create_modified_exemplary_batch(self.trainer.exmp_batch)
        self.eval_step = None
        self.create_jitted_functions()
        self._eval_metric_shapes = None

    def create_modified_exemplary_batch(self, exmp_batch: Batch) -> Batch:
        """
        Create a modified exemplary batch for evaluation. Is useful for passing additional
        information / metadata to the batch for post-processing.

        Args:
            exmp_batch: "Original" training exemplary batch

        Returns:
            Modified exemplary batch for evaluation, might be the unmodified original.
        """
        return exmp_batch

    def eval_function(self, params: Any, apply_fn: Any, batch: Batch, rng: jax.Array) -> tuple[Metrics, PyTree]:
        """
        The extended evaluation function calculating metrics.

        This function needs to be overwritten by a subclass.

        Args:
            params: The model parameters.
            apply_fn: The apply function of the state.
            batch: The current batch.
            rng: The random number generator.

        Returns:
            A tuple of metrics and mutable variables.
        """
        del params, apply_fn, batch, rng
        raise NotImplementedError
        # return metrics

    def create_jitted_functions(self):
        """
        Create jitted version of the evaluation function.
        """
        eval_step = self.create_evaluation_step_function()
        if self.trainer.trainer_config.debug:  # Skip jitting
            LOGGER.info("Skipping jitting due to debug=True")
            self.eval_step = eval_step
        else:
            self.eval_step = jax.jit(eval_step, donate_argnames=["metrics"])  # Donate metrics to avoid copying

    def create_evaluation_step_function(
        self,
    ) -> Callable[[TrainState, Batch, ImmutableMetrics | None], ImmutableMetrics]:
        """
        Create and return a function for the extended evaluation step.

        The function takes as input the training state and a batch from the val/test loader.
        The function is expected to return a dictionary of logging metrics and a new train state.

        Returns:
            Step function calculating metrics for one batch.
        """

        def eval_step(state: TrainState, batch: Batch, metrics: ImmutableMetrics | None) -> ImmutableMetrics:
            # In our multi-host setup, each local device will have a different batch.
            # So we first gather the batch across model and pipeline axes.
            batch = jax.lax.all_gather(
                batch, axis_name=(self.trainer.model_axis_name, self.trainer.pipeline_axis_name), axis=0, tiled=True
            )
            # Forward pass and compute metrics.
            step_metrics, _ = self.eval_function(
                state.params,
                state.apply_fn,
                batch,
                random.PRNGKey(self.trainer.trainer_config.seed_eval),
            )
            with jax.named_scope("sync_extended_metrics"):
                step_metrics = device_metrics_aggregation(self.trainer, step_metrics)
            metrics = update_metrics(metrics, step_metrics, default_log_modes=["mean_nopostfix"])
            return metrics

        # Shard the evaluation function.
        state_partition_specs = nn.get_partition_spec(self.trainer.state)
        eval_step_fn = shard_map(
            eval_step,
            self.trainer.mesh,
            in_specs=(state_partition_specs, self.trainer.batch_partition_specs, P()),
            out_specs=P(),
            check_rep=False,
        )
        return eval_step_fn

    def init_eval_metrics(
        self, batch: Batch | None = None, alternative_eval_step: Callable | None = None
    ) -> FrozenDict:
        """
        Initialize the evaluation metrics with zeros.

        We infer the evaluation metric shape from the eval_step function. This is done to prevent a
        double-compilation of the eval_step function, where the first step has to be done with metrics None,
        and the next one with the metrics shape.

        Args:
            batch: An input to the model with which the shapes are inferred. If None, the :attr:`exmp_batch` is used.
            alternative_eval_step: An optional alternative eval step (not self.eval_step). This is needed if
                for a more complex eval step, that internally computes multiple steps (i.e. infinite eval in
                `lmeval_extended_evaluation.py`)

        Returns:
            A dictionary of metrics with the same shape as the eval metrics.
        """
        if not hasattr(self, "_eval_metric_shapes"):
            self._eval_metric_shapes = None
        if self._eval_metric_shapes is None:
            if batch is None:
                batch = self.exmp_batch
            self._eval_metric_shapes = jax.eval_shape(
                self.eval_step if alternative_eval_step is None else alternative_eval_step,
                self.trainer.state,
                batch,
                None,
            )
            LOGGER.info(f"Initialized eval metrics with keys {self._eval_metric_shapes.keys()}.")

        return jax.tree.map(lambda x: jnp.zeros_like(x), self._eval_metric_shapes)

    def aggregate_metrics(self, aggregated_metrics: HostMetrics, eval_metrics: ImmutableMetrics) -> HostMetrics:
        """
        Aggregate metrics over multiple batches.

        This is needed for "expensive" metrics that go beyond a scalar value and an accumulation count. These are then
        aggregated in CPU memory. The individual batch metrics might already be an actual aggregate for scalar values.


        Args:
            aggregated_metrics: Old aggregated metrics
            eval_metrics: Single batch metrics

        Returns:
            aggregated_metrics including the new batch

        """
        return aggregate_metrics(aggregated_metrics, eval_metrics)

    def eval_model(self, data_loader: Iterator, mode: str = "test", epoch_idx: int = 0) -> HostMetrics:
        """
        Evaluate the model on a dataset.

        Args:
            data_loader: Data loader of the dataset to evaluate on.
            mode: Whether 'val' or 'test'
            epoch_idx: Current epoch index.

        Returns:
            A dictionary of the evaluation metrics, averaged over data points in the dataset.
        """
        # Test model on all batches of a data loader and return metrics
        self.on_extended_evaluation_start()
        eval_metrics = self.init_eval_metrics()
        aggregated_metrics = {}
        step_count = 0
        for batch in self.trainer.tracker(data_loader, desc=mode.capitalize(), leave=False):
            eval_metrics = self.eval_step(self.trainer.state, batch, eval_metrics)
            aggregated_metrics = self.aggregate_metrics(aggregated_metrics, eval_metrics)
            step_count += 1
        if step_count == 0:
            LOGGER.warning(f"No batches in {mode} loader at epoch {epoch_idx}.")

        final_metrics = self.finalize_metrics(aggregated_metrics=aggregated_metrics)
        self.on_extended_evaluation_end(final_metrics)
        return final_metrics

    def finalize_metrics(self, aggregated_metrics: HostMetrics) -> HostMetrics:
        """
        Calculate final metrics from aggregated_metrics. (i,e, mean=sum/count)

        Args:
            aggregated_metrics: Aggregated metrics over the whole epoch

        Returns:
            Final metrics that are to be reported / logged.
        """
        return get_metrics(aggregated_metrics)[1]

    def on_extended_evaluation_start(self):
        """
        Callback for extended evaluation start.
        """

    def on_extended_evaluation_end(self, final_metrics: HostMetrics):
        """
        Callback for extended evaluation end with final_metrics attached.
        """
