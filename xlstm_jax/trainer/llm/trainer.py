from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

from xlstm_jax.dataset import LLMBatch
from xlstm_jax.distributed import fold_rng_over_axis, split_array_over_mesh
from xlstm_jax.trainer.base.trainer import TrainerConfig, TrainerModule, TrainState
from xlstm_jax.trainer.metrics import HostMetrics, ImmutableMetrics, Metrics, update_metrics

from .sampling import generate_tokens, temperature_sampling

PyTree = Any


@dataclass(kw_only=True, frozen=False)
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
        self,
        params: Any,
        apply_fn: Any,
        batch: LLMBatch,
        rng: jax.Array,
        train: bool = True,
        mutable_variables: dict[str, Any] | None = None,
    ) -> tuple[jax.Array, tuple[Metrics, PyTree]]:
        """
        Calculate the loss function for the model.

        Args:
            params: The model parameters.
            apply_fn: The model apply function.
            batch: The input batch.
            rng: The random number generator.
            train: Whether the model is in training mode. Defaults to True.
            mutable_variables: Additional mutable variables which are passed to the model apply function and set to
                mutable. Defaults to None.

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
        if mutable_variables is None:
            mutable_variables = {}
        logits, mutable_variables = apply_fn(
            {"params": params, **mutable_variables},
            batch.inputs,
            pos_idx=batch.inputs_position,
            document_borders=batch.get_document_borders(),
            train=train,
            rngs={"dropout": dropout_rng},
            mutable=["intermediates"] + list(mutable_variables.keys()),
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
        # For training, we log additional metrics.
        if train:
            # Token and document statistics.
            step_metrics["token_utilization"] = {
                "value": num_targets,
                "count": targets_mask.size,
                "log_modes": ["mean"],
            }
            step_metrics["tokens_per_batch"] = {
                "value": num_targets,
                "count": 1.0 / self.mesh.size,  # Local has 1 / mesh size of the global batch.
                "log_modes": ["mean"],
            }
            step_metrics["num_docs_per_batch"] = {
                "value": jnp.max(batch.targets_segmentation, axis=-1).sum(),
                "count": 1.0 / self.mesh.size,  # Local has 1 / mesh size of the global batch.
                "log_modes": ["mean"],
            }
            # If enabled, log the norm, std, and max of the logits.
            if self.trainer_config.log_logit_stats:
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
            metric_keys = list(metrics.keys())
            for key in metric_keys:
                for key_postfix in ["loss", "loss_mean", "loss_single"]:
                    if key.endswith(key_postfix):
                        key_perplexity = f"{key[:-len(key_postfix)]}perplexity{key_postfix[4:]}"
                        metrics[key_perplexity] = np.exp(metrics[key]).item()
                        break
            return metrics

        return _postprocess_metrics

    def get_generate_fn(
        self,
        max_length: int = 2048,
        eod_token_id: int = -1,
        token_sample_fn: Callable[[jax.Array, jax.Array], jax.Array] = temperature_sampling,
        gather_params_once: bool = False,
    ) -> Callable[[TrainState, jax.Array, jax.Array, jax.Array | None, PyTree | None], tuple[jax.Array, jax.Array]]:
        """
        Create a function to generate text from the model.

        Args:
            max_length: The maximum length of the generated text. Defaults to 2048.
            eod_token_id: The end-of-document token id. If all sequences hit this token, generation will stop. Defaults
                to -1, ie will not have an effect.
            token_sample_fn: The token sampler to use for sampling tokens. Defaults to temperature sampling with
                temperature 1.0.
            gather_params_once: Whether to gather fsdp-sharded parameters once before generating. This reduces
                communication overhead between devices, but requires the model to fit on a single device (up to TP
                parallelism). Defaults to false.

        Returns:
            The generate function. Takes as input the state, an RNG key, prefix tokens, prefix mask, and an optional
            cache. It returns the generated tokens and a mask for valid tokens, including the prefix given.
        """
        if self.mesh.shape[self.model_axis_name] > 1:
            raise ValueError("Generation is only supported for models with a single model axis at the moment.")

        # Set static arguments of the generate_tokens function.
        _generate_fn = partial(
            generate_tokens,
            max_length=max_length,
            eod_token_id=eod_token_id,
            token_sample_fn=token_sample_fn,
            gather_params_once=gather_params_once,
            data_axis_name=self.data_axis_name,
            fsdp_axis_name=self.fsdp_axis_name,
        )

        # Shard the generate function.
        state_partition_specs = nn.get_partition_spec(self.state)
        token_partition_specs = P((self.data_axis_name, self.fsdp_axis_name))
        generate_fn = shard_map(
            _generate_fn,
            self.mesh,
            in_specs=(state_partition_specs, P(), token_partition_specs, token_partition_specs),
            out_specs=(token_partition_specs, token_partition_specs),
            check_rep=False,
        )
        generate_fn = jax.jit(generate_fn)
        return generate_fn

    def get_cache_shape_dtype_struct(self, exmp_batch: LLMBatch | None = None) -> PyTree:
        """
        Get the shape, dtype, and structure of the cache.

        Args:
            exmp_batch: An example batch to determine the shape of the cache. Defaults to None, in which case the
                example batch from the trainer is used.

        Returns:
            A PyTree with leafs being ShapeDtypeStruct of the cache elements.
        """
        if exmp_batch is None:
            exmp_batch = self.exmp_batch

        def _init_cache(state: TrainState, batch: LLMBatch) -> PyTree:
            # In our multi-host setup, each local device will have a different batch.
            # So we first gather the batch across model and pipeline axes.
            batch = jax.lax.all_gather(
                batch, axis_name=(self.model_axis_name, self.pipeline_axis_name), axis=0, tiled=True
            )
            _, mutable_variables = state.apply_fn(
                {"params": state.params},
                batch.inputs,
                pos_idx=batch.inputs_position,
                document_borders=batch.get_document_borders(),
                train=False,
                mutable="cache",
            )
            return mutable_variables.get("cache", {})

        state_partition_specs = nn.get_partition_spec(self.state)
        init_cache_fn = shard_map(
            _init_cache,
            self.mesh,
            in_specs=(state_partition_specs, self.batch_partition_specs),
            out_specs=P(),
            check_rep=False,
        )
        cache_shape = jax.eval_shape(init_cache_fn, self.state, exmp_batch)
        return cache_shape

    def create_recurrent_evaluation_step_function(
        self,
        chunk_size: int,
        exmp_batch: LLMBatch | None = None,
        cache_init_fn: Callable[[PyTree], PyTree] | None = None,
    ) -> Callable[[TrainState, LLMBatch, ImmutableMetrics | None], ImmutableMetrics]:
        """
        Create and return a function for the evaluation step.

        Compared to the `create_evaluation_step_function`, this evaluation supports much longer sequences by chunking
        the input and running the model recurrently over the chunks. This is useful for evaluation on long documents.
        This is enabled by keeping a cache, which is forwarded between evaluation steps.

        Note: do *not* jit this function if you want to support arbitrary input shapes. This function jit's the
        recurrent function for a single chunk, and adds a python loop around it to handle arbitrary length sequences.
        Thus, no outer jit is needed.

        Note: this function is explicitly meant for recurrent models like xLSTM. Using this function on a non-recurrent
        model will lead to unexpected, incorrect results.

        Args:
            chunk_size: Size of the chunks to split the input into. The slices are performed over the sequence length.
            exmp_batch: An example batch to determine the shape of the cache. Defaults to None, in which case the
                example batch from the trainer is used.
            cache_init_fn: A function to initialize the cache. If not provided, the cache is initialized with zeros.
                The function should take the shape dtype struct of the cache as input and return the initialized cache.

        Returns:
            The evaluation step function with support for arbitrary length sequences.
        """
        if cache_init_fn is None:
            cache_init_fn = partial(jax.tree.map, jnp.zeros_like)

        def rec_eval_step(
            state: TrainState, cache: PyTree, batch: LLMBatch, metrics: ImmutableMetrics | None
        ) -> tuple[PyTree, ImmutableMetrics]:
            # In our multi-host setup, each local device will have a different batch.
            # So we first gather the batch across model and pipeline axes.
            batch = jax.lax.all_gather(
                batch, axis_name=(self.model_axis_name, self.pipeline_axis_name), axis=0, tiled=True
            )
            # Forward pass and compute metrics.
            _, (step_metrics, mutable_variables) = self.loss_function(
                state.params,
                state.apply_fn,
                batch,
                jax.random.PRNGKey(self.trainer_config.seed_eval),
                train=False,
                mutable_variables={"cache": cache},
            )
            cache = mutable_variables["cache"]
            with jax.named_scope("sync_metrics"):
                step_metrics = jax.tree.map(
                    lambda x: jax.lax.psum(
                        x,
                        axis_name=(
                            self.data_axis_name,
                            self.fsdp_axis_name,
                            self.pipeline_axis_name,
                            self.model_axis_name,
                        ),
                    )
                    if not isinstance(x, str)
                    else x,
                    step_metrics,
                )
            metrics = update_metrics(metrics, step_metrics, default_log_modes=("mean_nopostfix",))
            return cache, metrics

        # Determine cache structure.
        cache_shape = self.get_cache_shape_dtype_struct(exmp_batch=exmp_batch)

        # Shard the single-step recurrent evaluation function.
        state_partition_specs = nn.get_partition_spec(self.state)
        cache_partition_specs = nn.get_partition_spec(cache_shape)
        rec_eval_step_fn = shard_map(
            rec_eval_step,
            self.mesh,
            in_specs=(state_partition_specs, cache_partition_specs, self.batch_partition_specs, P()),
            out_specs=P(),
            check_rep=False,
        )

        # Jit the step function. We donate also the cache to free up memory.
        rec_eval_step_fn = jax.jit(
            rec_eval_step_fn,
            donate_argnames=["cache", "metrics"],
        )

        # Create the looped evaluation step function.
        def looped_eval_step_fn(
            state: TrainState, batch: LLMBatch, metrics: ImmutableMetrics | None
        ) -> ImmutableMetrics:
            # Create initial cache.
            cache = cache_init_fn(cache_shape)
            # Run the evaluation per chunk with forwarding the cache.
            num_chunks = (batch.inputs.shape[1] - 1) // chunk_size + 1
            for i in range(num_chunks):
                # Get current chunk of batch.
                chunk_start = i * chunk_size
                chunk_end = min((i + 1) * chunk_size, batch.inputs.shape[1])
                chunk_batch = batch[:, chunk_start:chunk_end]
                if chunk_batch.inputs.shape[1] < chunk_size:
                    # Pad batch.
                    pad_size = chunk_size - chunk_batch.inputs.shape[1]
                    chunk_batch = jax.tree.map(
                        lambda x: jnp.pad(x, ((0, 0), (0, pad_size)), mode="constant"), chunk_batch
                    )
                # Run the evaluation step.
                cache, metrics = rec_eval_step_fn(state, cache, chunk_batch, metrics)
            # Return final metrics.
            return metrics

        return looped_eval_step_fn
