from typing import Any
from functools import partial
import jax
import jax.numpy as jnp
import optax
import numpy as np
from flax import linen as nn
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from .xlstm_lm_model import xLSTMLMModelConfig, xLSTMLMModel
from collections import defaultdict
from tabulate import tabulate as python_tabulate
from flax.core.frozen_dict import FrozenDict

from distributed.data_parallel import fold_rng_over_axis, shard_module_params, sync_gradients
from distributed.pipeline_parallel import ModelParallelismWrapper, PipelineModule
from distributed.single_gpu import (
    Batch,
    TrainState,
    accumulate_gradients,
    get_num_params,
    print_metrics,
)
from distributed.tensor_parallel_transformer import (
    TPInputEmbedding,
    TPTransformerBlock,
    TPTransformerParallelBlock,
    TransformerBackbone,
    split_array_over_mesh,
)
from .utils import ParallelConfig

PyTree = Any
Parameter = jax.Array | nn.Partitioned
Metrics = dict[str, tuple[jax.Array, ...]]

def loss_fn(
    params: PyTree,
    apply_fn: Any,
    batch: Batch,
    rng: jax.Array,
    config: ParallelConfig,
) -> tuple[jax.Array, dict[str, Any]]:
    # Since dropout masks vary across the batch dimension, we want each device to generate a
    # different mask. We can achieve this by folding the rng over the data axis, so that each
    # device gets a different rng and thus mask.
    dropout_rng = fold_rng_over_axis(
        rng, (config.data_axis_name, config.pipeline_axis_name, config.model_axis_name)
    )
    # Remaining computation is the same as before for single device.
    logits = apply_fn(
        {"params": params},
        batch.inputs,
        train=True,
        rngs={"dropout": dropout_rng},
    )
    # Select the labels per device.
    labels = batch.labels
    labels = split_array_over_mesh(labels, axis_name=config.pipeline_axis_name, split_axis=1)
    labels = split_array_over_mesh(labels, axis_name=config.model_axis_name, split_axis=1)
    assert (
        logits.shape[:-1] == labels.shape
    ), f"Logits and labels shapes do not match: {logits.shape} vs {labels.shape}"
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    correct_pred = jnp.equal(jnp.argmax(logits, axis=-1), labels)
    batch_size = np.prod(labels.shape)
    # Collect metrics and return loss.
    step_metrics = {
        "loss": (loss.sum(), batch_size),
        "accuracy": (correct_pred.sum(), batch_size),
    }
    loss = loss.mean()
    return loss, step_metrics


def train_step(
    state: TrainState,
    metrics: Metrics | None,
    batch: Batch,
    config: ParallelConfig,
    gradient_accumulate_steps: int = 1
) -> tuple[TrainState, Metrics]:
    rng, step_rng = jax.random.split(state.rng)
    grads, step_metrics = accumulate_gradients(
        state,
        batch,
        step_rng,
        gradient_accumulate_steps,
        loss_fn=partial(loss_fn, config=config),
    )
    # Update parameters. We need to sync the gradients across devices before updating.
    with jax.named_scope("sync_gradients"):
        grads = sync_gradients(
            grads, (config.data_axis_name, config.pipeline_axis_name, config.model_axis_name)
        )
    new_state = state.apply_gradients(grads=grads, rng=rng)
    # Sum metrics across replicas. Alternatively, we could keep the metrics separate
    # and only synchronize them before logging. For simplicity, we sum them here.
    with jax.named_scope("sync_metrics"):
        step_metrics = jax.tree.map(
            lambda x: jax.lax.psum(
                x,
                axis_name=(
                    config.data_axis_name,
                    config.pipeline_axis_name,
                    config.model_axis_name,
                ),
            ),
            step_metrics,
        )
    if metrics is None:
        metrics = step_metrics
    else:
        metrics = jax.tree.map(jnp.add, metrics, step_metrics)
    return new_state, metrics


def get_train_step_fn(
    state: TrainState,
    batch: Batch,
    mesh: Mesh,
    config: ParallelConfig,
    gradient_accumulate_steps: int = 1,
) -> tuple[callable, PyTree]:
    state_specs = nn.get_partition_spec(state)
    train_step_fn = jax.jit(
        shard_map(
            partial(train_step, config=config, gradient_accumulate_steps=gradient_accumulate_steps),
            mesh,
            in_specs=(state_specs, P(), P(config.data_axis_name)),
            out_specs=(state_specs, P()),
            check_rep=False,
        ),
        donate_argnames=("state", "metrics"),
    )
    _, metric_shapes = jax.eval_shape(
        train_step_fn,
        state,
        None,
        batch,
    )
    metrics = jax.tree.map(lambda x: jnp.zeros(x.shape, dtype=x.dtype), metric_shapes)
    return train_step_fn, metrics


def init_xlstm(
    config: xLSTMLMModelConfig, 
    mesh: Mesh, 
    rng: jax.Array, 
    input_array: jax.Array,
    optimizer: callable,
):
    model = xLSTMLMModel(config)

    def _init_model(init_rng: jax.random.PRNGKey, x: jax.Array) -> TrainState:
        param_rng, init_rng = jax.random.split(init_rng)
        variables = model.init({"params": param_rng}, x, train=False)
        params = variables.pop("params")
        state = TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=optimizer,
            rng=init_rng,
        )
        return state
    init_model_fn = jax.jit(
        shard_map(
            _init_model,
            mesh,
            in_specs=(P(), P(config.parallel.data_axis_name)),
            out_specs=P(),
            check_rep=False,
        ),
    )
    state_xlstm_shapes = jax.eval_shape(
        init_model_fn, rng, input_array
    )
    state_xlstm_specs = nn.get_partition_spec(state_xlstm_shapes)

    init_model_fn = jax.jit(
        shard_map(
            _init_model,
            mesh,
            in_specs=(P(), P(config.parallel.data_axis_name)),
            out_specs=state_xlstm_specs,
            check_rep=False,
        ),
    )
    state_xlstm = init_model_fn(rng, input_array)
    print(f"Number of parameters: {get_num_params(state_xlstm):_}")
    print(tabulate_params(state_xlstm))
    return state_xlstm

def flatten_dict(d: dict) -> dict:
    """Flattens a nested dictionary."""
    flat_dict = {}
    for k, v in d.items():
        if isinstance(v, (dict, FrozenDict)):
            flat_dict.update({f"{k}.{k2}": v2 for k2, v2 in flatten_dict(v).items()})
        else:
            flat_dict[k] = v
    return flat_dict

def tabulate_params(state: TrainState) -> str:
    """Prints a summary of the parameters represented as table.

    Args:
        exmp_input: An input to the model with which the shapes are inferred.
    """
    params = state.params
    params = flatten_dict(params)
    param_shape = jax.tree.map(lambda x: x.shape, params)
    param_count = jax.tree.map(lambda x: int(np.prod(x.shape)), params)
    param_dtype = jax.tree.map(lambda x: str(x.dtype), params)
    param_sharding = jax.tree.map(lambda x: str(x.sharding), params)
    # param_mean = jax.tree.map(lambda x: jnp.mean(x).item(), params)
    # param_std = jax.tree.map(lambda x: jnp.std(x).item(), params)
    # param_min = jax.tree.map(lambda x: jnp.min(x).item() if x.size > 0 else 0, params)
    # param_max = jax.tree.map(lambda x: jnp.max(x).item() if x.size > 0 else 0, params)
    summary = defaultdict(list)
    for key in sorted(list(params.keys())):
        summary["Name"].append(key)
        summary["Shape"].append(param_shape[key])
        summary["Count"].append(param_count[key])
        summary["Dtype"].append(param_dtype[key])
        summary["Sharding"].append(param_sharding[key])
        # summary["Mean"].append(param_mean[key])
        # summary["Std"].append(param_std[key])
        # summary["Min"].append(param_min[key])
        # summary["Max"].append(param_max[key])
    return python_tabulate(summary, headers="keys", intfmt="_", floatfmt=".3f")
