#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import functools
from collections.abc import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp


def execute_pipeline_step(
    module: nn.Module,
    state: jax.Array,
    input: jax.Array,
    *args,
    model_axis_name: str,
    **kwargs,
) -> tuple[jax.Array, jax.Array]:
    """
    Single micro-batch pipeline step.

    Args:
        module: Flax module representing the stage to execute.
        state: Last communicated features between stages. Used as input to the module for all stages except the first.
        input: Original micro-batch input to the pipeline stage. Used as input to the module for the first stage.
        *args: Additional arguments to the module.
        model_axis_name: Name of the model axis in the mesh/shard_map.
        **kwargs: Additional keyword arguments to the module.

    Returns:
        Tuple of the new state (after communication) and the output of the module.
    """
    num_stages = jax.lax.psum(1, model_axis_name)
    stage_index = jax.lax.axis_index(model_axis_name)
    # For the first stage, we use the microbatches as input.
    # For all other stages, we use the last state from the
    # previous stage as input.
    state = jnp.where(stage_index == 0, input, state)
    state = module(state, *args, **kwargs)
    # For the last stage, we return the state as output.
    # For all other stages, we return zeros.
    output = jnp.where(
        stage_index == num_stages - 1,
        state,
        jnp.zeros_like(state),
    )
    # Communicate the last state to the next stage.
    state = jax.lax.ppermute(
        state,
        model_axis_name,
        perm=[(i, (i + 1) % num_stages) for i in range(num_stages)],
    )
    return state, output


@jax.named_scope("pipeline")  # Naming scope for profiling.
def execute_pipeline(
    module: nn.Module,
    x: jax.Array,
    *args,
    num_microbatches: int,
    model_axis_name: str,
    **kwargs,
) -> jax.Array:
    """
    Execute a pipeline of stages on a batch of data.

    Uses the principle of GPipe in splitting the batch into micro-batches
    and running the pipeline stages in parallel.

    Args:
        module: Flax module representing the pipeline stage to execute.
        x: Batch of input data, only needed on device of the first stage. Data will be split into micro-batches.
        *args: Additional arguments to the module.
        num_microbatches: Number of micro-batches to split the batch into.
        model_axis_name: Name of the model axis in the mesh/shard_map.
        **kwargs: Additional keyword arguments to the module.

    Returns:
        Output of the last stage of the pipeline. For devices that are not
        the last stage, the output is zeros.
    """
    num_stages = jax.lax.psum(1, model_axis_name)
    # Structure the input data into micro-batches.
    batch_size = x.shape[0]
    assert (
        batch_size % num_microbatches == 0
    ), f"Batch size {batch_size} must be divisible by number of microbatches {num_microbatches}"
    microbatch_size = batch_size // num_microbatches
    microbatches = jnp.reshape(x, (num_microbatches, microbatch_size, *x.shape[1:]))
    inputs = jnp.concatenate(  # Add zeros for unused computation blocks in first stage.
        [
            microbatches,
            jnp.zeros((num_stages - 1, *microbatches.shape[1:]), dtype=microbatches.dtype),
        ],
        axis=0,
    )
    state = jnp.zeros_like(microbatches[0])
    num_iterations = inputs.shape[0]
    # Run loop over pipeline steps.
    _, outputs = nn.scan(
        functools.partial(
            execute_pipeline_step,
            *args,
            model_axis_name=model_axis_name,
            **kwargs,
        ),
        variable_broadcast={"params": True},
        split_rngs={"params": False, "dropout": True},
        length=num_iterations,
        in_axes=0,
        out_axes=0,
    )(module, state, inputs)
    # Take last N outputs (first ones are zeros from unused computation blocks in last stage).
    outputs = jnp.concatenate(outputs[-num_microbatches:], axis=0)
    return outputs


class PipelineModule(nn.Module):
    """
    Module wrapper for executing a pipeline of stages.

    This module is used to wrap a stage of a pipeline to execute in pipeline parallelism.

    Args:
        model_axis_name: Name of the model axis in the mesh/shard_map.
        num_microbatches: Number of micro-batches to split the batch into.
        module_fn: Function that returns the module to execute in the pipeline.
    """

    model_axis_name: str
    num_microbatches: int
    module_fn: Callable[..., nn.Module]

    @nn.compact
    def __call__(self, *args, **kwargs):
        module = self.module_fn()
        return execute_pipeline(
            module,
            *args,
            **kwargs,
            num_microbatches=self.num_microbatches,
            model_axis_name=self.model_axis_name,
        )
