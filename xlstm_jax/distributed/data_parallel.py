import functools
from collections.abc import Callable, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from absl import logging
from jax import lax

from xlstm_jax.common_types import Parameter, PyTree


@jax.named_scope("shard_params")
def shard_params(params: PyTree, axis_name: str, min_weight_size: int = 2**18) -> PyTree:
    """
    Shard parameters across the given mesh axis.

    Args:
        params: The parameters to shard.
        axis_name: The axis to shard parameters across.
        min_weight_size: The minimum size of a parameter to shard. Parameters with fewer values will not be sharded.

    Returns:
        PyTree of same structure as params, but with leaves sharded over new axis if possible.
    """
    axis_idx = jax.lax.axis_index(axis_name)
    axis_size = jax.lax.psum(1, axis_name)

    def _split(x: Parameter) -> Parameter:
        if isinstance(x, nn.Partitioned):
            value, names = x.value, x.names
        else:
            value = x
            names = (None,) * value.ndim

        if axis_name in names:
            logging.warning(f"Parameter {value.shape} with names {names} already sharded on axis {axis_name}.")
            return x
        if value.size <= min_weight_size:
            logging.info(
                f"Parameter {value.shape} with names {names} too small to shard, size {value.size} < {min_weight_size}."
            )
            return x

        shape = value.shape
        idx = np.argsort(shape)[::-1]  # Shard along largest possible axis.
        for i in idx:
            if shape[i] % axis_size == 0 and names[i] is None:
                split_size = shape[i] // axis_size
                p_sharded = nn.Partitioned(
                    value=lax.dynamic_slice_in_dim(  # Shard to keep on present device.
                        value, axis_idx * split_size, split_size, axis=i
                    ),
                    names=names[:i] + (axis_name,) + names[i + 1 :],
                )
                return p_sharded
        logging.warning(
            f"Could not shard {value.shape} with names {names} on axis {axis_name}, no suitable axis found."
        )
        return x

    return jax.tree.map(
        _split,
        params,
        is_leaf=lambda x: isinstance(x, nn.Partitioned),  # Consider a nn.Partitioned object as a leaf.
    )


def gather_array_with_mean_grads(
    x: jax.Array,
    axis: int,
    axis_name: str,
    gather_dtype: jnp.dtype | None = None,
    grad_scatter_dtype: jnp.dtype | None = None,
) -> jax.Array:
    """
    Gathering with averaging gradients across replicas.

    Args:
        x: The array to gather.
        axis: The axis of the array to gather across.
        axis_name: The axis name of the mesh to gather across.
        gather_dtype: The dtype to cast the array to before gathering. If None, no casting is performed.
        grad_scatter_dtype: The dtype to cast the gradients to before scattering. If None, the dtype of x is used.

    Returns:
        The gathered array with a gradient function that averages across replicas.
    """
    axis_size = jax.lax.psum(1, axis_name)
    param_dtype = x.dtype
    if grad_scatter_dtype is None:
        grad_scatter_dtype = param_dtype

    # Define a custom gradient for the gather operation.
    @jax.custom_gradient
    def f(x):
        def grad_fn(g):
            # pmean_scatter
            g = g.astype(grad_scatter_dtype)
            g = jax.lax.psum_scatter(g, axis_name, scatter_dimension=axis, tiled=True) / axis_size
            g = g.astype(param_dtype)
            return g

        if gather_dtype is not None:
            x = x.astype(gather_dtype)
        return jax.lax.all_gather(x, axis_name, axis=axis, tiled=True), grad_fn

    return f(x)


@jax.named_scope("gather_params")
def gather_params(
    params: PyTree, axis_name: str, gather_dtype: jnp.dtype | None = None, grad_scatter_dtype: jnp.dtype | None = None
) -> PyTree:
    """
    Gather parameters from all replicas across the given axis.

    Args:
        params: The parameters to gather.
        axis_name: The axis to gather parameters across.
        gather_dtype: The dtype to cast the parameters to before gathering. If None, no casting is performed.
        grad_scatter_dtype: The dtype to cast the gradients to before scattering. If None, the dtype of the parameters
            is used.

    Returns:
        PyTree of same structure as params, but with leaves gathered if they were a nn.Partitioned object.
    """

    def _gather(p: Parameter) -> Parameter:
        if isinstance(p, nn.Partitioned) and axis_name in p.names:
            param_shard = p.names
            shard_axis = param_shard.index(axis_name)
            value = gather_array_with_mean_grads(
                p.value,
                axis=shard_axis,
                axis_name=axis_name,
                gather_dtype=gather_dtype,
                grad_scatter_dtype=grad_scatter_dtype,
            )
            # If there are any other axes that are sharded, we need to keep the partitioned structure.
            # Otherwise, we can return the value directly.
            param_shard = param_shard[:shard_axis] + (None,) + param_shard[shard_axis + 1 :]
            if any(name is not None for name in param_shard):
                return nn.Partitioned(value, param_shard)
            return value
        return p

    return jax.tree.map(_gather, params, is_leaf=lambda x: isinstance(x, nn.Partitioned))


def shard_module_params(
    target: nn.Module | Callable,
    axis_name: str,
    min_weight_size: int = 2**18,
    gather_dtype: jnp.dtype | None = None,
    grad_scatter_dtype: jnp.dtype | None = None,
) -> nn.Module | Callable:
    """
    Shard parameters of a module across replicas.

    Args:
        target: The module to shard.
        axis_name: The axis name to shard parameters across.
        min_weight_size: The minimum size of a parameter to shard. Parameters with fewer values will not be sharded.
        gather_dtype: The dtype to cast the parameters to before gathering. If None, no casting is performed.
        grad_scatter_dtype: The dtype to cast the gradients to before scattering. If None, the dtype of the parameters
            is used.

    Returns:
        The module with sharded parameters.
    """
    return nn.map_variables(
        target,
        trans_in_fn=functools.partial(
            gather_params, axis_name=axis_name, gather_dtype=gather_dtype, grad_scatter_dtype=grad_scatter_dtype
        ),
        trans_out_fn=functools.partial(shard_params, axis_name=axis_name, min_weight_size=min_weight_size),
        mapped_collections="params",
        mutable=True,
    )


def sync_gradients(
    grads: PyTree,
    axis_names: Sequence[str],
) -> PyTree:
    """
    Synchronize gradients across devices.

    Gradients for parameters that are replicated over a given axis are averaged across devices.
    Parameters that are partitioned over a given axis are considered to already have a mean of
    the gradients on each device, and hence do not need to be altered.

    Args:
        grads: The gradients to synchronize.
        axis_names: The axis names to synchronize gradients across.

    Returns:
        The gradients averaged over the specified axes if they are replicated.
    """

    def sync_grad(g: Parameter) -> Parameter:
        if isinstance(g, nn.Partitioned):
            replication_axis_names = [name for name in axis_names if name not in jax.tree.leaves(g.names)]
            if len(replication_axis_names) == 0:
                # Parameters partitioned over all axes.
                return g
            # Average over remaining replicated axes.
            return g.replace(value=jax.lax.pmean(g.value, axis_name=replication_axis_names))
        # Parameters are replicated over all axes.
        return jax.lax.pmean(g, axis_name=axis_names)

    return jax.tree.map(sync_grad, grads, is_leaf=lambda x: isinstance(x, nn.Partitioned))
