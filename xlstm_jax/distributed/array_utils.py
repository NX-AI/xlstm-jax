import jax
import jax.numpy as jnp
from flax import linen as nn

from xlstm_jax.common_types import Parameter, PRNGKeyArray, PyTree


def fold_rng_over_axis(rng: PRNGKeyArray, axis_name: str) -> PRNGKeyArray:
    """
    Folds the random number generator over the given axis.

    This is useful for generating a different random number for each device
    across a certain axis (e.g. the model axis).

    Args:
        rng: The random number generator.
        axis_name: The axis name to fold the random number generator over.

    Returns:
        A new random number generator, different for each device index along the axis.
    """
    axis_index = jax.lax.axis_index(axis_name)
    return jax.random.fold_in(rng, axis_index)


def split_array_over_mesh(x: jax.Array, axis_name: str, split_axis: int) -> jax.Array:
    """
    Split an array over the given mesh axis.

    Args:
        x: The array to split.
        axis_name: The axis name of the mesh to split over.
        split_axis: The axis of the array to split.

    Returns:
        The slice of the array for the current device along the given axis.
    """
    axis_size = jax.lax.psum(1, axis_name)
    if axis_size == 1:
        return x
    axis_index = jax.lax.axis_index(axis_name)
    slice_size = x.shape[split_axis] // axis_size
    x = jax.lax.dynamic_slice_in_dim(
        x,
        axis_index * slice_size,
        slice_size,
        axis=split_axis,
    )
    return x


def stack_params(
    params: PyTree,
    axis_name: str,
    axis: int = 0,
    mask_except: jax.Array | int | None = None,
) -> PyTree:
    """
    Stacks sharded parameters along a given axis name.

    Args:
        params: PyTree of parameters.
        axis_name: Name of the axis to stack along.
        axis: Index of the axis to stack along.
        mask_except: If not None, only the `mask_except`-th shard will be non-zero.

    Returns:
        PyTree of parameters with the same structure as `params`, but with the leaf
        nodes replaced by `nn.Partitioned` objects with sharding over axis name added
        to `axis`-th axis of parameters.
    """

    def _stack(x: Parameter) -> Parameter:
        if isinstance(x, nn.Partitioned):
            value, names = x.value, x.names
        else:
            value, names = x, (None,) * x.ndim
        if mask_except is not None:
            axis_index = jax.lax.axis_index(axis_name)
            value = jnp.where(axis_index == mask_except, value, 0.0)
        value = jnp.expand_dims(value, axis)
        names = names[:axis] + (axis_name,) + names[axis:]
        return nn.Partitioned(value, names=names)

    return jax.tree.map(_stack, params, is_leaf=lambda x: isinstance(x, nn.Partitioned))


def unstack_params(params: PyTree, axis_name: str) -> PyTree:
    """
    Unstacks parameters along a given axis name.

    Inverse operation to `stack_params`.

    Args:
        params: PyTree of parameters.
        axis_name: Name of the axis to unstack along.

    Returns:
        PyTree of parameters with the same structure as `params`, but
        with the leaf nodes having the sharding over the axis name removed.
    """

    def _unstack(x: Parameter) -> Parameter:
        if isinstance(x, nn.Partitioned) and axis_name in x.names:
            value = x.value
            names = x.names
            axis_idx = names.index(axis_name)
            value = value.squeeze(axis_idx)
            names = names[:axis_idx] + names[axis_idx + 1 :]
            if all([n is None for n in names]):
                return value
            else:
                return nn.Partitioned(value, names=names)
        else:
            return x

    return jax.tree.map(_unstack, params, is_leaf=lambda x: isinstance(x, nn.Partitioned))
