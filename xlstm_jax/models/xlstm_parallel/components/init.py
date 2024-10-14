from typing import Literal

import jax
import jax.numpy as jnp
from jax.nn.initializers import Initializer

InitDistribution = Literal["normal", "truncated_normal", "uniform"]
InitFnName = Literal["small", "wang", "wang2", "zeros"]


def bias_linspace_init(start: float, end: float, axis_name: str | None = None) -> Initializer:
    """Linearly spaced bias init across dimensions.

    Only supports 1D array shapes. Array values are including start and end. If axis name is provided, the linspace
    is sharded over the axis.

    Args:
        start: Start value for the linspace.
        end: End value for the linspace.
        axis_name: Optional axis name to shard over.

    Returns:
        Initializer function that creates a 1D array with linearly spaced values between start and end.
    """

    def init_fn(key, shape, dtype=jnp.float32):
        assert len(shape) == 1, "Linspace init only supports 1D tensors."
        n_dims = shape[0]
        dstart, dend = start, end
        if axis_name is not None:
            axis_size = jax.lax.psum(1, axis_name)
            if axis_size > 1:
                axis_idx = jax.lax.axis_index(axis_name)
                step_size = (dend - dstart) / (axis_size * n_dims - 1)
                dstart = dstart + step_size * axis_idx * n_dims
                dend = dstart + step_size * (n_dims - 1)
        init_vals = jnp.linspace(dstart, dend, num=n_dims, dtype=dtype)
        return init_vals

    return init_fn


def small_init(dim: int, distribution: InitDistribution = "normal") -> Initializer:
    """Create initializer of Nguyen et al. (2019).

    Adopted from https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/init_functions.py.
    The initializer creates an array with values according to the method described in:
    "Transformers without Tears: Improving the Normalization of Self-Attention", Nguyen, T. & Salazar, J. (2019).
    The array values are sampled with a standard deviation of sqrt(2 / (5 * dim)).

    Args:
        dim: Feature dimensionality to use in the initializer.
        distribution: The distribution to sample from. Supported are normal, truncated normal, and uniform.

    Returns:
        Initializer function following the above described method.
    """
    std = jnp.sqrt(2 / (5 * dim))
    return _dist_from_stddev(std, distribution)


def wang_init(dim: int, num_blocks: int, distribution: InitDistribution = "normal") -> Initializer:
    """Create Wang initializer.

    Adopted from https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/init_functions.py.
    Commonly used for the output layers of residual blocks. The array values are sampled with a standard deviation
    of 2 / num_blocks / sqrt(dim).

    Args:
        dim: Feature dimensionality to use in the initializer.
        num_blocks: Number of layers / blocks in the model.
        distribution: The distribution to sample from. Supported are normal, truncated normal, and uniform.

    Returns:
        Initializer function of the wang init.
    """
    std = 2 / num_blocks / jnp.sqrt(dim)
    return _dist_from_stddev(std, distribution)


def create_common_init_fn(
    fn_name: InitFnName, dim: int, num_blocks: int, distribution: InitDistribution = "normal"
) -> Initializer:
    """Create common initializer function.

    Allows to create different types of initializers with a single function call.

    Args:
        fn_name: Name of the initializer function to create. Supported are "small" (:func:`~small_init`),
            "wang" (:func:`~wang_init`), "wang2" (:func:`~wang_init` with 2x block num), and
            "zeros" (zero initializer).
        dim: Feature dimensionality to use in the initializer.
        num_blocks: Number of layers / blocks in the model.
        distribution: The distribution to sample from. Supported are normal, truncated normal, and uniform.

    Returns:
        Initializer function of the specified type.
    """
    if fn_name == "small":
        return small_init(dim, distribution)
    elif fn_name == "wang":
        return wang_init(dim, num_blocks, distribution)
    elif fn_name == "wang2":
        return wang_init(dim, 2 * num_blocks, distribution)
    elif fn_name == "zeros":
        return jax.nn.initializers.zeros
    else:
        raise ValueError(f"Invalid initializer function name {fn_name}.")


def _dist_from_stddev(stddev: float, distribution: InitDistribution) -> Initializer:
    """Create initializer with specified standard deviation and distribution.

    The distribution has a zero mean and specified standard deviation.

    Args:
        stddev: The standard deviation of the distribution.
        distribution: The distribution to sample from. Supported are normal, truncated normal, and uniform.

    Returns:
        Initializer function that samples the array value from the specified distribution with the given standard
        deviation.
    """
    if distribution == "normal":
        return jax.nn.initializers.normal(stddev=stddev)
    elif distribution == "truncated_normal":
        return jax.nn.initializers.truncated_normal(stddev=stddev)
    elif distribution == "uniform":
        uniform_bounds = jnp.sqrt(3.0) * stddev
        return uniform_init(-uniform_bounds, uniform_bounds)
    else:
        raise ValueError(f"Invalid distribution type {distribution}.")


def uniform_init(min_val: float, max_val: float) -> Initializer:
    """Create uniform initializer.

    Args:
        min_val: Minimum value of the uniform distribution.
        max_val: Maximum value of the uniform distribution.

    Returns:
        An initializer function which samples values randomly between min_val and max_val.
    """

    def init_fn(key, shape, dtype=jnp.float32):
        return jax.random.uniform(key, shape, minval=min_val, maxval=max_val, dtype=dtype)

    return init_fn
