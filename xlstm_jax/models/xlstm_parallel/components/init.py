from typing import Literal

import jax
import jax.numpy as jnp
from jax.nn.initializers import Initializer

InitDistribution = Literal["normal", "truncated_normal", "uniform"]


def bias_linspace_init(start: float, end: float) -> Initializer:
    """Linearly spaced bias init across dimensions.

    Only supports 1D array shapes. Array values are including start and end.

    Args:
        start: Start value for the linspace.
        end: End value for the linspace.

    Returns:
        Initializer function that creates a 1D array with linearly spaced values between start and end.
    """

    def init_fn(key, shape, dtype=jnp.float32):
        assert len(shape) == 1, "Linspace init only supports 1D tensors."
        n_dims = shape[0]
        init_vals = jnp.linspace(start, end, num=n_dims, dtype=dtype)
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
