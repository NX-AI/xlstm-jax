import jax
import jax.numpy as jnp


def bias_linspace_init(start: float, end: float) -> callable:
    """Linearly spaced bias init across dimensions."""

    def init_fn(key, shape, dtype=jnp.float32):  # pylint: disable=unused-argument
        assert len(shape) == 1, "Linspace init only supports 1D tensors."
        n_dims = shape[0]
        init_vals = jnp.linspace(start, end, num=n_dims, dtype=dtype)
        return init_vals

    return init_fn


def small_init(dim: int) -> callable:
    """
    Return initializer that creates a tensor with values according to the method described in:
    "Transformers without Tears: Improving the Normalization of Self-Attention", Nguyen, T. & Salazar, J. (2019).

    Adopted from https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/init_functions.py.
    """
    return jax.nn.initializers.normal(stddev=jnp.sqrt(2 / (5 * dim)))


def wang_init(dim: int, num_blocks: int) -> callable:
    """Adopted from https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/init_functions.py."""
    return jax.nn.initializers.normal(stddev=2 / num_blocks / jnp.sqrt(dim))


def uniform_init(min_val: float, max_val: float) -> callable:
    """Uniform initializer."""

    def init_fn(key, shape, dtype=jnp.float32):
        return jax.random.uniform(key, shape, minval=min_val, maxval=max_val, dtype=dtype)

    return init_fn
