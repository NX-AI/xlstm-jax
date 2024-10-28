import jax
import jax.numpy as jnp
from jax.nn.initializers import Initializer


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
        del key
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
