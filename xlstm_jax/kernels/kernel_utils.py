import jax.numpy as jnp
import triton.language as tl

_jax_to_triton_dtype = {
    "float32": tl.float32,
    "float16": tl.float16,
    "bfloat16": tl.bfloat16,
}


def jax2triton_dtype(dtype: jnp.dtype | str) -> tl.dtype:
    """
    Converts a JAX dtype to a Triton dtype.

    Args:
        dtype: JAX dtype.

    Returns:
        Triton dtype.
    """
    return getattr(tl, str(dtype))
