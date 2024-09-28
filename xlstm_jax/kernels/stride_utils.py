import jax
import numpy as np


def get_strides(array: jax.Array | jax.ShapeDtypeStruct) -> list[int]:
    """
    Returns the strides of a JAX array.

    Args:
        array: JAX array or shape-dtype struct.

    Returns:
        The strides of the array. Length is equal to the number of dimensions.
    """
    shape = array.shape
    size = array.size
    strides = []
    for s in shape:
        size = size // s
        strides.append(int(size))
    return strides


def get_stride(array: jax.Array | jax.ShapeDtypeStruct, axis: int) -> int:
    """
    Returns the stride of a JAX array at a given axis.

    To calculate all strides, use get_strides.

    Args:
        array: JAX array or shape-dtype struct.
        axis: The axis at which to calculate the stride.

    Returns:
        The stride of the array at the given axis.
    """
    shape = array.shape
    size = array.size
    stride = size // np.prod(shape[: axis + 1])
    return int(stride)
