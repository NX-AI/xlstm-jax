#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xlstm_jax.kernels.stride_utils import get_stride, get_strides


def test_get_strides():
    """Test calculating strides of arrays."""
    array = jnp.ones((2, 3, 4))
    strides = get_strides(array)
    assert strides == [12, 4, 1]

    array = jax.ShapeDtypeStruct((2, 3, 4), jnp.float32)
    strides = get_strides(array)
    assert strides == [12, 4, 1]

    array = jax.ShapeDtypeStruct((10, 4, 2, 8, 7), jnp.bfloat16)
    strides = get_strides(array)
    assert strides == [448, 112, 56, 7, 1]


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_get_stride(seed: int):
    """Test single vs all strides calculation."""
    rng = np.random.default_rng(seed)
    num_dims = rng.integers(1, 10)
    shape = rng.integers(1, 10, size=num_dims)
    array = jax.ShapeDtypeStruct(shape, jnp.float32)
    strides = get_strides(array)
    for i in range(num_dims):
        stride = get_stride(array, axis=i)
        assert stride == strides[i], f"Expected {strides[i]}, got {stride} (shape={shape}, axis={i})"
