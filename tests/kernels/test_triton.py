#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

"""Test to check if jax-triton works correctly"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

if pytest.triton_available:
    import jax_triton as jt
    import triton
    import triton.language as tl


@pytest.mark.skipif(not pytest.triton_available, reason="Triton is not available.")
def test_jax_triton():
    """This test executes the simple example from the Quickstart at https://github.com/jax-ml/jax-triton.
    It checks if jax-triton works correctly.
    """

    @triton.jit
    def add_kernel(
        x_ptr,
        y_ptr,
        length,
        output_ptr,
        block_size: tl.constexpr,
    ):
        """Adds two vectors."""
        pid = tl.program_id(axis=0)
        block_start = pid * block_size
        offsets = block_start + tl.arange(0, block_size)
        mask = offsets < length
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)

    def add(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        out_shape = jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype)
        block_size = 8
        return jt.triton_call(
            x, y, x.size, kernel=add_kernel, out_shape=out_shape, grid=(x.size // block_size,), block_size=block_size
        )

    x_val = jnp.arange(8)
    y_val = jnp.arange(8, 16)
    add_1 = add(x_val, y_val)
    add_2 = jax.jit(add)(x_val, y_val)
    np.testing.assert_array_equal(add_1, add_2)
