#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import jax
import pytest

from xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.triton_kernels import (
    mLSTMBackendTriton,
    mLSTMBackendTritonConfig,
)


@pytest.mark.skipif(not pytest.triton_available, reason="Triton is not available.")
def test_mlstm_kernel_backend(default_qkvif: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]):
    """Test a simple forward pass in the backend module."""
    q, k, v, igate_preact, fgate_preact = default_qkvif
    B, NH, S, DH = q.shape
    CHUNK_SIZE = 64

    backend = mLSTMBackendTriton(
        config=mLSTMBackendTritonConfig(
            autocast_dtype="float32",
            chunk_size=CHUNK_SIZE,
            stabilize_correctly=True,
            eps=1e-5,
            backend_name="triton_stablef",
            norm_val=1.0,
        )
    )
    h_out_triton = backend.apply({}, q, k, v, igate_preact, fgate_preact)
    assert h_out_triton.shape == (B, NH, S, DH)
