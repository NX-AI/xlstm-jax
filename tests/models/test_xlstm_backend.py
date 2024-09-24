from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.simple import (
    parallel_stabilized_simple as parallel_stabilized_simple_jax,
)
from xlstm_jax.models.xlstm_pytorch.blocks.mlstm.backend.simple import (
    parallel_stabilized_simple as parallel_stabilized_simple_torch,
)

PyTree = Any


@pytest.mark.parametrize("context_length", [8, 128, 1024, 2048])
@pytest.mark.parametrize("use_jit", [False, True])
def test_parallelize_backend_pytorch_vs_jax(context_length: int, use_jit: bool):
    """
    Tests the parallelization of the mLSTM backend.

    In float64, both backends are equivalent for all context lengths up to high precision,
    but this is difficult to test due to JAX requiring the flag "JAX_ENABLE_X64" which
    messes with the default dtype and fails the other tests.
    """
    # Prepare the input data.
    B = 1
    S = context_length
    DH = 64
    qkv_dtype = "float32"
    qkv_torch_dtype = torch.float32
    gate_dtype = "float32"
    gate_torch_dtype = torch.float32
    rng = np.random.default_rng(2)
    q = rng.normal(size=(B, S, DH)).astype(np.float32)
    k = rng.normal(size=(B, S, DH)).astype(np.float32)
    v = rng.normal(size=(B, S, DH)).astype(np.float32)
    igate_preact = rng.normal(size=(B, S, 1)).astype(np.float32)
    fgate_preact = rng.normal(size=(B, S, 1)).astype(np.float32) + 4.5

    # Run the JAX backend.
    jax_backend = parallel_stabilized_simple_jax
    if use_jit:
        jax_backend = jax.jit(jax_backend)
    out_jax = jax_backend(
        jnp.array(q, dtype=qkv_dtype),
        jnp.array(k, dtype=qkv_dtype),
        jnp.array(v, dtype=qkv_dtype),
        jnp.array(igate_preact, dtype=gate_dtype),
        jnp.array(fgate_preact, dtype=gate_dtype),
    )
    out_jax = jax.device_get(out_jax)

    # Run the PyTorch backend.
    with torch.no_grad():
        out_pytorch = parallel_stabilized_simple_torch(
            torch.from_numpy(q).to(qkv_torch_dtype)[:, None],
            torch.from_numpy(k).to(qkv_torch_dtype)[:, None],
            torch.from_numpy(v).to(qkv_torch_dtype)[:, None],
            torch.from_numpy(igate_preact).to(gate_torch_dtype)[:, None],
            torch.from_numpy(fgate_preact).to(gate_torch_dtype)[:, None],
        )
    out_pytorch = out_pytorch.cpu().numpy()[:, 0]

    # Compare the results. High tolerance due to the float32 precision. In float64, both
    # backends are equivalent up to the default tolerance 1e-7.
    np.testing.assert_allclose(
        out_jax,
        out_pytorch,
        atol=1e-5,
        rtol=1e-2,
        err_msg=(
            f"Mismatch between JAX and PyTorch backends at context_length={context_length} and JIT set to {use_jit}."
        ),
    )


@pytest.mark.parametrize("context_length", [8, 16])
@pytest.mark.parametrize("use_jit", [False, True])
def test_parallelize_backend_float32_vs_bfloat16(context_length: int, use_jit: bool):
    """
    Tests the parallelization of the mLSTM backend.

    Only checks for short context lengths, as for longer context lengths, the bfloat16
    precision significantly diverges from the float32 precision.
    """
    # Prepare the input data.
    B = 1
    S = context_length
    DH = 64
    rng = np.random.default_rng(2)
    q = rng.normal(size=(B, S, DH)).astype(np.float32)
    k = rng.normal(size=(B, S, DH)).astype(np.float32)
    v = rng.normal(size=(B, S, DH)).astype(np.float32)
    igate_preact = rng.normal(size=(B, S, 1)).astype(np.float32)
    fgate_preact = rng.normal(size=(B, S, 1)).astype(np.float32) + 4.5

    # Run the float32 precision.
    jax_backend = parallel_stabilized_simple_jax
    if use_jit:
        jax_backend = jax.jit(jax_backend)
    out_float32 = jax_backend(
        jnp.array(q, dtype="bfloat16").astype("float32"),
        jnp.array(k, dtype="bfloat16").astype("float32"),
        jnp.array(v, dtype="bfloat16").astype("float32"),
        jnp.array(igate_preact, dtype="float32"),
        jnp.array(fgate_preact, dtype="float32"),
        eps=1e-5,
    )
    out_float32 = jax.device_get(out_float32)

    # Run the bfloat16 precision.
    out_bfloat16 = jax_backend(
        jnp.array(q, dtype="bfloat16"),
        jnp.array(k, dtype="bfloat16"),
        jnp.array(v, dtype="bfloat16"),
        jnp.array(igate_preact, dtype="float32"),
        jnp.array(fgate_preact, dtype="float32"),
        eps=1e-5,
    )
    out_bfloat16 = out_bfloat16.astype("float32")
    out_bfloat16 = jax.device_get(out_bfloat16)

    # Compare the results.
    np.testing.assert_allclose(
        out_float32,
        out_bfloat16,
        atol=1e-2,
        rtol=1e-2,
        err_msg=(
            "Mismatch between float32 and bfloat16 backends at "
            f"context_length={context_length} and JIT set to {use_jit}."
        ),
    )
