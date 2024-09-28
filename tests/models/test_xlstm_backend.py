from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.fwbw import (
    mlstm_fwbw_custom_grad as mlstm_fwbw_custom_grad_jax,
)
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.simple import (
    parallel_stabilized_simple as parallel_stabilized_simple_jax,
)
from xlstm_jax.models.xlstm_pytorch.blocks.mlstm.backend.fwbw import (
    mLSTMfwbw as mLSTMfwbw_torch,
    mLSTMfwbwConfig as mLSTMfwbwConfig_torch,
)
from xlstm_jax.models.xlstm_pytorch.blocks.mlstm.backend.simple import (
    parallel_stabilized_simple as parallel_stabilized_simple_torch,
)

PyTree = Any


@pytest.mark.parametrize("context_length", [8, 128])
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
    # Fails for ctx > 128
    # qkv_dtype = "bfloat16"
    # qkv_torch_dtype = torch.bfloat16
    # gate_dtype = "float32"
    # gate_torch_dtype = torch.float32
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
    out_jax = out_jax.astype(np.float32)
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
    out_pytorch = out_pytorch.to(torch.float32)
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


@pytest.mark.parametrize("context_length", [128, 256])
@pytest.mark.parametrize("chunk_size", [8, 64])
@pytest.mark.parametrize("stabilize_correctly", [False, True])
def test_jax_fwbw_vs_parallelized(context_length: int, chunk_size: int, stabilize_correctly: bool):
    """
    Tests the mLSTM forward-backward stabilized backend compared to parallel.
    """
    if chunk_size > context_length:
        pytest.skip("Chunk size must be smaller than the context length.")
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

    config = mLSTMfwbwConfig_torch(
        chunk_size=chunk_size,
        stabilize_correctly=stabilize_correctly,
    )

    # Run the JAX backend.
    backend = mlstm_fwbw_custom_grad_jax(config)
    out_fwbw = backend(
        jnp.array(q),
        jnp.array(k),
        jnp.array(v),
        jnp.array(igate_preact),
        jnp.array(fgate_preact),
    )
    out_fwbw = jax.device_get(out_fwbw)

    # Run parallelized backend.
    out_parallel = parallel_stabilized_simple_jax(
        jnp.array(q),
        jnp.array(k),
        jnp.array(v),
        jnp.array(igate_preact),
        jnp.array(fgate_preact),
    )
    out_parallel = jax.device_get(out_parallel)

    if not stabilize_correctly:
        out_fwbw = jax.nn.standardize(out_fwbw, axis=-1)
        out_parallel = jax.nn.standardize(out_parallel, axis=-1)

    # Compare the results.
    np.testing.assert_allclose(
        out_fwbw,
        out_parallel,
        atol=1e-5,
        rtol=1e-2,
        err_msg="Mismatch between parallel and fwbw backends.",
    )


@pytest.mark.parametrize("context_length", [128, 256])
@pytest.mark.parametrize("chunk_size", [8, 64])
@pytest.mark.parametrize("use_jit", [False])
@pytest.mark.parametrize("stabilize_correctly", [False, True])
def test_fwbw_pytorch_vs_jax(context_length: int, chunk_size: int, use_jit: bool, stabilize_correctly: bool):
    """
    Tests the mLSTM forward-backward stabilized backend in PyTorch compared to JAX.
    """
    if chunk_size > context_length:
        pytest.skip("Chunk size must be smaller than the context length.")
    # Prepare the input data.
    B = 1
    S = context_length
    DH = 64
    rng = np.random.default_rng(2)
    q = rng.normal(size=(B, 1, S, DH)).astype(np.float32)
    k = rng.normal(size=(B, 1, S, DH)).astype(np.float32)
    v = rng.normal(size=(B, 1, S, DH)).astype(np.float32)
    igate_preact = rng.normal(size=(B, 1, S, 1)).astype(np.float32)
    fgate_preact = rng.normal(size=(B, 1, S, 1)).astype(np.float32) + 4.5

    config = mLSTMfwbwConfig_torch(
        chunk_size=chunk_size,
        stabilize_correctly=stabilize_correctly,
    )

    # Run the JAX backend.
    backend = mlstm_fwbw_custom_grad_jax(config)
    if use_jit:
        backend = jax.jit(backend)
    out_jax = backend(
        jnp.array(q)[:, 0],
        jnp.array(k)[:, 0],
        jnp.array(v)[:, 0],
        jnp.array(igate_preact)[:, 0],
        jnp.array(fgate_preact)[:, 0],
    )
    out_jax = jax.device_get(out_jax)

    # Run the PyTorch backend.
    torch_module = mLSTMfwbw_torch(config)
    with torch.no_grad():
        out_torch = torch_module(
            torch.from_numpy(q),
            torch.from_numpy(k),
            torch.from_numpy(v),
            torch.from_numpy(igate_preact),
            torch.from_numpy(fgate_preact),
        )
    out_torch = out_torch.cpu().numpy()[:, 0]

    # Compare the results.
    np.testing.assert_allclose(
        out_jax,
        out_torch,
        atol=1e-5,
        rtol=1e-2,
        err_msg="Mismatch between JAX and PyTorch backends in fwbw.",
    )


@pytest.mark.parametrize("context_length", [128, 256])
@pytest.mark.parametrize("chunk_size", [8, 64])
@pytest.mark.parametrize("stabilize_correctly", [False, True])
def test_pytorch_fwbw_vs_parallel(context_length: int, chunk_size: int, stabilize_correctly: bool):
    """
    Tests the parallelization of the mLSTM backend.

    In float64, both backends are equivalent for all context lengths up to high precision,
    but this is difficult to test due to JAX requiring the flag "JAX_ENABLE_X64" which
    messes with the default dtype and fails the other tests.
    """
    if chunk_size > context_length:
        pytest.skip("Chunk size must be smaller than the context length.")
    # Prepare the input data.
    B = 1
    S = context_length
    DH = 64
    rng = np.random.default_rng(2)
    q = rng.normal(size=(B, 1, S, DH)).astype(np.float32)
    k = rng.normal(size=(B, 1, S, DH)).astype(np.float32)
    v = rng.normal(size=(B, 1, S, DH)).astype(np.float32)
    igate_preact = rng.normal(size=(B, 1, S, 1)).astype(np.float32)
    fgate_preact = rng.normal(size=(B, 1, S, 1)).astype(np.float32) + 4.5

    config = mLSTMfwbwConfig_torch(
        chunk_size=chunk_size,
        stabilize_correctly=stabilize_correctly,
    )

    # Run the PyTorch fwbw backend.
    torch_module = mLSTMfwbw_torch(config)
    with torch.no_grad():
        out_torch_fwbw = torch_module(
            torch.from_numpy(q),
            torch.from_numpy(k),
            torch.from_numpy(v),
            torch.from_numpy(igate_preact),
            torch.from_numpy(fgate_preact),
        )
    out_torch_fwbw = out_torch_fwbw.cpu().numpy()[:, 0]

    # Run the PyTorch parallel backend.
    with torch.no_grad():
        out_torch_parallel = parallel_stabilized_simple_torch(
            torch.from_numpy(q),
            torch.from_numpy(k),
            torch.from_numpy(v),
            torch.from_numpy(igate_preact),
            torch.from_numpy(fgate_preact),
        )
    out_torch_parallel = out_torch_parallel.cpu().numpy()[:, 0]

    if not stabilize_correctly:
        out_torch_fwbw = jax.nn.standardize(out_torch_fwbw, axis=-1)
        out_torch_parallel = jax.nn.standardize(out_torch_parallel, axis=-1)

    # Compare the results. High tolerance due to the float32 precision. In float64, both
    # backends are equivalent up to the default tolerance 1e-7.
    np.testing.assert_allclose(
        out_torch_fwbw,
        out_torch_parallel,
        atol=1e-5,
        rtol=1e-2,
        err_msg=(
            "Mismatch between parallel and fwbw in PyTorch backends at "
            f"context_length={context_length} and chunk size {chunk_size}."
        ),
    )


@pytest.mark.parametrize("context_length", [128, 256])
@pytest.mark.parametrize("chunk_size", [8, 64])
@pytest.mark.parametrize("use_jit", [False])
@pytest.mark.parametrize("stabilize_correctly", [False])  # True fails for few elements
def test_fwbw_pytorch_vs_jax_backward(context_length: int, chunk_size: int, use_jit: bool, stabilize_correctly: bool):
    """
    Tests the mLSTM forward-backward stabilized backend in PyTorch compared to JAX.
    """
    if chunk_size > context_length:
        pytest.skip("Chunk size must be smaller than the context length.")
    # Prepare the input data.
    B = 1
    S = context_length
    DH = 64
    rng = np.random.default_rng(2)
    q = rng.normal(size=(B, 1, S, DH)).astype(np.float32)
    k = rng.normal(size=(B, 1, S, DH)).astype(np.float32)
    v = rng.normal(size=(B, 1, S, DH)).astype(np.float32)
    igate_preact = rng.normal(size=(B, 1, S, 1)).astype(np.float32)
    fgate_preact = rng.normal(size=(B, 1, S, 1)).astype(np.float32) + 4.5

    config = mLSTMfwbwConfig_torch(
        chunk_size=chunk_size,
        stabilize_correctly=stabilize_correctly,
    )

    # Run the JAX backend.
    def _jax_backend(q, k, v, igate_preact, fgate_preact):
        out_jax = mlstm_fwbw_custom_grad_jax(config)(q, k, v, igate_preact, fgate_preact, None, None, None)
        return ((out_jax - jax.lax.stop_gradient(q)) ** 2).sum()

    jax_grad_backend = jax.value_and_grad(_jax_backend, argnums=(0, 1, 2, 3, 4))
    if use_jit:
        jax_grad_backend = jax.jit(jax_grad_backend)
    out_jax, grad_jax = jax_grad_backend(
        jnp.array(q)[:, 0],
        jnp.array(k)[:, 0],
        jnp.array(v)[:, 0],
        jnp.array(igate_preact)[:, 0],
        jnp.array(fgate_preact)[:, 0],
    )
    grad_jax = jax.device_get(grad_jax)

    # Run the PyTorch backend.
    torch_module = mLSTMfwbw_torch(config)
    q_torch = torch.from_numpy(q).requires_grad_()
    k_torch = torch.from_numpy(k).requires_grad_()
    v_torch = torch.from_numpy(v).requires_grad_()
    igate_preact_torch = torch.from_numpy(igate_preact).requires_grad_()
    fgate_preact_torch = torch.from_numpy(fgate_preact).requires_grad_()
    out_torch = torch_module(
        q_torch,
        k_torch,
        v_torch,
        igate_preact_torch,
        fgate_preact_torch,
    )
    out_torch = ((out_torch - q_torch.detach()) ** 2).sum()
    out_torch.backward()
    grad_torch = [
        t.grad.cpu().numpy()[:, 0] for t in [q_torch, k_torch, v_torch, igate_preact_torch, fgate_preact_torch]
    ]

    # Compare the results.
    np.testing.assert_allclose(
        jax.device_get(out_jax),
        out_torch.detach().cpu().numpy(),
        atol=1e-5,
        rtol=1e-2,
        err_msg="Mismatch between JAX and PyTorch outputs in fwbw.",
    )
    grad_names = ["q", "k", "v", "igate_preact", "fgate_preact"]
    for gjax, gtorch, gname in zip(grad_jax, grad_torch, grad_names):
        np.testing.assert_allclose(
            gjax,
            gtorch,
            atol=1e-5,
            rtol=1e-2,
            err_msg=f"Mismatch between JAX and PyTorch gradient backends in fwbw for {gname}.",
        )
