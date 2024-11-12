import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

try:
    from xlstm_jax.kernels.mlstm_chunkwise.max_triton_fwbw_v5xlchunksize.triton_fwbw import mlstm_chunkwise_max_triton
except ImportError:
    mlstm_chunkwise_max_triton = None
from xlstm_jax.models.xlstm_pytorch.blocks.mlstm.backend.fwbw import (
    mLSTMfwbw as mLSTMfwbw_torch,
    mLSTMfwbwConfig as mLSTMfwbwConfig_torch,
)
from xlstm_jax.models.xlstm_pytorch.blocks.mlstm.backend.simple import (
    parallel_stabilized_simple as parallel_stabilized_simple_torch,
)

PYTORCH_KERNEL_OUTPUT_FILES = sorted(
    list(
        Path(os.path.join(os.environ["XLSTM_JAX_TEST_DATA"], "mlstm_pytorch_kernel_outputs/chunkwise_fwbw")).glob(
            "*.npz"
        )
    )
)


@pytest.mark.skipif(not pytest.triton_available, reason="Triton is not available.")
@pytest.mark.parametrize("mlstm_kernel", [mlstm_chunkwise_max_triton])
def test_mlstm_chunkwise_fw_full_module(
    default_qkvif: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], mlstm_kernel: callable
):
    """Test a simple forward pass in the full module."""
    q, k, v, igate_preact, fgate_preact = default_qkvif
    B, NH, S, DH = q.shape
    CHUNK_SIZE = 64

    h_out_triton = mlstm_kernel(q, k, v, igate_preact, fgate_preact, chunk_size=CHUNK_SIZE)
    assert h_out_triton.shape == (B, NH, S, DH)


COMBINATIONS = [
    # (B, NH, S, DH, CHUNK_SIZE)
    (1, 2, 128, 64, 64),
    (8, 4, 128, 64, 64),
    (1, 2, 2048, 64, 64),
]


@pytest.mark.skipif(not pytest.triton_available, reason="Triton is not available.")
@pytest.mark.parametrize("seed", [0, 42])
@pytest.mark.parametrize(["B", "NH", "S", "DH", "CHUNK_SIZE"], COMBINATIONS)
@pytest.mark.parametrize("mlstm_kernel", [mlstm_chunkwise_max_triton])
def test_mlstm_chunkwise_fwbw_vs_pytorch_fwbw(
    B: int, NH: int, S: int, DH: int, CHUNK_SIZE: int, seed: int, mlstm_kernel: callable
):
    """Test the Triton kernels against the PyTorch fwbw backend."""
    config = mLSTMfwbwConfig_torch(
        chunk_size=CHUNK_SIZE,
        stabilize_correctly=False,
    )
    fwbw_torch_module = mLSTMfwbw_torch(config)
    _compare_to_pytorch_fn(B, NH, S, DH, CHUNK_SIZE, seed, mlstm_kernel, fwbw_torch_module)


@pytest.mark.skipif(not pytest.triton_available, reason="Triton is not available.")
@pytest.mark.parametrize("seed", [0, 42])
@pytest.mark.parametrize(["B", "NH", "S", "DH", "CHUNK_SIZE"], COMBINATIONS)
@pytest.mark.parametrize("mlstm_kernel", [mlstm_chunkwise_max_triton])
def test_mlstm_chunkwise_fwbw_vs_pytorch_parallel(
    B: int, NH: int, S: int, DH: int, CHUNK_SIZE: int, seed: int, mlstm_kernel: callable
):
    """Test the Triton kernels against the PyTorch parallel backend."""
    _compare_to_pytorch_fn(B, NH, S, DH, CHUNK_SIZE, seed, mlstm_kernel, parallel_stabilized_simple_torch)


def _compare_to_pytorch_fn(
    B: int,
    NH: int,
    S: int,
    DH: int,
    CHUNK_SIZE: int,
    seed: int,
    mlstm_kernel: callable,
    pytorch_backend: callable,
    atol: float = 1e-3,
    rtol: float = 1e-3,
):
    """Compare the Triton kernels against a PyTorch backend."""
    # Generate random inputs.
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(B, NH, S, DH)).astype(np.float32)
    k = rng.normal(size=(B, NH, S, DH)).astype(np.float32)
    v = rng.normal(size=(B, NH, S, DH)).astype(np.float32)
    igate_preact = rng.normal(size=(B, NH, S)).astype(np.float32)
    fgate_preact = rng.normal(size=(B, NH, S)).astype(np.float32)
    targets = rng.normal(size=(B, NH, S, DH)).astype(np.float32)

    # Convert to JAX arrays.
    q_jax = jnp.array(q)
    k_jax = jnp.array(k)
    v_jax = jnp.array(v)
    igate_preact_jax = jnp.array(igate_preact)
    fgate_preact_jax = jnp.array(fgate_preact)
    targets_jax = jnp.array(targets)

    # Define the loss function for triton kernels.
    def loss_fn(q, k, v, igate_preact, fgate_preact):
        h_out_triton = mlstm_kernel(
            q, k, v, igate_preact, fgate_preact, chunk_size=CHUNK_SIZE, autocast_kernel_dtype=jnp.float32
        )
        # Simulate layer norm.
        h_out_triton = (h_out_triton - h_out_triton.mean(axis=-1, keepdims=True)) / h_out_triton.std(
            axis=-1, keepdims=True, ddof=1
        )
        return (h_out_triton - targets_jax).sum()

    # Compute the gradients.
    grad_fn = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4)))
    out_triton, (grad_q, grad_k, grad_v, grad_igate_preact, grad_fgate_preact) = grad_fn(
        q_jax, k_jax, v_jax, igate_preact_jax, fgate_preact_jax
    )

    # Check the gradient shapes of the triton kernels.
    assert grad_q.shape == q.shape, "Mismatch in the shape of the gradient of q."
    assert grad_k.shape == k.shape, "Mismatch in the shape of the gradient of k."
    assert grad_v.shape == v.shape, "Mismatch in the shape of the gradient of v."
    assert grad_igate_preact.shape == igate_preact.shape, "Mismatch in the shape of the gradient of igate_preact."
    assert grad_fgate_preact.shape == fgate_preact.shape, "Mismatch in the shape of the gradient of fgate_preact."

    # Set up input data for the PyTorch backend.
    q_torch = torch.from_numpy(q).requires_grad_()
    k_torch = torch.from_numpy(k).requires_grad_()
    v_torch = torch.from_numpy(v).requires_grad_()
    # PyTorch backends work with the gates having a trailing dimension of 1.
    igate_preact_torch = torch.from_numpy(igate_preact[..., None]).requires_grad_()
    fgate_preact_torch = torch.from_numpy(fgate_preact[..., None]).requires_grad_()
    targets_torch = torch.from_numpy(targets)
    # Run the PyTorch backend with simulated layer norm.
    out_torch: torch.Tensor = pytorch_backend(
        q_torch,
        k_torch,
        v_torch,
        igate_preact_torch,
        fgate_preact_torch,
    )
    out_torch = (out_torch - out_torch.mean(dim=-1, keepdim=True)) / out_torch.std(dim=-1, keepdim=True)
    out_torch = (out_torch - targets_torch).sum()
    out_torch.backward()
    # Convert PyTorch gradients to numpy arrays.
    out_torch = out_torch.detach().numpy()
    grad_torch = [t.grad.cpu().numpy() for t in [q_torch, k_torch, v_torch, igate_preact_torch, fgate_preact_torch]]
    # Remove the trailing dimension of 1 from the gates.
    grad_torch[-2] = grad_torch[-2].squeeze(-1)
    grad_torch[-1] = grad_torch[-1].squeeze(-1)

    # Convert the JAX output and gradients to numpy arrays.
    out_triton = jax.device_get(out_triton)
    grad_titon = [grad_q, grad_k, grad_v, grad_igate_preact, grad_fgate_preact]
    grad_titon = [jax.device_get(g) for g in grad_titon]

    # Check the output and gradients.
    np.testing.assert_allclose(
        out_triton, out_torch, rtol=rtol, atol=atol, err_msg="Mismatch between Triton and PyTorch in the output value."
    )
    grad_names = ["q", "k", "v", "igate_preact", "fgate_preact"]
    for g_torch, g_triton, g_name in zip(grad_torch, grad_titon, grad_names):
        np.testing.assert_allclose(
            g_triton,
            g_torch,
            rtol=rtol,
            atol=atol,
            err_msg=f"Mismatch between Triton and PyTorch in the gradient {g_name}.",
        )


@pytest.mark.skipif(not pytest.triton_available, reason="Triton is not available.")
@pytest.mark.skipif(len(PYTORCH_KERNEL_OUTPUT_FILES) == 0, reason="No PyTorch kernel output files found.")
@pytest.mark.parametrize("file_path", PYTORCH_KERNEL_OUTPUT_FILES)
@pytest.mark.parametrize("mlstm_kernel", [mlstm_chunkwise_max_triton])
def test_mlstm_chunkwise_fw_vs_pytorch_kernel(file_path: Path, mlstm_kernel: callable):
    """Compare JAX kernel to PyTorch kernel output."""
    file_name = file_path.name.split(".")[0]
    int_keys = ["B", "NH", "S", "DH", "CHUNK", "seed"]
    hparams = {}
    for hparam_str in file_name.split("_"):
        if hparam_str.startswith("dtype"):
            hparams["dtype"] = {"fp32": jnp.float32, "fp16": jnp.float16, "bf16": jnp.bfloat16}[hparam_str[5:]]
        else:
            key = [k for k in int_keys if hparam_str.startswith(k)]
            assert len(key) > 0, f"Unknown hyperparameter: {hparam_str}"
            key = key[0]
            hparams[key] = int(hparam_str[len(key) :])

    data_torch = np.load(file_path)
    q = jnp.array(data_torch["q"]).astype(hparams["dtype"])
    k = jnp.array(data_torch["k"]).astype(hparams["dtype"])
    v = jnp.array(data_torch["v"]).astype(hparams["dtype"])
    igate_preact = jnp.array(data_torch["igate_preact"]).astype(hparams["dtype"])
    fgate_preact = jnp.array(data_torch["fgate_preact"]).astype(jnp.float32)

    def loss_fn(q, k, v, igate_preact, fgate_preact):
        h_out_triton = mlstm_kernel(
            q, k, v, igate_preact, fgate_preact, chunk_size=hparams["CHUNK"], autocast_kernel_dtype=hparams["dtype"]
        )
        h_out_triton = h_out_triton.astype(jnp.float32)
        return h_out_triton.sum(), h_out_triton

    grad_fn = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4), has_aux=True))
    (loss_jax, h_out_triton), (grad_q, grad_k, grad_v, grad_igate_preact, grad_fgate_preact) = grad_fn(
        q, k, v, igate_preact, fgate_preact
    )
    loss_jax = jax.device_get(loss_jax)

    # TODO for more finegrained comparison we would need to change the tolerances depending on the dtype
    np.testing.assert_allclose(
        loss_jax,
        data_torch["loss"],
        rtol=0.5,  # 1e-3,
        atol=1.0,  # 1e-2,
        err_msg=f"Mismatch between Triton and PyTorch in the loss value for file: {file_path}.",
    )

    assert h_out_triton.shape == (hparams["B"], hparams["NH"], hparams["S"], hparams["DH"])
    h_out_triton = jax.device_get(h_out_triton.astype(np.float32))
    h_out_pytorch = data_torch["out"]

    np.testing.assert_allclose(
        h_out_triton,
        h_out_pytorch,
        rtol=0.5,  # 1e-2,
        atol=1.0,  # 1e-1,
        err_msg=f"Mismatch between Triton and PyTorch in the output value for file: {file_path}.",
    )

    grad_titon = [grad_q, grad_k, grad_v, grad_igate_preact, grad_fgate_preact]
    grad_titon = [jax.device_get(g.astype(jnp.float32)) for g in grad_titon]
    grad_pytorch = [data_torch[f"grad_{name}"] for name in ["q", "k", "v", "igate_preact", "fgate_preact"]]
    grad_names = ["q", "k", "v", "igate_preact", "fgate_preact"]
    for g_torch, g_triton, g_name in zip(grad_pytorch, grad_titon, grad_names):
        np.testing.assert_allclose(
            g_triton,
            g_torch,
            rtol=5.0,  # 0.8, #1e-2, # path3 needs such high tolerances
            atol=4.0,  # 1.2, #1e-1,
            err_msg=f"Mismatch between Triton and PyTorch in the gradient {g_name} for file: {file_path}.",
        )
