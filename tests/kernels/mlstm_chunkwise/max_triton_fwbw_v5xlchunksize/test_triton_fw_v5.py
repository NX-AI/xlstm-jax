# pylint: disable=invalid-name
import jax
import jax.numpy as jnp
import numpy as np
import pytest

try:
    from xlstm_jax.kernels.mlstm_chunkwise.max_triton_fwbw_v5xlchunksize._combined_fw import mlstm_chunkwise_fw
    from xlstm_jax.kernels.mlstm_chunkwise.max_triton_fwbw_v5xlchunksize._recurrent_fw import (
        mlstm_chunkwise__recurrent_fw_C,
    )
except ImportError:
    mlstm_chunkwise_fw = None
    mlstm_chunkwise__recurrent_fw_C = None
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.fwbw import (
    mlstm_fwbw_custom_grad as mlstm_fwbw_custom_grad_jax,
    mLSTMBackendFwbwConfig as mLSTMfwbwConfig_jax,
)
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.simple import (
    parallel_stabilized_simple as parallel_stabilized_simple_jax,
)


@pytest.mark.skipif(not pytest.triton_available, reason="Triton is not available.")
@pytest.mark.parametrize("kernel", [mlstm_chunkwise_fw])
def test_mlstm_chunkwise_fw_vs_jax_backends(
    default_qkvif: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], kernel: callable
):
    """Test the forward pass with the kernels versus native-JAX backends."""
    q, k, v, igate_preact, fgate_preact = default_qkvif
    B, NH, S, DH = q.shape
    CHUNK_SIZE = 64

    h_out_triton, n_out, m_out, _, _ = kernel(
        matQ=q,
        matK=k,
        matV=v,
        vecI=igate_preact,
        vecF=fgate_preact,
        chunk_size_inter=64,
        chunk_size_intra=64,
        eps=1e-6,
    )
    assert h_out_triton.shape == (B, NH, S, DH)
    assert n_out.shape == (B, NH, S)
    assert m_out.shape == (B, NH, S)

    backend_parallel = parallel_stabilized_simple_jax
    backend_parallel = jax.vmap(backend_parallel, in_axes=(1, 1, 1, 1, 1), out_axes=1)
    h_out_jax_parallel = backend_parallel(q, k, v, igate_preact[..., None], fgate_preact[..., None])
    assert h_out_jax_parallel.shape == (B, NH, S, DH)

    fwbw_config = mLSTMfwbwConfig_jax(
        chunk_size=CHUNK_SIZE,
        stabilize_correctly=True,
    )
    backend = mlstm_fwbw_custom_grad_jax(fwbw_config)
    backend = jax.vmap(backend, in_axes=(1, 1, 1, 1, 1), out_axes=1)
    h_out_jax_fwbw = backend(
        q,
        k,
        v,
        igate_preact[..., None],
        fgate_preact[..., None],
    )
    assert h_out_jax_fwbw.shape == (B, NH, S, DH)

    h_out_triton = jax.nn.standardize(h_out_triton, axis=-1)
    h_out_jax_fwbw = jax.nn.standardize(h_out_jax_fwbw, axis=-1)
    h_out_jax_parallel = jax.nn.standardize(h_out_jax_parallel, axis=-1)

    h_out_triton = jax.device_get(h_out_triton.astype(jnp.float32))
    h_out_jax_fwbw = jax.device_get(h_out_jax_fwbw.astype(jnp.float32))
    h_out_jax_parallel = jax.device_get(h_out_jax_parallel.astype(jnp.float32))

    np.testing.assert_allclose(
        h_out_jax_fwbw, h_out_jax_parallel, atol=1e-2, rtol=1e-2, err_msg="Mismatch between JAX fwbw and JAX parallel."
    )
    np.testing.assert_allclose(
        h_out_triton, h_out_jax_fwbw, atol=1e-2, rtol=1e-2, err_msg="Mismatch between Triton and JAX fwbw."
    )
    np.testing.assert_allclose(
        h_out_triton, h_out_jax_parallel, atol=1e-2, rtol=1e-2, err_msg="Mismatch between Triton and JAX parallel."
    )


@pytest.mark.skipif(not pytest.triton_available, reason="Triton is not available.")
@pytest.mark.parametrize("kernel", [mlstm_chunkwise__recurrent_fw_C])
def test_mlstm_chunkwise_fw_C_w_init_state_vs_without_init_state(
    default_qkvif: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], kernel: callable
):
    """Test the forward pass with the kernels with inital state init as zero vs. no initial state."""
    q, k, v, igate_preact, fgate_preact = default_qkvif
    B, NH, S, DH = q.shape
    L = 64
    NC = S // L

    h_out_triton, n_out, m_out = kernel(
        matK=k,
        matV=v,
        vecI=igate_preact,
        vecF=fgate_preact,
        chunk_size=L,
    )
    assert h_out_triton.shape == (B, NH, (NC + 1) * DH, DH)
    assert n_out.shape == (B, NH, (NC + 1) * DH)
    assert m_out.shape == (B, NH, (NC + 1))

    c_initial = np.zeros((B, NH, DH, DH), dtype=np.float32)
    n_initial = np.zeros((B, NH, DH), dtype=np.float32)
    m_initial = np.zeros((B, NH), dtype=np.float32)

    c_initial = jnp.array(c_initial)
    n_initial = jnp.array(n_initial)
    m_initial = jnp.array(m_initial)

    h_out_triton2, n_out2, m_out2 = kernel(
        matK=k,
        matV=v,
        vecI=igate_preact,
        vecF=fgate_preact,
        matC_initial=c_initial,
        vecN_initial=n_initial,
        scaMinter_initial=m_initial,
        chunk_size=L,
    )
    assert h_out_triton2.shape == (B, NH, (NC + 1) * DH, DH)
    assert n_out2.shape == (B, NH, (NC + 1) * DH)
    assert m_out2.shape == (B, NH, (NC + 1))

    np.testing.assert_allclose(
        h_out_triton,
        h_out_triton2,
        atol=1e-4,
        rtol=1e-2,
        err_msg="Mismatch between with initial state and without initial state.",
    )
    np.testing.assert_allclose(
        n_out, n_out2, atol=1e-4, rtol=1e-2, err_msg="Mismatch between with initial state and without initial state."
    )
    np.testing.assert_allclose(
        m_out, m_out2, atol=1e-4, rtol=1e-2, err_msg="Mismatch between with initial state and without initial state."
    )
