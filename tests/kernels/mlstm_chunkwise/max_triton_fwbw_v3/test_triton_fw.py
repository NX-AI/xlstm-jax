import jax
import jax.numpy as jnp
import numpy as np
import pytest

try:
    from xlstm_jax.kernels.mlstm_chunkwise.max_triton_fwbw_v3._triton_fw import _mlstm_chunkwise_fw
except ImportError:
    _mlstm_chunkwise_fw = None
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.fwbw import (
    mlstm_fwbw_custom_grad as mlstm_fwbw_custom_grad_jax,
    mLSTMBackendFwbwConfig as mLSTMfwbwConfig_jax,
)
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.simple import (
    parallel_stabilized_simple as parallel_stabilized_simple_jax,
)


@pytest.mark.skipif(not pytest.triton_available, reason="Triton is not available.")
def test_mlstm_chunkwise_fw_vs_jax_backends(
    default_qkvif: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
):
    """Test the forward pass with the kernels versus native-JAX backends."""
    q, k, v, igate_preact, fgate_preact = default_qkvif
    B, NH, S, DH = q.shape
    CHUNK_SIZE = 64

    h_out_triton, n_out, m_out, _, _ = _mlstm_chunkwise_fw(
        matQ=q, matK=k, matV=v, vecI=igate_preact, vecF=fgate_preact, CHUNK_SIZE=CHUNK_SIZE
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
