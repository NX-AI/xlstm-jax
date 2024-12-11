#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

# pylint: disable=invalid-name
import jax
import jax.numpy as jnp
import numpy as np
import pytest

try:
    from xlstm_jax.kernels import mlstm_recurrent_step_triton_fused
    from xlstm_jax.kernels.mlstm_chunkwise.max_triton_fwbw_v3noslice._triton_fw import _mlstm_chunkwise_fw
    from xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.recurrent import recurrent_step_fw
except ImportError:
    mlstm_recurrent_step_triton_fused = None
    _mlstm_chunkwise_fw = None

from xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.recurrent import (
    mLSTMBackendRecurrent,
    mLSTMBackendRecurrentConfig,
    recurrent_sequence_fw,
)
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.recurrent_triton import (
    mLSTMBackendRecurrentTriton,
    mLSTMBackendRecurrentTritonConfig,
)


@pytest.mark.skipif(not pytest.triton_available, reason="Triton is not available.")
def test_mlstm_recurrent_fw_vs_chunkwise(
    default_qkvif: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
):
    """Test the forward pass with the recurrent kernel vs chunkwise."""
    q, k, v, igate_preact, fgate_preact = default_qkvif
    B, NH, S, DH = q.shape
    CHUNK_SIZE = 64

    h_out_chunkwise, _, _, (c_out_chunkwise, n_out_chunkwise, m_out_chunkwise), _ = _mlstm_chunkwise_fw(
        matQ=q,
        matK=k,
        matV=v,
        vecI=igate_preact,
        vecF=fgate_preact,
        CHUNK_SIZE=CHUNK_SIZE,
        return_last_states=True,
        EPS=1e-6,
    )
    assert h_out_chunkwise.shape == (B, NH, S, DH)
    assert c_out_chunkwise.shape == (B, NH, DH, DH)
    assert n_out_chunkwise.shape == (B, NH, DH)
    assert m_out_chunkwise.shape == (B, NH)

    h_out_recurrent, (c_out_recurrent, n_out_recurrent, m_out_recurrent) = recurrent_sequence_fw(
        mlstm_step_fn=mlstm_recurrent_step_triton_fused,
        queries=q,
        keys=k,
        values=v,
        igate_preact=igate_preact,
        fgate_preact=fgate_preact,
        return_last_states=True,
        eps=1e-6,
    )
    assert h_out_recurrent.shape == (B, NH, S, DH)
    assert c_out_recurrent.shape == (B, NH, DH, DH)
    assert n_out_recurrent.shape == (B, NH, DH)
    assert m_out_recurrent.shape == (B, NH)

    h_out_chunkwise = jax.device_get(h_out_chunkwise.astype(jnp.float32))
    h_out_recurrent = jax.device_get(h_out_recurrent.astype(jnp.float32))

    np.testing.assert_allclose(
        h_out_chunkwise,
        h_out_recurrent,
        atol=1e-2,
        rtol=1e-2,
        err_msg="Mismatch between chunkwise and recurrent kernel.",
    )

    np.testing.assert_allclose(
        c_out_chunkwise,
        c_out_recurrent,
        atol=1e-2,
        rtol=1e-2,
        err_msg="Mismatch between chunkwise and recurrent kernel.",
    )

    np.testing.assert_allclose(
        n_out_chunkwise,
        n_out_recurrent,
        atol=1e-3,
        rtol=1e-3,
        err_msg="Mismatch between chunkwise and recurrent kernel.",
    )

    np.testing.assert_allclose(
        m_out_chunkwise,
        m_out_recurrent,
        atol=1e-3,
        rtol=1e-3,
        err_msg="Mismatch between chunkwise and recurrent kernel.",
    )


@pytest.mark.skipif(not pytest.triton_available, reason="Triton is not available.")
def test_mlstm_recurrent_sequence_fw_vs_chunkwise(
    default_qkvif: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
):
    """Test the forward pass with the recurrent kernel vs chunkwise."""
    q, k, v, igate_preact, fgate_preact = default_qkvif
    B, NH, S, DH = q.shape

    h_out_recurrent_j, (c_out_recurrent_j, n_out_recurrent_j, m_out_recurrent_j) = recurrent_sequence_fw(
        mlstm_step_fn=recurrent_step_fw,
        queries=q,
        keys=k,
        values=v,
        igate_preact=igate_preact,
        fgate_preact=fgate_preact,
        return_last_states=True,
        eps=0.0,
    )
    assert h_out_recurrent_j.shape == (B, NH, S, DH)
    assert c_out_recurrent_j.shape == (B, NH, DH, DH)
    assert n_out_recurrent_j.shape == (B, NH, DH)
    assert m_out_recurrent_j.shape == (B, NH)

    h_out_recurrent_tr, (c_out_recurrent_tr, n_out_recurrent_tr, m_out_recurrent_tr) = recurrent_sequence_fw(
        mlstm_step_fn=mlstm_recurrent_step_triton_fused,
        queries=q,
        keys=k,
        values=v,
        igate_preact=igate_preact,
        fgate_preact=fgate_preact,
        return_last_states=True,
        eps=0.0,
    )
    assert h_out_recurrent_tr.shape == (B, NH, S, DH)
    assert c_out_recurrent_tr.shape == (B, NH, DH, DH)
    assert n_out_recurrent_tr.shape == (B, NH, DH)
    assert m_out_recurrent_tr.shape == (B, NH)

    # h_out_chunkwise = jax.nn.standardize(h_out_chunkwise, axis=-1)
    # h_out_recurrent_tr = jax.nn.standardize(h_out_recurrent_tr, axis=-1)

    h_out_recurrent_j = jax.device_get(h_out_recurrent_j.astype(jnp.float32))
    h_out_recurrent_tr = jax.device_get(h_out_recurrent_tr.astype(jnp.float32))

    np.testing.assert_allclose(
        h_out_recurrent_j,
        h_out_recurrent_tr,
        atol=1e-3,
        rtol=1e-3,
        err_msg="Mismatch between chunkwise and recurrent kernel.",
    )


@pytest.mark.skipif(not pytest.triton_available, reason="Triton is not available.")
def test_mlstm_recurrent_step_triton_vs_native_jax(
    default_qkvif: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
):
    """Test the forward pass with the kernels versus native-JAX backends."""
    q, k, v, igate_preact, fgate_preact = default_qkvif
    B, NH, S, DH = q.shape
    print("Q", q.shape, q.dtype)
    print("K", k.shape, k.dtype)
    print("V", v.shape, v.dtype)
    print("I", igate_preact.shape, igate_preact.dtype)
    print("F", fgate_preact.shape, fgate_preact.dtype)

    config_jax = mLSTMBackendRecurrentConfig(eps=1e-6)
    backend_jax = mLSTMBackendRecurrent(config_jax)
    h_out_jax, (c_out_jax, n_out_jax, m_out_jax) = backend_jax(
        q,
        k,
        v,
        igate_preact,
        fgate_preact,
        return_last_states=True,
    )
    assert h_out_jax.shape == (B, NH, S, DH)
    assert c_out_jax.shape == (B, NH, DH, DH)
    assert n_out_jax.shape == (B, NH, DH)
    assert m_out_jax.shape == (B, NH)

    config_triton = mLSTMBackendRecurrentTritonConfig(eps=1e-6)
    backend_triton = mLSTMBackendRecurrentTriton(config_triton)
    h_out_triton, (c_out_triton, n_out_triton, m_out_triton) = backend_triton(
        q,
        k,
        v,
        igate_preact,
        fgate_preact,
        return_last_states=True,
    )
    assert h_out_triton.shape == (B, NH, S, DH)
    assert c_out_triton.shape == (B, NH, DH, DH)
    assert n_out_triton.shape == (B, NH, DH)
    assert m_out_triton.shape == (B, NH)

    h_out_jax = jax.device_get(h_out_jax.astype(jnp.float32))
    h_out_triton = jax.device_get(h_out_triton.astype(jnp.float32))

    for name, tensor_jax, tensor_triton in zip(
        ["h", "c", "n", "m"],
        [h_out_jax, c_out_jax, n_out_jax, m_out_jax],
        [h_out_triton, c_out_triton, n_out_triton, m_out_triton],
    ):
        np.testing.assert_allclose(
            tensor_jax,
            tensor_triton,
            atol=1e-3,
            rtol=1e-3,
            err_msg=f"Mismatch between JAX and Triton for {name}.",
        )
