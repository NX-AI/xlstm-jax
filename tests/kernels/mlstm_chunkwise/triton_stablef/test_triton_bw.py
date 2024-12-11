#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import jax
import pytest

try:
    from xlstm_jax.kernels.mlstm_chunkwise.triton_stablef._triton_bw import _mlstm_chunkwise_bw
    from xlstm_jax.kernels.mlstm_chunkwise.triton_stablef._triton_fw import _mlstm_chunkwise_fw
except ImportError:
    _mlstm_chunkwise_fw = None
    _mlstm_chunkwise_bw = None


@pytest.mark.skipif(not pytest.triton_available, reason="Triton is not available.")
def test_mlstm_chunkwise_bw(default_qkvif: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]) -> None:
    """Test a simple backward pass with the Triton kernels."""
    q, k, v, igate_preact, fgate_preact = default_qkvif
    B, NH, S, DH = q.shape
    K, V = k.shape[-1], v.shape[-1]
    CHUNK_SIZE = 64

    h_out_triton, n_out, m_out, _, (matC_states, _, scaM_states) = _mlstm_chunkwise_fw(
        matQ=q, matK=k, matV=v, vecI=igate_preact, vecF=fgate_preact, chunk_size=CHUNK_SIZE, return_all_states=True
    )
    assert h_out_triton.shape == (B, NH, S, DH)
    assert n_out.shape == (B, NH, S)
    assert m_out.shape == (B, NH, S)
    assert matC_states.shape == (B, NH, S // CHUNK_SIZE * K, V)
    assert scaM_states.shape == (B, NH, S // CHUNK_SIZE + 1)

    grads = _mlstm_chunkwise_bw(
        matQ=q,
        matK=k,
        matV=v,
        vecI=igate_preact,
        vecF=fgate_preact,
        matC_initial=None,
        vecN_initial=None,
        scaM_initial=None,
        matC_states=matC_states,
        scaM_states=scaM_states,
        vecN_out=n_out,
        vecM_out=m_out,
        matDeltaH=h_out_triton,
        matDeltaC_last=None,
        chunk_size=CHUNK_SIZE,
    )
    (
        grad_q,
        grad_k,
        grad_v,
        grad_igate_preact,
        grad_fgate_preact,
        _,
        _,
        _,
    ) = grads
    assert grad_q.shape == q.shape
    assert grad_k.shape == k.shape
    assert grad_v.shape == v.shape
    assert grad_igate_preact.shape == igate_preact.shape
    assert grad_fgate_preact.shape == fgate_preact.shape
