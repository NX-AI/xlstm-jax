# pylint: disable=invalid-name
import jax
import pytest

try:
    from xlstm_jax.kernels.mlstm_chunkwise.max_triton_fwbw_v3._triton_bw import _mlstm_chunkwise_bw
    from xlstm_jax.kernels.mlstm_chunkwise.max_triton_fwbw_v3._triton_fw import _mlstm_chunkwise_fw
    from xlstm_jax.kernels.mlstm_chunkwise.max_triton_fwbw_v3noslice._triton_bw import (
        _mlstm_chunkwise_bw as _mlstm_chunkwise_bw_noslice,
    )
    from xlstm_jax.kernels.mlstm_chunkwise.max_triton_fwbw_v3noslice._triton_fw import (
        _mlstm_chunkwise_fw as _mlstm_chunkwise_fw_noslice,
    )
except ImportError:
    _mlstm_chunkwise_fw = None
    _mlstm_chunkwise_bw = None
    _mlstm_chunkwise_fw_noslice = None
    _mlstm_chunkwise_bw_noslice = None


@pytest.mark.skipif(not pytest.triton_available, reason="Triton is not available.")
@pytest.mark.parametrize("chunkwise_fw", [_mlstm_chunkwise_fw, _mlstm_chunkwise_fw_noslice])
@pytest.mark.parametrize("chunkwise_bw", [_mlstm_chunkwise_bw, _mlstm_chunkwise_bw_noslice])
def test_mlstm_chunkwise_bw(
    default_qkvif: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    chunkwise_fw: callable,
    chunkwise_bw: callable,
) -> None:
    """Test a simple backward pass with the Triton kernels."""
    q, k, v, igate_preact, fgate_preact = default_qkvif
    B, NH, S, DH = q.shape
    CHUNK_SIZE = 64

    h_out_triton, n_out, m_out, _, _ = chunkwise_fw(
        matQ=q, matK=k, matV=v, vecI=igate_preact, vecF=fgate_preact, CHUNK_SIZE=CHUNK_SIZE
    )
    assert h_out_triton.shape == (B, NH, S, DH)
    assert n_out.shape == (B, NH, S)
    assert m_out.shape == (B, NH, S)

    grads = chunkwise_bw(
        matQ=q,
        matK=k,
        matV=v,
        vecI=igate_preact,
        vecF=fgate_preact,
        matC_initial=None,
        vecN_initial=None,
        scaM_initial=None,
        matC_all=None,
        vecN_all=None,
        scaM_all=None,
        vecN_out=n_out,
        vecM_out=m_out,
        matDeltaH=h_out_triton,
        matDeltaC_last=None,
        CHUNK_SIZE=CHUNK_SIZE,
        EPS=1e-6,
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
