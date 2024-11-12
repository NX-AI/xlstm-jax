from mlstm_kernels.mlstm_kernels.kernel_utils import is_power_of_2
from mlstm_kernels.mlstm_kernels.mlstm.chunkwise.max_triton_fwbw_v5xlchunksize._triton_recurrent_bw import (
    _mlstm_chunkwise__recurrent_bw_dC_kernel,
)

import jax
import jax.numpy as jnp
import jax_triton as jt
import triton

from xlstm_jax.kernels.kernel_utils import jax2triton_dtype
from xlstm_jax.kernels.stride_utils import get_stride


def mlstm_chunkwise__recurrent_bw_dC(
    matQ: jax.Array,  # (B, NH, S, DHQK)
    vecF: jax.Array,  # (B, NH, NC * L) = (B, NH, S)
    scaM_inter: jax.Array,  # (B, NH, NC+1)
    vecM_combine: jax.Array,  # (B, NH, S)
    matDeltaH: jax.Array,  # (B, NH, S, DHHV)
    vecN_out: jax.Array,  # (B, NH, S)
    matDeltaC_last: jax.Array | None = None,  # (B, NH, DHQK, DHHV)
    qk_scale: float | None = None,
    chunk_size: int = 64,
    save_states_every_nth_chunk: int = 1,
    num_warps: int | None = None,
    num_stages: int | None = None,
    eps: float = 0.0,
) -> jax.Array:  # matDeltaC_states (B, NH, (NC+1) * DHQK, DHHV)
    """
    Computes only the deltaC gradients for the backward pass.

    The other gradients are computed in the other (kernel) function.
    We do not need to compute the gradients for the denominator, as it cancels out in the forward in the groupnorm.

    Args:
        matQ: Tensor containing the query vectors. Shape (B, NH, S, DHQK).
        vecF: Tensor containing theforget gate pre-activations. Shape (B, NH, NC * L) = (B, NH, S).
        scaM_inter: States of the M scalar. Shape (B, NH, NC+1).
        vecM_combine: Combined M states. Shape (B, NH, S).
        matDeltaH: Tensor containing the H gradients. Shape (B, NH, S, DHHV).
        vecN_out: States of the N vector. Shape (B, NH, NC * DHQK).
        matDeltaC_last: Tensor containing the last C gradients. Shape (B, NH, DHQK, DHHV).
            Defaults to None.
        qk_scale: Scale factor for the QK matrix. Defaults to None.
        chunk_size: Chunk size. Defaults to 64.
        save_states_every_nth_chunk: Save the states every nth chunk. Defaults to 1.
        num_warps: Number of warps. Defaults to None.
        num_stages: Number of stages. Defaults to None.
        eps: Epsilon value. Defaults to 1e-6.


    Returns:
        Tensor containing the C gradients. Shape (B, NH, (NC+1) * DHQK, DHHV).
    """
    B, NH, S, DHQK, DHHV = *matQ.shape, matDeltaH.shape[-1]
    L = chunk_size
    assert is_power_of_2(L), "Chunk size must be a power of 2."
    assert S % L == 0, "S must be divisible by chunk_size."
    NC = S // L

    assert save_states_every_nth_chunk > 0, "save_states_every_nth_chunk must be positive."
    assert save_states_every_nth_chunk <= NC, "save_states_every_nth_chunk must be <= NC."

    assert is_power_of_2(
        save_states_every_nth_chunk
    ), f"save_states_every_nth_chunk must be a power of 2. Got {save_states_every_nth_chunk}."

    if qk_scale is None:
        qk_scale = DHQK**-0.5

    USE_LAST_STATE = matDeltaC_last is not None

    num_chunks_saved = NC // save_states_every_nth_chunk

    matDeltaC_states = jax.ShapeDtypeStruct(
        shape=(B, NH, (num_chunks_saved + 1) * DHQK, DHHV),
        dtype=jnp.float32,
    )
    matDeltaC_states = jnp.empty_like(matDeltaC_states)

    siz_b_DHQK = min(64, triton.next_power_of_2(DHQK))
    siz_b_DHHV = min(64, triton.next_power_of_2(DHHV))

    num_b_DHQK = triton.cdiv(DHQK, siz_b_DHQK)
    num_b_DHHV = triton.cdiv(DHHV, siz_b_DHHV)

    num_stages = 1 if num_stages is None else num_stages
    if num_warps is None:
        num_warps = 4 if siz_b_DHQK == 64 else 2

    grid = (num_b_DHQK, num_b_DHHV, B * NH)
    matDeltaC_states = jt.triton_call(
        matQ,
        vecF,
        scaM_inter,
        vecM_combine,
        matDeltaH,
        vecN_out,
        matDeltaC_last,
        matDeltaC_states,
        out_shape=(matDeltaC_states),
        qk_scale=qk_scale,
        str_matQ_B_NH=get_stride(matQ, axis=1),
        str_matQ_S=get_stride(matQ, axis=2),
        str_matQ_DHQK=get_stride(matQ, axis=3),
        str_vecF_B_NH=get_stride(vecF, axis=1),
        str_scaM_inter_B_NH=get_stride(scaM_inter, axis=1),
        str_scaM_inter_NC=get_stride(scaM_inter, axis=2),
        str_vecM_combine_B_NH=get_stride(vecM_combine, axis=1),
        str_vecM_combine_S=get_stride(vecM_combine, axis=2),
        str_matDeltaH_B_NH=get_stride(matDeltaH, axis=1),
        str_matDeltaH_S=get_stride(matDeltaH, axis=2),
        str_matDeltaH_DHHV=get_stride(matDeltaH, axis=3),
        str_vecN_out_B_NH=get_stride(vecN_out, axis=1),
        str_vecN_out_S=get_stride(vecN_out, axis=2),
        str_matDeltaC_last_B_NH=get_stride(matDeltaC_last, axis=1) if USE_LAST_STATE else 0,
        str_matDeltaC_last_DHQK=get_stride(matDeltaC_last, axis=2) if USE_LAST_STATE else 0,
        str_matDeltaC_last_DHHV=get_stride(matDeltaC_last, axis=3) if USE_LAST_STATE else 0,
        str_matDeltaC_states_B_NH=get_stride(matDeltaC_states, axis=1),
        str_matDeltaC_states_NCDHQK=get_stride(matDeltaC_states, axis=2),
        str_matDeltaC_states_DHHV=get_stride(matDeltaC_states, axis=3),
        B=B,
        NH=NH,
        S=S,
        DHQK=DHQK,
        DHHV=DHHV,
        NC=NC,
        L=L,
        siz_b_DHQK=siz_b_DHQK,
        siz_b_DHHV=siz_b_DHHV,
        save_states_every_nth_chunk=save_states_every_nth_chunk,
        USE_LAST_STATE=USE_LAST_STATE,
        DTYPE=jax2triton_dtype(matQ.dtype),
        EPS=eps,
        num_stages=num_stages,
        num_warps=num_warps,
        grid=grid,
        kernel=_mlstm_chunkwise__recurrent_bw_dC_kernel,
    )

    return matDeltaC_states
