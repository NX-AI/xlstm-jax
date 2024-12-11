#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

"""Triton backend for the backward pass of the mLSTM chunkwise formulation.

This file has been adapted from the original PyTorch Triton implementation to JAX.
For the Triton kernels, see mlstm_kernels/mlstm_kernels/mlstm/chunkwise/max_triton_fwbw_v3/_triton_bw.py.

In this file, we use the following notation:

Dimensions:
    B: batch size
    NH: number of heads
    S: sequence length (K, V)
    T: sequence length (Q)
    DHQK: hidden dimension (Q, K)
    DHHV: hidden dimension (H, V)
    NC: number of chunks
    L: chunk size

Variables:
    vecA, a: forward gate contribution, contribution of forget gates from last chunk state C_{k-1} to
            current timestep t
    vecB, b: backward gate contribution, contribution of forget and input gates up to next chunk
            state C_k (form current timestep t)
    scaG, g: "go through" gate contribution, contribution of forget gates from C_{k-1} to C_k.
    matD, D: gating matrix for the parallel form.
"""

import jax
import jax.numpy as jnp
import jax_triton as jt
import triton

from mlstm_kernels.mlstm_kernels.kernel_utils import is_power_of_2
from mlstm_kernels.mlstm_kernels.mlstm.chunkwise.max_triton_fwbw_v3noslice._triton_bw import (
    _mlstm_chunkwise__parallel_bw_dQKV_kernel,
    _mlstm_chunkwise__recurrent_bw_dC_kernel,
)

from xlstm_jax.kernels.kernel_utils import jax2triton_dtype
from xlstm_jax.kernels.stride_utils import get_stride

from ._triton_fw import _mlstm_chunkwise__recurrent_fw_C


def _mlstm_chunkwise__recurrent_bw_dC(
    matQ: jax.Array,  # (B, NH, S, DHQK)
    vecB: jax.Array,  # (B, NH, NC, L)
    scaM_inter: jax.Array,  # (B, NH, NC+1)
    vecM_combine: jax.Array,  # (B, NH, S)
    matDeltaH: jax.Array,  # (B, NH, S, DHHV)
    vecN_out: jax.Array,  # (B, NH, S)
    matDeltaC_last: jax.Array | None = None,  # (B, NH, DHQK, DHHV)
    qk_scale: float | None = None,
    CHUNK_SIZE: int = 64,
    NUM_CHUNKS: int = 1,
    EPS: float = 1e-6,
) -> jax.Array:  # matDeltaC_states (B, NH, (NC+1) * DHQK, DHHV)
    """
    Computes only the deltaC gradients for the backward pass.

    The other gradients are computed in the other (kernel) function.
    We do not need to compute the gradients for the denominator, as it cancels out in the forward in the groupnorm.

    Args:
        matQ: Tensor containing the query vectors. Shape (B, NH, S, DHQK).
        vecB: Tensor containing the summed log forget gate activations. Shape (B, NH, NC, L).
        scaM_inter: States of the M scalar. Shape (B, NH, NC+1).
        vecM_combine: Combined M states. Shape (B, NH, S).
        matDeltaH: Tensor containing the H gradients. Shape (B, NH, S, DHHV).
        vecN_out: States of the N vector. Shape (B, NH, NC * DHQK).
        matDeltaC_last: Tensor containing the last C gradients. Shape (B, NH, DHQK, DHHV).
            Defaults to None.
        qk_scale: Scale factor for the QK matrix. Defaults to None.
        CHUNK_SIZE: Chunk size. Defaults to 64.
        NUM_CHUNKS: Number of chunks. Defaults to 1.
        EPS: Epsilon value. Defaults to 1e-6.

    Returns:
        Tensor containing the C gradients. Shape (B, NH, (NC+1) * DHQK, DHHV).
    """
    B, NH, S, DHQK, DHHV = *matQ.shape, matDeltaH.shape[-1]
    NC = NUM_CHUNKS
    L = CHUNK_SIZE
    _dtype = matQ.dtype

    assert NC == vecB.shape[2], "Number of chunks must match the number in vecB."
    assert L == vecB.shape[3], "Chunk size must match the chunk size in vecB."
    assert is_power_of_2(L), "Chunk size must be a power of 2."

    if qk_scale is None:
        qk_scale = DHQK**-0.5

    USE_LAST_STATE = matDeltaC_last is not None

    matDeltaC_states = jax.ShapeDtypeStruct(shape=(B, NH, (NC + 1) * DHQK, DHHV), dtype=jnp.float32)
    if matDeltaC_last is None:
        matDeltaC_last = jnp.zeros((1,), dtype=_dtype)

    siz_b_DHQK = min(64, triton.next_power_of_2(DHQK))
    siz_b_DHHV = min(64, triton.next_power_of_2(DHHV))

    num_b_DHQK = triton.cdiv(DHQK, siz_b_DHQK)
    num_b_DHHV = triton.cdiv(DHHV, siz_b_DHHV)

    num_stages = 1
    num_warps = 4 if siz_b_DHQK == 64 else 2

    grid = (num_b_DHQK, num_b_DHHV, B * NH)
    matDeltaC_states = jt.triton_call(
        matQ,
        vecB,
        scaM_inter,
        vecM_combine,
        matDeltaH,
        vecN_out,
        matDeltaC_last,
        out_shape=matDeltaC_states,
        qk_scale=qk_scale,
        str_matQ_B_NH=get_stride(matQ, axis=1),
        str_matQ_S=get_stride(matQ, axis=2),
        str_matQ_DHQK=get_stride(matQ, axis=3),
        str_vecB_B_NH=get_stride(vecB, axis=1),
        str_vecB_NC=get_stride(vecB, axis=2),
        str_vecB_L=get_stride(vecB, axis=3),
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
        USE_LAST_STATE=USE_LAST_STATE,
        DTYPE=jax2triton_dtype(_dtype),
        EPS=EPS,
        num_stages=num_stages,
        num_warps=num_warps,
        grid=grid,
        kernel=_mlstm_chunkwise__recurrent_bw_dC_kernel,
    )

    return matDeltaC_states


def _mlstm_chunkwise__parallel_bw_dQKV(
    matQ: jax.Array,  # (B, NH, S, DHQK)
    matK: jax.Array,  # (B, NH, S, DHQK)
    matV: jax.Array,  # (B, NH, S, DHHV)
    vecB: jax.Array,  # (B, NH, NC, L)
    vecI: jax.Array,  # (B, NH, NC, L)
    vecM_combine: jax.Array,  # (B, NH, S) = (B, NH, NC * L)
    scaM_inter: jax.Array,  # (B, NH, NC+1)
    matC_states: jax.Array,  # (B, NH, (NC+1) * DHQK, DHHV)
    matDeltaH: jax.Array,  # (B, NH, S, DHHV)
    vecN_out: jax.Array,  # (B, NH, S)
    matDeltaC_states: jax.Array,  # (B, NH, (NC+1) * DHQK, DHHV)
    qk_scale: float | None = None,
    CHUNK_SIZE: int = 64,
    NUM_CHUNKS: int = 1,
    EPS: float = 1e-6,
) -> tuple[
    jax.Array, jax.Array, jax.Array
]:  # matDeltaQ (B,NH,S,DHQK), matDeltaK (B,NH,S,DHQK), matDeltaV (B,NH,S,DHHV)
    """
    Computes the gradients for the query, key and value matrices.

    Args:
        matQ: Tensor containing the query vectors. Shape (B, NH, S, DHQK).
        matK: Tensor containing the key vectors. Shape (B, NH, S, DHQK).
        matV: Tensor containing the value vectors. Shape (B, NH, S, DHHV).
        vecB: Tensor containing the summed log forget gate activations. Shape (B, NH, NC, L).
        vecI: Tensor containing the input gate pre-activations. Shape (B, NH, NC, L).
        vecM_combine: Combined M states. Shape (B, NH, S).
        scaM_inter: States of the M scalar. Shape (B, NH, NC+1).
        matC_states: States of the C matrix. Shape (B, NH, NC * DHQK, DHHV).
        matDeltaH: Tensor containing the H gradients. Shape (B, NH, S, DHHV).
        vecN_out: States of the N vector. Shape (B, NH, S).
        matDeltaC_states: Tensor containing the C gradients. Shape (B, NH, (NC+1) * DHQK, DHHV).
        qk_scale: Scale factor for the QK matrix. Defaults to None.
        CHUNK_SIZE (int, optional): Chunk size. Defaults to 64.
        NUM_CHUNKS (int, optional): Number of chunks. Defaults to 1.
        EPS: Epsilon value. Defaults to 1e-6.

    Returns:
        Gradients for the query, key and value matrices. Shapes (B, NH, S, DHQK), (B, NH, S, DHQK), (B, NH, S, DHHV).
    """
    B, NH, S, DHQK, DHHV = *matQ.shape, matV.shape[-1]
    NC = NUM_CHUNKS
    L = CHUNK_SIZE
    _dtype = matQ.dtype

    assert NC == vecB.shape[2], "Number of chunks must match the number in vecB."
    assert L == vecB.shape[3], "Chunk size must match the chunk size in vecB."
    assert is_power_of_2(L), "Chunk size must be a power of 2."

    if qk_scale is None:
        qk_scale = DHQK**-0.5

    siz_b_DHQK = min(32 if str(_dtype) == "float32" else 64, triton.next_power_of_2(DHQK))
    siz_b_DHHV = min(32 if str(_dtype) == "float32" else 64, triton.next_power_of_2(DHHV))

    num_b_DHQK = triton.cdiv(DHQK, siz_b_DHQK)

    num_stages = 1
    num_warps = 4 if siz_b_DHQK == 64 else 2

    # Specify output shapes.
    matDeltaQ = jax.ShapeDtypeStruct(shape=(B, NH, S, DHQK), dtype=_dtype)
    matDeltaK = jax.ShapeDtypeStruct(shape=(B, NH, S, DHQK), dtype=_dtype)
    # each b_DHQK thread block computes the contribution of its siz_b_DHQK block of matDeltaC
    # we need to sum them up to get the final result (we do this outside the kernel)
    matDeltaV = jax.ShapeDtypeStruct(shape=(num_b_DHQK, B, NH, S, DHHV), dtype=_dtype)

    # Define the grid and call the triton kernel.
    grid = (num_b_DHQK, NC, B * NH)
    matDeltaQ, matDeltaK, matDeltaV = jt.triton_call(
        matQ,  # (B, NH, S, DHQK)
        matK,  # (B, NH, S, DHQK)
        matV,  # (B, NH, S, DHHV)
        vecB,  # (B, NH, NC, L)
        vecI,  # (B, NH, NC, L)
        vecM_combine,  # (B, NH, S)
        scaM_inter,  # (B, NH, NC+1)
        matC_states,  # (B, NH, (NC+1) * DHQK, DHHV)
        matDeltaH,  # (B, NH, S, DHHV)
        vecN_out,  # (B, NH, S)
        matDeltaC_states,  # (B, NH, (NC+1) * DHQK, DHHV)
        out_shape=(matDeltaQ, matDeltaK, matDeltaV),
        qk_scale=qk_scale,
        str_matQK_B_NH=get_stride(matQ, axis=1),
        str_matQK_S=get_stride(matQ, axis=2),
        str_matQK_DHQK=get_stride(matQ, axis=3),
        str_matDV_num_b_DHQK=get_stride(matDeltaV, axis=0),
        str_matHV_B_NH=get_stride(matV, axis=1),
        str_matHV_S=get_stride(matV, axis=2),
        str_matHV_DHHV=get_stride(matV, axis=3),
        str_vecBI_B_NH=get_stride(vecI, axis=1),
        str_vecBI_NC=get_stride(vecI, axis=2),
        str_vecBI_L=get_stride(vecI, axis=3),
        str_vecM_combine_B_NH=get_stride(vecM_combine, axis=1),
        str_vecM_combine_S=get_stride(vecM_combine, axis=2),
        str_scaM_inter_B_NH=get_stride(scaM_inter, axis=1),
        str_scaM_inter_NC=get_stride(scaM_inter, axis=2),
        str_matC_states_B_NH=get_stride(matC_states, axis=1),
        str_matC_states_NCDHQK=get_stride(matC_states, axis=2),
        str_matC_states_DHHV=get_stride(matC_states, axis=3),
        str_vecN_out_B_NH=get_stride(vecN_out, axis=1),
        str_vecN_out_S=get_stride(vecN_out, axis=2),
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
        DTYPE=jax2triton_dtype(_dtype),
        EPS=EPS,
        num_stages=num_stages,
        num_warps=num_warps,
        grid=grid,
        kernel=_mlstm_chunkwise__parallel_bw_dQKV_kernel,
    )
    # sum up the contributions of each siz_b_DHQK block
    matDeltaV = matDeltaV.sum(axis=0)  # (B, NH, S, DHHV)

    return matDeltaQ, matDeltaK, matDeltaV


def _mlstm_chunkwise_bw(
    # Forward arguments
    matQ: jax.Array,  # (B, NH, S, DHQK)
    matK: jax.Array,  # (B, NH, S, DHQK)
    matV: jax.Array,  # (B, NH, S, DHV)
    vecI: jax.Array,  # (B, NH, S)
    vecF: jax.Array,  # (B, NH, S)
    matC_initial: jax.Array | None = None,  # (B, NH, DHQK, DHV)
    vecN_initial: jax.Array | None = None,  # (B, NH, DHQK)
    scaM_initial: jax.Array | None = None,  # (B, NH)
    qk_scale: float = None,
    # Backward arguments
    matC_all: jax.Array | None = None,  # (B, NH, NC * DHQK, DHV)
    vecN_all: jax.Array | None = None,  # (B, NH, NC * DHQK)
    scaM_all: jax.Array | None = None,  # (B, NH, NC)
    vecN_out: jax.Array | None = None,  # (B, NH, NC * L) = (B, NH, S)
    vecM_out: jax.Array | None = None,  # (B, NH, NC * L) = (B, NH, S)
    matDeltaH: jax.Array | None = None,  # (B, NH, S, DHV)
    matDeltaC_last: jax.Array | None = None,  # (B, NH, DHQK, DHV)
    # Common arguments
    CHUNK_SIZE: int = 64,
    EPS: float = 1e-6,
    reduce_slicing: bool = False,  # pylint: disable=unused-argument
) -> tuple[
    jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array | None, jax.Array | None, jax.Array | None
]:  # matDeltaQ, matDeltaK, matDeltaV, vecDeltaI, vecDeltaF, matDeltaC_initial, vecDeltaN_initial, scaDeltaM_initial
    """
    Computes the backward pass of the mLSTM chunkwise formulation.

    Args:
        matQ: Tensor containing the query vectors. Shape (B, NH, S, DHQK).
        matK: Tensor containing the key vectors. Shape (B, NH, S, DHQK).
        matV: Tensor containing the value vectors. Shape (B, NH, S, DHV).
        vecI: Tensor containing the input gate pre-activations. Shape (B, NH, S).
        vecF: Tensor containing the forget gate pre-activations. Shape (B, NH, S).
        matC_initial: Tensor containing the initial C states. Shape (B, NH, DHQK, DHV).
            Defaults to None.
        vecN_initial: Tensor containing the initial N states. Shape (B, NH, DHQK).
            Defaults to None.
        scaM_initial: Tensor containing the initial M states. Shape (B, NH).
            Defaults to None.
        qk_scale: Scale factor for the QK matrix. Defaults to None.
        matC_all: Tensor containing all C states. Shape (B, NH, NC * DHQK, DHV).
            Defaults to None.
        vecN_all: Tensor containing all N states. Shape (B, NH, NC * DHQK).
            Defaults to None.
        scaM_all: Tensor containing all M states. Shape (B, NH, NC).
            Defaults to None.
        vecN_out: Tensor containing the N states for the output. Shape (B, NH, S).
            Defaults to None.
        vecM_out: Tensor containing the M states for the output. Shape (B, NH, S).
            Defaults to None.
        matDeltaH: Tensor containing the H gradients. Shape (B, NH, S, DHV).
            Defaults to None.
        matDeltaC_last: Tensor containing the last C gradients. Shape (B, NH, DHQK, DHV).
            Defaults to None.
        CHUNK_SIZE: Chunk size. Defaults to 64.
        EPS: Epsilon value. Defaults to 1e-6.
        reduce_slicing: If True, reduces the slicing operations taken in the preprocessing to
            the kernel. This leads to performance improvements during training while returning
            the same results. Defaults to False.

    Returns:
        Gradients for the query, key, value, vecI and vecF matrices. Shapes (B, NH, S, DHQK),
        (B, NH, S, DHQK), (B, NH, S, DHV), (B, NH, S), (B, NH, S). If initial states are provided,
        the function also returns the gradients for the initial C, N and M states.
    """
    B, NH, S, DHQK = matQ.shape

    assert S % CHUNK_SIZE == 0, f"Sequence length {S} is not divisible by chunk size {CHUNK_SIZE}."

    NC = S // CHUNK_SIZE

    if qk_scale is None:
        qk_scale = DHQK**-0.5

    vecI = vecI.reshape(B, NH, NC, CHUNK_SIZE)
    vecF = vecF.reshape(B, NH, NC, CHUNK_SIZE).astype(jnp.float32)

    # Compute the gates, the g and the a and b vectors.
    vecF_logsig: jax.Array = jax.nn.log_sigmoid(vecF)
    vecB = vecF_logsig.cumsum(axis=-1)

    # Recompute the "all" states if needed.
    if matC_all is None:
        assert (
            (matC_all is None) and (vecN_all is None) and (scaM_all is None)
        ), "Either all or none of the states must be provided."
        matC_all, vecN_all, scaM_all = _mlstm_chunkwise__recurrent_fw_C(
            matK=matK,
            matV=matV,
            vecB=vecB,
            vecI=vecI,
            matC_initial=matC_initial,
            vecN_initial=vecN_initial,
            scaMinter_initial=scaM_initial,
            qk_scale=qk_scale,
            CHUNK_SIZE=CHUNK_SIZE,
            NUM_CHUNKS=NC,
        )

    # Recurrent backward: compute the deltaC gradients.
    matDeltaC_states = _mlstm_chunkwise__recurrent_bw_dC(
        matQ=matQ,  # (B, NH, S, DHQK)
        vecB=vecB,  # (B, NH, NC, L)
        scaM_inter=scaM_all,  # (B, NH, NC+1)
        vecM_combine=vecM_out,  # (B, NH, S)
        matDeltaH=matDeltaH,  # (B, NH, S, DHV)
        vecN_out=vecN_out,  # (B, NH, S)
        matDeltaC_last=matDeltaC_last,  # (B, NH, DHQK, DHV)
        qk_scale=qk_scale,
        CHUNK_SIZE=CHUNK_SIZE,
        NUM_CHUNKS=NC,
        EPS=EPS,
    )  # (B, NH, NC * DHQK, DHV)

    # Parallel backward: compute the deltaQ, deltaK, deltaV, deltaI gradients

    matDeltaQ, matDeltaK, matDeltaV = _mlstm_chunkwise__parallel_bw_dQKV(
        matQ=matQ,
        matK=matK,
        matV=matV,
        vecB=vecB,
        vecI=vecI,
        vecM_combine=vecM_out,
        scaM_inter=scaM_all,  # (B, NH, NC)
        matC_states=matC_all,  # (B, NH, (NC+1) * DHQK, DHV)
        matDeltaH=matDeltaH,  # (B, NH, S, DHV)
        vecN_out=vecN_out,  # (B, NH, S)
        matDeltaC_states=matDeltaC_states,  # (B, NH, (NC+1) * DHQK, DHV)
        qk_scale=qk_scale,
        CHUNK_SIZE=CHUNK_SIZE,
        NUM_CHUNKS=NC,
        EPS=EPS,
    )

    # Postprocessing: compute deltaF and deltaI gradients.
    vecF = vecF.reshape(B, NH, S)
    # Compute the vecDeltaFbar values with dfbar = rev_cumsum((q*dq - k*dk).sum(-1)).
    matQ = matQ.astype(jnp.float32)
    matK = matK.astype(jnp.float32)
    matDeltaQ = matDeltaQ.astype(jnp.float32)
    matDeltaK = matDeltaK.astype(jnp.float32)
    vecDeltaFbar_acc = ((matQ * matDeltaQ) - (matK * matDeltaK)).sum(axis=-1)
    vecDeltaFbar = jnp.flip(vecDeltaFbar_acc, axis=-1).astype(jnp.float32)
    vecDeltaFbar = jnp.flip(vecDeltaFbar.cumsum(axis=-1), axis=-1)
    vecDeltaF = vecDeltaFbar * jax.nn.sigmoid(-vecF)
    # Compute deltaI.
    # Both are equivalent:
    # vecDeltaI = (matV * matDeltaV).sum(-1)
    vecDeltaI = (matK * matDeltaK).sum(axis=-1)

    matDeltaC_initial = matDeltaC_states[:, :, :DHQK, :] if matC_initial is not None else None
    vecDeltaN_initial = jnp.zeros_like(vecN_initial) if vecN_initial is not None else None
    scaDeltaM_initial = jnp.zeros_like(scaM_initial) if scaM_initial is not None else None

    return (
        matDeltaQ,
        matDeltaK,
        matDeltaV,
        vecDeltaI,
        vecDeltaF,
        matDeltaC_initial,
        vecDeltaN_initial,
        scaDeltaM_initial,
    )
