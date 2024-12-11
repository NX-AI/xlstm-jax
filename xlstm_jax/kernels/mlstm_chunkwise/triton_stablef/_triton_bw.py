#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.
"""Triton backend for the backward pass of the mLSTM chunkwise formulation.

This file has been adapted from the original PyTorch Triton implementation to JAX.
For the Triton kernels, see mlstm_kernels/mlstm_kernels/mlstm/chunkwise/triton_fwbw_stablef.py.

In this file, we use the following notation:

Dimensions:
    B: batch size
    H: number of heads
    S: sequence length (K, V)
    T: sequence length (Q)
    K: hidden dimension (Q, K)
    V: hidden dimension (H, V)
    NT: number of chunks
    BT: chunk size

"""

import jax
import jax.numpy as jnp
import jax_triton as jt
import triton

from mlstm_kernels.mlstm_kernels.kernel_utils import is_power_of_2
from mlstm_kernels.mlstm_kernels.mlstm.chunkwise.triton_fwbw_stablef import (
    chunk_mlstm_bwd_kernel_dC,
    chunk_mlstm_bwd_kernel_dqkvif,
)

from xlstm_jax.kernels.stride_utils import get_stride

from ._triton_fw import _mlstm_chunkwise__recurrent_fw_C


def _mlstm_chunkwise__recurrent_bw_dC(
    matQ: jax.Array,  # (B, H, T, K)
    vecF: jax.Array,  # (B, H, NT, BT)
    scaM_inter: jax.Array,  # (B, H, NT+1)
    vecM: jax.Array,  # (B, H, T)
    vecN_out: jax.Array,  # (B, H, T)
    matDeltaH: jax.Array,  # (B, H, T, V)
    matDeltaC_last: jax.Array | None = None,  # (B, H, DHQK, DHHV)
    qk_scale: float | None = None,
    chunk_size: int = 64,
    num_chunks: int = 1,
    store_initial_state: bool = False,
) -> tuple[
    jax.Array, jax.Array | None, jax.Array | None
]:  # matDeltaC_states (B, H, NT * DHQK, DHHV), (B, H, DHQK, DHHV)
    """
    Computes only the deltaC gradients for the backward pass.

    The other gradients are computed in the other (kernel) function.
    We do not need to compute the gradients for the denominator, as it cancels out in the forward in the groupnorm.

    Args:
        matQ: Tensor containing the query vectors. Shape (B, H, T, K).
        vecF: Tensor containing the log forget gate activations. Shape (B, H, NT, BT).
        scaM_inter: States of the M scalar. Shape (B, H, NT+1).
        vecM: M states. Shape (B, H, T).
        matDeltaH: Tensor containing the H gradients. Shape (B, H, T, V).
        vecN_out: States of the N vector. Shape (B, H, NT * DHQK).
        matDeltaC_last: Tensor containing the last C gradients. Shape (B, H, DHQK, DHHV).
            Defaults to None.
        qk_scale: Scale factor for the QK matrix. Defaults to None.
        chunk_size: Chunk size. Defaults to 64.
        num_chunks: Number of chunks. Defaults to 1.
        store_initial_state: Whether to store the inital state gradient and logscale (m state)

    Returns:
        Tensor containing the C gradients and the C_first gradients.
        Shapes (B, H, NT * DHQK, DHHV), (B, H, DHQK, DHHV).
    """
    B, H, T, K, V = *matQ.shape, matDeltaH.shape[-1]
    NT = num_chunks
    BT = chunk_size

    assert NT == vecF.shape[2], "Number of chunks must match the number in vecF."
    assert BT == vecF.shape[3], "Chunk size must match the chunk size in vecF."
    assert is_power_of_2(BT), "Chunk size must be a power of 2."

    if qk_scale is None:
        qk_scale = K**-0.5

    USE_LAST_STATE = matDeltaC_last is not None

    out_shape = (
        jax.ShapeDtypeStruct(shape=(B, H, NT * K, V), dtype=jnp.float32),
        jax.ShapeDtypeStruct(shape=(B, H, K, V), dtype=jnp.float32)
        if store_initial_state
        else jax.ShapeDtypeStruct(shape=(1,), dtype=jnp.float32),
        jax.ShapeDtypeStruct(shape=(B, H), dtype=jnp.float32)
        if store_initial_state
        else jax.ShapeDtypeStruct(shape=(1,), dtype=jnp.float32),
    )
    if matDeltaC_last is None:
        # use the state dtype here
        matDeltaC_last = jnp.zeros((1,), dtype=jnp.float32)

    BHQK = min(64, triton.next_power_of_2(K))
    BHHV = min(64, triton.next_power_of_2(V))

    NK = triton.cdiv(K, BHQK)
    NV = triton.cdiv(V, BHHV)

    num_stages = 1
    num_warps = 4 if BHQK == 64 else 2

    matM_final = scaM_inter[:, :, -1]

    grid = (NK, NV, B * H)

    matDeltaC_states, matDeltaC_initial, scaM_initial = jt.triton_call(
        matQ,
        vecF,
        scaM_inter,
        vecM,
        vecN_out,
        matM_final,
        matDeltaH,
        matDeltaC_last,
        out_shape=out_shape,
        str_QK_H=get_stride(matQ, axis=1),
        str_QK_t=get_stride(matQ, axis=2),
        str_QK_d=get_stride(matQ, axis=3),
        str_VH_H=get_stride(matDeltaH, axis=1),
        str_VH_t=get_stride(matDeltaH, axis=2),
        str_VH_d=get_stride(matDeltaH, axis=3),
        str_C_H=get_stride(out_shape[0], axis=1),
        str_C_K=get_stride(out_shape[0], axis=2),
        scale=qk_scale,
        T=T,
        K=K,
        V=V,
        BT=BT,
        BHQK=BHQK,
        BHHV=BHHV,
        NT=NT,
        USE_LAST_STATE=USE_LAST_STATE,
        STORE_INITIAL_STATE=store_initial_state,
        num_stages=num_stages,
        num_warps=num_warps,
        grid=grid,
        kernel=chunk_mlstm_bwd_kernel_dC,
    )
    if not store_initial_state:
        matDeltaC_initial = None
        scaM_initial = None
    return matDeltaC_states, matDeltaC_initial, scaM_initial


def _mlstm_chunkwise__parallel_bw_dQKV(
    matQ: jax.Array,  # (B, H, T, K)
    matK: jax.Array,  # (B, H, T, K)
    matV: jax.Array,  # (B, H, T, V)
    vecI: jax.Array,  # (B, H, NT, BT)
    vecF: jax.Array,  # (B, H, NT, BT)
    vecM_combine: jax.Array,  # (B, H, T) = (B, H, NT * BT)
    scaM_inter: jax.Array,  # (B, H, NT+1)
    matC_states: jax.Array,  # (B, H, NT * DHQK, DHHV)
    matDeltaH: jax.Array,  # (B, H, T, V)
    vecN_out: jax.Array,  # (B, H, T)
    matDeltaC_states: jax.Array,  # (B, H, NT * DHQK, DHHV)
    qk_scale: float | None = None,
    chunk_size: int = 64,
    num_chunks: int = 1,
) -> tuple[
    jax.Array, jax.Array, jax.Array
]:  # matDeltaQ (B,NH,S,DHQK), matDeltaK (B,NH,S,DHQK), matDeltaV (B,NH,S,DHHV)
    """
    Computes the gradients for the query, key and value matrices.

    Args:
        matQ: Tensor containing the query vectors. Shape (B, H, T, K).
        matK: Tensor containing the key vectors. Shape (B, H, T, K).
        matV: Tensor containing the value vectors. Shape (B, H, T, V).
        vecF: Tensor containing the summed log forget gate activations. Shape (B, H, NT, BT).
        vecI: Tensor containing the input gate pre-activations. Shape (B, H, NT, BT).
        vecM_combine: Combined M states. Shape (B, H, T).
        scaM_inter: States of the M scalar. Shape (B, H, NT+1).
        matC_states: States of the C matrix. Shape (B, H, NT * DHQK, DHHV).
        matDeltaH: Tensor containing the H gradients. Shape (B, H, T, V).
        vecN_out: States of the N vector. Shape (B, H, T).
        matDeltaC_states: Tensor containing the C gradients. Shape (B, H, (NT+1) * DHQK, DHHV).
        qk_scale: Scale factor for the QK matrix. Defaults to None.
        chunk_size (int, optional): Chunk size. Defaults to 64.
        num_chunks (int, optional): Number of chunks. Defaults to 1.

    Returns:
        Gradients for the query, key and value matrices. Shapes (B, H, T, K), (B, H, T, K), (B, H, T, V).
    """
    B, H, T, K, V = *matQ.shape, matV.shape[-1]
    NT = num_chunks
    BT = chunk_size
    _dtype = matQ.dtype

    assert NT == vecF.shape[2], "Number of chunks must match the number in vecB."
    assert BT == vecF.shape[3], "Chunk size must match the chunk size in vecB."
    assert is_power_of_2(BT), "Chunk size must be a power of 2."

    if qk_scale is None:
        qk_scale = K**-0.5

    BHQK = min(32 if str(_dtype) == "float32" else 64, triton.next_power_of_2(K))
    BHHV = min(32 if str(_dtype) == "float32" else 64, triton.next_power_of_2(V))

    NK = triton.cdiv(K, BHQK)

    num_stages = 1
    num_warps = 4 if BHQK == 64 else 2

    # Specify output shapes.
    matDeltaQ = jax.ShapeDtypeStruct(shape=(B, H, T, K), dtype=_dtype)
    matDeltaK = jax.ShapeDtypeStruct(shape=(B, H, T, K), dtype=_dtype)
    # each b_DHQK thread block computes the contribution of its siz_b_DHQK block of matDeltaC
    # we need to sum them up to get the final result (we do this outside the kernel)
    matDeltaV = jax.ShapeDtypeStruct(shape=(NK, B, H, T, V), dtype=_dtype)
    vecDeltaI = jax.ShapeDtypeStruct(shape=(NK, B, H, T), dtype=jnp.float32)
    vecDeltaFq = jax.ShapeDtypeStruct(shape=(NK, B, H, T), dtype=jnp.float32)
    vecDeltaFk = jax.ShapeDtypeStruct(shape=(NK, B, H, T + 1), dtype=jnp.float32)
    scaDeltaFc = jax.ShapeDtypeStruct(shape=(NK, B, H, NT), dtype=jnp.float32)

    # Define the grid and call the triton kernel.
    grid = (NK, NT, B * H)
    matDeltaQ, matDeltaK, matDeltaV, vecDeltaI, vecDeltaFq, vecDeltaFk, scaDeltaFc = jt.triton_call(
        matQ,  # (B, H, T, K)
        matK,  # (B, H, T, K)
        matV,  # (B, H, T, V)
        matC_states,  # (B, H, NT * DHQK, DHHV)
        scaM_inter,  # (B, H, NT+1)
        vecM_combine,  # (B, H, T)
        vecN_out,  # (B, H, T)
        vecI,  # (B, H, NT, BT)
        vecF,  # (B, H, NT, BT)
        matDeltaH,  # (B, H, T, V)
        matDeltaC_states,  # (B, H, NT * DHQK, DHHV)
        out_shape=(matDeltaQ, matDeltaK, matDeltaV, vecDeltaI, vecDeltaFq, vecDeltaFk, scaDeltaFc),
        zeroed_outputs=(4, 5, 6),
        str_QK_H=get_stride(matQ, axis=1),
        str_QK_t=get_stride(matQ, axis=2),
        str_QK_d=get_stride(matQ, axis=3),
        str_VH_H=get_stride(matV, axis=1),
        str_VH_t=get_stride(matV, axis=2),
        str_VH_d=get_stride(matV, axis=3),
        str_C_H=get_stride(matC_states, axis=1),
        str_C_K=get_stride(matC_states, axis=2),
        scale=qk_scale,
        T=T,
        K=K,
        V=V,
        BT=BT,
        BS=BT,
        BHQK=BHQK,
        BHHV=BHHV,
        NT=NT,
        num_stages=num_stages,
        num_warps=num_warps,
        grid=grid,
        kernel=chunk_mlstm_bwd_kernel_dqkvif,
    )
    # sum up the contributions of each siz_b_DHQK block
    matDeltaV = matDeltaV.sum(axis=0)  # (B, H, T, V)
    vecDeltaI = vecDeltaI.sum(axis=0)  # (B, H, T)
    vecDeltaF = vecDeltaFq.sum(axis=0).reshape((B, H, NT, BT))
    vecDeltaF = vecDeltaF + vecDeltaFk[:, :, :, :-1].sum(axis=0).reshape((B, H, NT, BT))
    vecDeltaF = vecDeltaF + scaDeltaFc.sum(0)[:, :, :, None]
    vecDeltaF = vecDeltaF.reshape((B, H, T))

    return matDeltaQ, matDeltaK, matDeltaV, vecDeltaI, vecDeltaF


def _mlstm_chunkwise_bw(
    # Forward arguments
    matQ: jax.Array,  # (B, H, T, K)
    matK: jax.Array,  # (B, H, T, K)
    matV: jax.Array,  # (B, H, S, DHV)
    vecI: jax.Array,  # (B, H, T)
    vecF: jax.Array,  # (B, H, T)
    matC_initial: jax.Array | None = None,  # (B, H, DHQK, DHV)
    vecN_initial: jax.Array | None = None,  # (B, H, DHQK)
    scaM_initial: jax.Array | None = None,  # (B, NH)
    qk_scale: float | None = None,
    # Backward arguments
    matC_states: jax.Array | None = None,  # (B, H, NT * DHQK, DHV) - can actually be None
    scaM_states: jax.Array | None = None,  # (B, H, NC)
    vecN_out: jax.Array | None = None,  # (B, H, NT * BT) = (B, H, T)
    vecM_out: jax.Array | None = None,  # (B, H, NT * BT) = (B, H, T)
    # In-coming gradients
    matDeltaH: jax.Array | None = None,  # (B, H, S, DHV)
    matDeltaC_last: jax.Array | None = None,  # (B, H, DHQK, DHV)
    # Common arguments
    chunk_size: int = 64,
) -> tuple[
    jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array | None, jax.Array | None, jax.Array | None
]:  # matDeltaQ, matDeltaK, matDeltaV, vecDeltaI, vecDeltaF, matDeltaC_initial, vecDeltaN_initial, scaDeltaM_initial
    """
    Computes the backward pass of the mLSTM chunkwise formulation.

    Args:
        matQ: Tensor containing the query vectors. Shape (B, H, T, K).
        matK: Tensor containing the key vectors. Shape (B, H, T, K).
        matV: Tensor containing the value vectors. Shape (B, H, S, DHV).
        vecI: Tensor containing the input gate pre-activations. Shape (B, H, T).
        vecF: Tensor containing the forget gate pre-activations. Shape (B, H, T).
        matC_initial: Tensor containing the initial C states. Shape (B, H, DHQK, DHV).
            Defaults to None.
        vecN_initial: Tensor containing the initial N states. Shape (B, H, DHQK).
            Defaults to None.
        scaM_initial: Tensor containing the initial M states. Shape (B, NH).
            Defaults to None.
        qk_scale: Scale factor for the QK matrix. Defaults to None.
        matC_states: Tensor containing all C states. Shape (B, H, NT * DHQK, DHV).
            Defaults to None.
        scaM_states: Tensor containing all M states. Shape (B, H, NC).
            Defaults to None.
        vecN_out: Tensor containing the N states for the output. Shape (B, H, T).
            Defaults to None.
        vecM_out: Tensor containing the M states for the output. Shape (B, H, T).
            Defaults to None.
        matDeltaH: Tensor containing the H gradients. Shape (B, H, S, DHV).
            Defaults to None.
        matDeltaC_last: Tensor containing the last C gradients. Shape (B, H, DHQK, DHV).
            Defaults to None.
        chunk_size: Chunk size. Defaults to 64.


    Returns:
        Gradients for the query, key, value, vecI and vecF matrices. Shapes (B, H, T, K),
        (B, H, T, K), (B, H, S, DHV), (B, H, T), (B, H, T). If initial states are provided,
        the function also returns the gradients for the initial C, N and M states.
    """
    B, H, T, K = matQ.shape

    assert scaM_states is not None, "Need to keep the M states"
    assert vecN_out is not None, "Need to keep the local normalizers"
    assert vecM_out is not None, "Need to keep the local maximizers"

    assert T % chunk_size == 0, f"Sequence length {T} is not divisible by chunk size {chunk_size}."

    BT = chunk_size
    NT = T // chunk_size

    if qk_scale is None:
        qk_scale = K**-0.5

    vecI = 1.44269504 * vecI.reshape(B, H, NT, BT)
    vecF = vecF.reshape(B, H, NT, BT).astype(jnp.float32)
    vecF_logsig: jax.Array = 1.44269504 * jax.nn.log_sigmoid(vecF)

    USE_INITIAL_STATE = matC_initial is not None

    # Recompute the C states if needed.
    if matC_states is None:
        matC_states, _, _ = _mlstm_chunkwise__recurrent_fw_C(
            matK=matK,
            matV=matV,
            vecI=vecI,
            vecF=vecF_logsig,
            matC_initial=matC_initial,
            vecN_initial=vecN_initial,
            scaMinter_initial=scaM_initial,
            chunk_size=chunk_size,
            num_chunks=NT,
            store_final_state=False,
        )

    # Recurrent backward: compute the deltaC gradients.
    matDeltaC_states, matDeltaC_initial, _ = _mlstm_chunkwise__recurrent_bw_dC(
        matQ=matQ,  # (B, H, T, K)
        vecF=vecF_logsig,  # (B, H, T)
        scaM_inter=scaM_states,  # (B, H, NT)
        vecM=vecM_out,  # (B, H, NT, BT)
        vecN_out=vecN_out,  # (B, H, NT, BT)
        matDeltaH=matDeltaH,  # (B, H, T, V)
        matDeltaC_last=matDeltaC_last,  # (B, H, NT*K, V)
        qk_scale=qk_scale,
        chunk_size=chunk_size,
        num_chunks=NT,
        store_initial_state=USE_INITIAL_STATE,
    )  # (B, H, NT * K, V)

    matDeltaQ, matDeltaK, matDeltaV, vecDeltaI, vecDeltaF = _mlstm_chunkwise__parallel_bw_dQKV(
        matQ=matQ,
        matK=matK,
        matV=matV,
        vecI=vecI,
        vecF=vecF_logsig,
        vecM_combine=vecM_out,
        scaM_inter=scaM_states,  # (B, H, NT)
        matC_states=matC_states,  # (B, H, NT * K, V)
        matDeltaH=matDeltaH,  # (B, H, T, V)
        vecN_out=vecN_out,  # (B, H, T)
        matDeltaC_states=matDeltaC_states,  # (B, H, NT * K, V)
        qk_scale=qk_scale,
        chunk_size=chunk_size,
        num_chunks=NT,
    )

    # Postprocessing: compute deltaF and deltaI gradients.
    vecF = vecF.reshape((B, H, T))
    vecDeltaF = vecDeltaF.reshape((B, H, T)) * jax.nn.sigmoid(-vecF)

    matDeltaC_initial = matDeltaC_initial if matC_initial is not None else None
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
