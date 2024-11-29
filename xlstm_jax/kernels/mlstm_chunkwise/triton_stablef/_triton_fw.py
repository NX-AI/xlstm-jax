# pylint: disable=invalid-name
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
    chunk_mlstm_fwd_kernel_C,
    chunk_mlstm_fwd_kernel_h,
)

from xlstm_jax.kernels.stride_utils import get_stride


def assert_equal(a, b):
    assert a == b, f"{a} is not equal to {b}"


def _mlstm_chunkwise__recurrent_fw_C(
    matK: jax.Array,  # (B, H, S, K)
    matV: jax.Array,  # (B, H, S, V)
    vecF: jax.Array,  # (B, H, NT, BT)
    vecI: jax.Array,  # (B, H, NT, BT)
    matC_initial: jax.Array | None = None,  # (B, H, K, V)
    vecN_initial: jax.Array | None = None,  # (B, H, K)
    scaMinter_initial: jax.Array | None = None,  # (B, H)
    chunk_size: int = 64,
    num_chunks: int = 1,
    store_final_state: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array] | tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Execute the recurrent forward kernel for the C computation in the mLSTM chunkwise formulation.

    This function defines the grid and block sizes for the kernel launch and calls the kernel. See
    the fwbw backend implementation and the triton kernels for more information.

    Args:
        matK: Tensor containing the keys. Shape (B, H, S, K).
        matV: Tensor containing the values. Shape (B, H, S, V).
        vecB: Tensor containing the summed log forget gate activations. Shape (B, H, NT, BT).
        vecI: Tensor containing the input gate. Shape (B, H, NT, BT).
        matC_states: Buffer for the states of the C matrix.
            Shape (B, H, NT * K, V). Defaults to None.
        vecN_states: Buffer for the states of the N vector. Shape (B, H, NT * K).
            Defaults to None.
        scaMinter_states: Buffer for the states of the M scalar. Shape (B, H, (NT + 1)).
            Defaults to None.
        matC_initial: Initial state of the C matrix. Shape (B, H, K, V).
            Defaults to None.
        vecN_initial: Initial state of the N vector. Shape (B, H, K).
            Defaults to None.
        scaMinter_initial: Initial state of the M scalar. Shape (B, H).
            Defaults to None.
        qk_scale: Scaling factor for the QK matrix. Defaults to None and will be inferred.
        chunk_size: Chunk size for the kernel. Defaults to 64.
        num_chunks: Number of chunks. Defaults to 1.
        store_final_state: Whether to return the final state


    Returns:
        Tuple containing the states of the C matrix, the N vector and the M scalar.
    """
    B, H, T, K = matK.shape
    V = matV.shape[-1]

    NT = num_chunks
    BT = chunk_size

    assert NT == vecF.shape[2], "Number of chunks must match the number of chunks in vecB."
    assert BT == vecF.shape[3], "Chunk size must match the chunk size in vecB."
    assert is_power_of_2(BT), "Chunk size must be a power of 2."

    BHQK = min(64, triton.next_power_of_2(K))
    BHHV = min(64, triton.next_power_of_2(V))

    NK = triton.cdiv(K, BHQK)
    NV = triton.cdiv(V, BHHV)

    num_stages = 1
    num_warps = 4 if BHQK == 64 else 2

    # Check if the initial states are provided.
    USE_INITIAL_STATE = matC_initial is not None

    # If the states are not provided, they are initialized to the correct shape in the jax-triton call.
    matC_states = jax.ShapeDtypeStruct((B, H, NT * K, V), dtype=jnp.float32)
    vecN_states = jax.ShapeDtypeStruct((B, H, NT * K), dtype=jnp.float32)
    scaMinter_states = jax.ShapeDtypeStruct((B, H, NT + 1), dtype=jnp.float32)

    # Shared kwargs for the triton call.
    grid = (NK, NV, B * H)
    triton_kwargs = {
        "str_QK_H": get_stride(matK, axis=1),
        "str_QK_t": get_stride(matK, axis=2),
        "str_QK_d": get_stride(matK, axis=3),
        "str_VH_H": get_stride(matV, axis=1),
        "str_VH_t": get_stride(matV, axis=2),
        "str_VH_d": get_stride(matV, axis=3),
        "str_C_H": get_stride(matC_states, axis=1),
        "str_C_K": get_stride(matC_states, axis=2),
        "str_N_H": get_stride(vecN_states, axis=1),
        "T": T,
        "K": K,
        "V": V,
        "BT": BT,
        "BHQK": BHQK,
        "BHHV": BHHV,
        "NT": NT,
        "USE_INITIAL_STATE": USE_INITIAL_STATE,
        "STORE_FINAL_STATE": store_final_state,
        "num_stages": num_stages,
        "num_warps": num_warps,
        "grid": grid,
        "kernel": chunk_mlstm_fwd_kernel_C,
    }

    matC_final_state = jax.ShapeDtypeStruct((B, H, K, V), dtype=jnp.float32)
    vecN_final_state = jax.ShapeDtypeStruct((B, H, K), dtype=jnp.float32)
    scaMinter_final_state = jax.ShapeDtypeStruct((B, H), dtype=jnp.float32)
    if matC_initial is None:
        matC_initial = jnp.zeros([1])
        vecN_initial = jnp.zeros([1])
        scaMinter_initial = jnp.zeros([1])

    out_shape = (
        matC_states,
        vecN_states,
        scaMinter_states,
        matC_final_state,
        vecN_final_state,
        scaMinter_final_state,
    )

    assert_equal(get_stride(scaMinter_states, axis=1), NT + 1)
    assert_equal(get_stride(vecI, axis=1), BT * NT)
    assert_equal(get_stride(vecI, axis=2), BT)
    assert_equal(get_stride(vecF, axis=1), BT * NT)
    assert_equal(get_stride(vecF, axis=2), BT)

    res = jt.triton_call(
        matK,  # (B, H, S, K)
        matV,  # (B, H, S, V)
        vecI,  # (B, H, NT, BT)
        vecF,  # (B, H, NT, BT)
        matC_initial,  # (B, H, K, V)
        vecN_initial,  # (B, H, K)
        scaMinter_initial,  # (B, H)
        out_shape=out_shape,
        # matC_initial=matC_initial,  # (B, H, K, V)
        # matN_initial=vecN_initial,  # (B, H, K)
        # matM_initial=scaMinter_initial,  # (B, H)
        **triton_kwargs,
    )
    matC_states, vecN_states, scaMinter_states, matC_final_state, vecN_final_state, scaMinter_final_state = res
    if store_final_state:
        return matC_states, vecN_states, scaMinter_states, matC_final_state, vecN_final_state, scaMinter_final_state
    return matC_states, vecN_states, scaMinter_states


def _mlstm_chunkwise__parallel_fw_H(
    matQ: jax.Array,  # (B, H, S, K)
    matK: jax.Array,  # (B, H, S, K)
    matV: jax.Array,  # (B, H, S, V)
    matC_states: jax.Array,  # (B, H, NT * K, V)
    vecN_states: jax.Array,  # (B, H, NT * K)
    scaMinter_states: jax.Array,  # (B, H, NT)
    vecI: jax.Array,  # (B, H, NT, BT)
    vecF: jax.Array,  # (B, H, NT, BT)
    qk_scale: float | None = None,
    chunk_size: int = 64,
    num_chunks: int = 1,
    eps: float = 1e-6,
    stabilize_correctly: bool = True,
    norm_val: float = 1.0,
) -> tuple[jax.Array, jax.Array, jax.Array]:  # matH_out (B, H, S, V), vecN_out (B, H, S)
    """
    Execute the parallel forward kernel for the H computation in the mLSTM chunkwise formulation.

    This function defines the grid and block sizes for the kernel launch and calls the kernel. See
    the fwbw backend implementation and the triton kernels for more information.

    Args:
        matQ: Tensor containing the queries. Shape (B, H, S, K).
        matK: Tensor containing the keys. Shape (B, H, S, K).
        matV: Tensor containing the values. Shape (B, H, S, V).
        matC_states: States of the C matrix. Shape (B, H, NT * K, V).
            This state and following states must be all states up to the last chunk, i.e. :-1.
        vecN_states: States of the N vector. Shape (B, H, NT * K).
        scaMinter_states: States of the M scalar. Shape (B, H, NT + 1).
        vecI: Tensor containing the input gate. Shape (B, H, NT, BT).
        vecB: Tensor containing the summed log forget gate activations. Shape (B, H, NT, BT).
        qk_scale: Scaling factor for the QK matrix. Defaults to None and will be inferred.
        CHUNK_SIZE: Chunk size for the kernel. Defaults to 64.
        NUM_CHUNKS: Number of chunks. Defaults to 1.
        EPS: Small value to avoid division by zero. Defaults to 1e-6.

    Returns:
        Tuple containing the output matrix H (shape (B, H, S, V)) and the N vector (shape (B, H, S)).
    """
    B, H, T, K = matK.shape
    V = matV.shape[-1]
    NT = num_chunks
    BT = chunk_size
    assert NT == vecF.shape[2], "Number of chunks must match the number of chunks in vecB."
    assert BT == vecF.shape[3], "Chunk size must match the chunk size in vecB."
    assert is_power_of_2(BT), "Chunk size must be a power of 2."

    if qk_scale is None:
        qk_scale = K**-0.5

    BHQK = min(64, triton.next_power_of_2(K))
    BHHV = min(64, triton.next_power_of_2(V))

    num_stages = 1
    num_warps = 4 if BHQK == 64 else 2

    # Prepare the output shapes.
    matH_out = jax.ShapeDtypeStruct((B, H, T, V), matQ.dtype)
    vecN_out = jax.ShapeDtypeStruct((B, H, T), jnp.float32)
    vecM_out = jax.ShapeDtypeStruct((B, H, T), jnp.float32)

    # Define the grid and call the triton kernel.
    grid = (BHHV, NT, B * H)
    assert_equal(get_stride(scaMinter_states, axis=1), NT + 1)
    assert_equal(get_stride(vecI, axis=1), BT * NT)
    assert_equal(get_stride(vecI, axis=2), BT)
    assert_equal(get_stride(vecF, axis=1), BT * NT)
    assert_equal(get_stride(vecF, axis=2), BT)
    assert_equal(get_stride(vecN_out, axis=1), T)
    assert_equal(get_stride(vecN_out, axis=1), T)
    assert_equal(get_stride(matH_out, axis=1), T * V)
    assert_equal(get_stride(matH_out, axis=2), V)

    matH_out, vecN_out, vecM_out = jt.triton_call(
        matQ,  # (B, H, S, K)
        matK,  # (B, H, S, K)
        matV,  # (B, H, S, V)
        matC_states,  # (B, H, NT * K, V)
        vecN_states,  # (B, H, NT * K)
        scaMinter_states,  # (B, H, NT)
        vecI,  # (B, H, NT, BT)
        vecF,  # (B, H, NT, BT)
        out_shape=(matH_out, vecN_out, vecM_out),
        str_QK_H=get_stride(matQ, axis=1),
        str_QK_t=get_stride(matQ, axis=2),
        str_QK_d=get_stride(matQ, axis=3),
        str_VH_H=get_stride(matV, axis=1),
        str_VH_t=get_stride(matV, axis=2),
        str_VH_d=get_stride(matV, axis=3),
        str_C_H=get_stride(matC_states, axis=1),
        str_C_K=get_stride(matC_states, axis=2),
        str_N_H=get_stride(vecN_states, axis=1),
        scale=qk_scale,
        EPS=eps,
        NORM_VAL=norm_val,
        STABILIZE_CORRECTLY=stabilize_correctly,
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
        kernel=chunk_mlstm_fwd_kernel_h,
    )

    return matH_out, vecN_out, vecM_out


def _mlstm_chunkwise_fw(
    matQ: jax.Array,  # (B, H, S, K)
    matK: jax.Array,  # (B, H, S, K)
    matV: jax.Array,  # (B, H, S, V)
    vecI: jax.Array,  # (B, H, S)
    vecF: jax.Array,  # (B, H, S)
    matC_initial: jax.Array | None = None,  # (B, H, K, V)
    vecN_initial: jax.Array | None = None,  # (B, H, K)
    scaM_initial: jax.Array | None = None,  # (B, H)
    qk_scale: float = None,
    return_last_states: bool = False,
    return_all_states: bool = False,
    chunk_size: int = 64,
    stabilize_correctly: bool = True,
    norm_val: float = 1.0,
    eps: float = 1e-6,
) -> tuple[
    jax.Array,  # matH_out (B, H, S, V)
    jax.Array,  # vecN_out (B, H, S)
    jax.Array,  # vecM_out (B, H, S)
    None
    | (
        tuple[jax.Array, jax.Array, jax.Array]
    ),  # last_states (matC_states (B, H, K, V), vecN_states (B, H, K), scaMinter_states (B, H))
    None
    | (
        tuple[jax.Array, jax.Array, jax.Array]
    ),  # all_states (matC_states (B, H, (NT+1) * K, V), vecN_states (B, H, (NT+1) * K),
    # scaMinter_states (B, H, (NT+1)))
]:
    """
    Execute the forward pass of the mLSTM chunkwise formulation.

    Args:
        matQ: Tensor containing the queries. Shape (B, H, S, K).
        matK: Tensor containing the keys. Shape (B, H, S, K).
        matV: Tensor containing the values. Shape (B, H, S, V).
        vecI: Tensor containing the input gate. Shape (B, H, S).
        vecF: Tensor containing the forget gate. Shape (B, H, S).
        matC_initial: Initial state of the C matrix. Shape (B, H, K, V).
            Defaults to None.
        vecN_initial: Initial state of the N vector. Shape (B, H, K).
            Defaults to None.
        scaM_initial: Initial state of the M scalar. Shape (B, H).
            Defaults to None.
        qk_scale: Scaling factor for the QK matrix. Defaults to None and will be inferred.
        return_last_states: Whether to return the last states. Defaults to False.
        return_all_states: Whether to return all states. Defaults to False.
        chunk_size: Chunk size for the kernel. Defaults to 64.
        stabilize_correctly: Whether to stabilize with max(norm_val*e^-m, |nk|) instead of max(norm_val, |nk|)
        norm_val: Norm scale in the max formula above.

    Returns:
        Tuple containing the output matrix H (shape (B, H, S, V)), the N vector (shape (B, H, S)),
        the M scalar (shape (B, H)). Optionally, it might contain last states (matC_states,
        vecN_states, scaMinter_states) and optional all states (matC_states, vecN_states,
        scaMinter_states).
    """
    B, H, T, K = matQ.shape
    assert T % chunk_size == 0, f"Sequence length {T} is not divisible by chunk size {chunk_size}."
    NT = T // chunk_size

    vecI = 1.44269504 * vecI.reshape(B, H, NT, chunk_size).astype(jnp.float32)
    vecF = vecF.reshape(B, H, NT, chunk_size).astype(jnp.float32)
    vecF_logsig: jax.Array = 1.44269504 * jax.nn.log_sigmoid(vecF)

    if qk_scale is None:
        qk_scale = K**-0.5

    # Materialize the C_k, n_k, m_k states for each chunk.
    if return_last_states:
        matC_states, vecN_states, scaMinter_states, matC_final_state, vecN_final_state, scaMinter_final_state = (
            _mlstm_chunkwise__recurrent_fw_C(
                matK=matK,
                matV=matV,
                vecF=vecF_logsig,
                vecI=vecI,
                matC_initial=matC_initial,
                vecN_initial=vecN_initial,
                scaMinter_initial=scaM_initial,
                store_final_state=return_last_states,
                chunk_size=chunk_size,
                num_chunks=NT,
            )
        )
    else:
        matC_states, vecN_states, scaMinter_states = _mlstm_chunkwise__recurrent_fw_C(
            matK=matK,
            matV=matV,
            vecF=vecF_logsig,
            vecI=vecI,
            matC_initial=matC_initial,
            vecN_initial=vecN_initial,
            scaMinter_initial=scaM_initial,
            store_final_state=return_last_states,
            chunk_size=chunk_size,
            num_chunks=NT,
        )

    # Compute the outputs within each chunk.
    matH_out, vecN_out, vecM_out = _mlstm_chunkwise__parallel_fw_H(
        matQ=matQ,
        matK=matK,
        matV=matV,
        matC_states=matC_states,
        vecN_states=vecN_states,
        scaMinter_states=scaMinter_states,
        vecI=vecI,
        vecF=vecF_logsig,
        qk_scale=qk_scale,
        chunk_size=chunk_size,
        stabilize_correctly=stabilize_correctly,
        norm_val=norm_val,
        num_chunks=NT,
        eps=eps,
    )

    # Return the outputs and optionally the states.
    ret_tuple = (
        matH_out,
        vecN_out,
        vecM_out,
    )
    if return_last_states:
        ret_tuple += (
            (
                matC_final_state,
                vecN_final_state,
                scaMinter_final_state,
            ),
        )
    else:
        ret_tuple += (None,)
    if return_all_states:
        ret_tuple += ((matC_states, vecN_states, scaMinter_states),)
    else:
        ret_tuple += (None,)

    return ret_tuple  # (matH_out, vecN_out, vecM_out, optional(last_states), optional(all_states))
