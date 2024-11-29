"""Triton backend for the forward pass of the mLSTM chunkwise formulation.

This file has been adapted from the original PyTorch Triton implementation to JAX.
For the Triton kernels, see mlstm_kernels/mlstm_kernels/mlstm/chunkwise/max_triton_fwbw_v3/_triton_fw.py.

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
from mlstm_kernels.mlstm_kernels.mlstm.chunkwise.max_triton_fwbw_v3._triton_fw import (
    _mlstm_chunkwise__recurrent_fw_C_kernel,
    _mlstm_chunkwise_parallel_fw_H_kernel,
)

from xlstm_jax.kernels.kernel_utils import jax2triton_dtype
from xlstm_jax.kernels.stride_utils import get_stride


def _mlstm_chunkwise__recurrent_fw_C(
    matK: jax.Array,  # (B, NH, S, DHQK)
    matV: jax.Array,  # (B, NH, S, DHHV)
    vecB: jax.Array,  # (B, NH, NC, L)
    vecI: jax.Array,  # (B, NH, NC, L)
    matC_states: jax.Array | None = None,  # (B, NH, (NC + 1) * DHQK, DHHV)
    vecN_states: jax.Array | None = None,  # (B, NH, (NC + 1) * DHQK)
    scaMinter_states: jax.Array | None = None,  # (B, NH, (NC + 1))
    matC_initial: jax.Array | None = None,  # (B, NH, DHQK, DHHV)
    vecN_initial: jax.Array | None = None,  # (B, NH, DHQK)
    scaMinter_initial: jax.Array | None = None,  # (B, NH)
    qk_scale: float | None = None,  # pylint: disable=unused-argument
    CHUNK_SIZE: int = 64,
    NUM_CHUNKS: int = 1,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Execute the recurrent forward kernel for the C computation in the mLSTM chunkwise formulation.

    This function defines the grid and block sizes for the kernel launch and calls the kernel. See
    the fwbw backend implementation and the triton kernels for more information.

    Args:
        matK: Tensor containing the keys. Shape (B, NH, S, DHQK).
        matV: Tensor containing the values. Shape (B, NH, S, DHHV).
        vecB: Tensor containing the summed log forget gate activations. Shape (B, NH, NC, L).
        vecI: Tensor containing the input gate. Shape (B, NH, NC, L).
        matC_states: Buffer for the states of the C matrix.
            Shape (B, NH, (NC + 1) * DHQK, DHHV). Defaults to None.
        vecN_states: Buffer for the states of the N vector. Shape (B, NH, (NC + 1) * DHQK).
            Defaults to None.
        scaMinter_states: Buffer for the states of the M scalar. Shape (B, NH, (NC + 1)).
            Defaults to None.
        matC_initial: Initial state of the C matrix. Shape (B, NH, DHQK, DHHV).
            Defaults to None.
        vecN_initial: Initial state of the N vector. Shape (B, NH, DHQK).
            Defaults to None.
        scaMinter_initial: Initial state of the M scalar. Shape (B, NH).
            Defaults to None.
        qk_scale: Scaling factor for the QK matrix. Defaults to None and will be inferred.
        CHUNK_SIZE: Chunk size for the kernel. Defaults to 64.
        NUM_CHUNKS: Number of chunks. Defaults to 1.

    Returns:
        Tuple containing the states of the C matrix, the N vector and the M scalar.
    """
    B, NH, S, DHQK = matK.shape
    DHHV = matV.shape[-1]

    NC = NUM_CHUNKS
    L = CHUNK_SIZE

    assert NC == vecB.shape[2], "Number of chunks must match the number of chunks in vecB."
    assert L == vecB.shape[3], "Chunk size must match the chunk size in vecB."
    assert is_power_of_2(L), "Chunk size must be a power of 2."

    siz_b_DHQK = min(64, triton.next_power_of_2(DHQK))
    siz_b_DHHV = min(64, triton.next_power_of_2(DHHV))

    num_b_DHQK = triton.cdiv(DHQK, siz_b_DHQK)
    num_b_DHHV = triton.cdiv(DHHV, siz_b_DHHV)

    num_stages = 1
    num_warps = 4 if siz_b_DHQK == 64 else 2

    # Check if the initial states are provided.
    USE_INITIAL_STATE = matC_initial is not None
    if USE_INITIAL_STATE:
        assert vecN_initial is not None and scaMinter_initial is not None
        str_matCinitial_B_NH = get_stride(matC_initial, axis=1)
        str_matCinitial_DHQK = get_stride(matC_initial, axis=2)
        str_matCinitial_DHHV = get_stride(matC_initial, axis=3)
        str_vecNinitial_B_NH = get_stride(vecN_initial, axis=1)
        str_vecNinitial_DHQK = get_stride(vecN_initial, axis=2)
        str_scaMinterinitial_B_NH = get_stride(scaMinter_initial, axis=1)
    else:
        assert matC_initial is None and vecN_initial is None and scaMinter_initial is None
        # Note: We need to pass empty arrays for the jax_triton.triton_call() to work.
        # triton_call() expects the first arguments to be the input arrays, and the last arguments to
        # be the output arrays.
        # The output arrays (whose shape is passed into out_shape argument) are allocated by the triton kernel.
        # Since the matC_initial, vecN_initial, and scaMinter_initial are optional INPUT arguments to the kernel,
        # we always need to pass them in order for the output arrays to be always at the correct position
        # in the argument list. So these empty arrays serve as placeholders in the argument list
        # and are not used within the kernel as USE_INITIAL_STATE is False.
        matC_initial = jnp.empty((1,), dtype=jnp.float32)
        vecN_initial = jnp.empty((1,), dtype=jnp.float32)
        scaMinter_initial = jnp.empty((1,), dtype=jnp.float32)
        str_matCinitial_B_NH = 0
        str_matCinitial_DHQK = 0
        str_matCinitial_DHHV = 0
        str_vecNinitial_B_NH = 0
        str_vecNinitial_DHQK = 0
        str_scaMinterinitial_B_NH = 0

    # If the states are not provided, they are initialized to the correct shape in the jax-triton call.
    matC_states = (
        jax.ShapeDtypeStruct((B, NH, (NC + 1) * DHQK, DHHV), dtype=jnp.float32) if matC_states is None else matC_states
    )
    vecN_states = (
        jax.ShapeDtypeStruct((B, NH, (NC + 1) * DHQK), dtype=jnp.float32) if vecN_states is None else vecN_states
    )
    scaMinter_states = (
        jax.ShapeDtypeStruct((B, NH, (NC + 1)), dtype=jnp.float32) if scaMinter_states is None else scaMinter_states
    )

    # Shared kwargs for the triton call.
    grid = (num_b_DHQK, num_b_DHHV, B * NH)
    triton_kwargs = {
        "str_matK_B_NH": get_stride(matK, axis=1),
        "str_matK_S": get_stride(matK, axis=2),
        "str_matK_DHQK": get_stride(matK, axis=3),
        "str_matV_B_NH": get_stride(matV, axis=1),
        "str_matV_S": get_stride(matV, axis=2),
        "str_matV_DHHV": get_stride(matV, axis=3),
        "str_vecBI_B_NH": get_stride(vecB, axis=1),
        "str_vecBI_NC": get_stride(vecB, axis=2),
        "str_vecBI_L": get_stride(vecB, axis=3),
        "str_matCstates_B_NH": get_stride(matC_states, axis=1),
        "str_matCstates_NCDHQK": get_stride(matC_states, axis=2),
        "str_matCstates_DHHV": get_stride(matC_states, axis=3),
        "str_vecNstates_B_NH": get_stride(vecN_states, axis=1),
        "str_vecNstates_NCDHQK": get_stride(vecN_states, axis=2),
        "str_scaMinterstates_B_NH": get_stride(scaMinter_states, axis=1),
        "str_scaMinterstates_NC": get_stride(scaMinter_states, axis=2),
        "str_matCinitial_B_NH": str_matCinitial_B_NH,
        "str_matCinitial_DHQK": str_matCinitial_DHQK,
        "str_matCinitial_DHHV": str_matCinitial_DHHV,
        "str_vecNinitial_B_NH": str_vecNinitial_B_NH,
        "str_vecNinitial_DHQK": str_vecNinitial_DHQK,
        "str_scaMinterinitial_B_NH": str_scaMinterinitial_B_NH,
        "B": B,
        "NH": NH,
        "S": S,
        "DHQK": DHQK,
        "DHHV": DHHV,
        "NC": NC,
        "L": L,
        "siz_b_DHQK": siz_b_DHQK,
        "siz_b_DHHV": siz_b_DHHV,
        "USE_INITIAL_STATE": USE_INITIAL_STATE,
        "DTYPE": jax2triton_dtype(matK.dtype),
        "num_stages": num_stages,
        "num_warps": num_warps,
        "grid": grid,
        "kernel": _mlstm_chunkwise__recurrent_fw_C_kernel,
    }

    matC_states, vecN_states, scaMinter_states = jt.triton_call(
        matK,  # (B, NH, S, DHQK)
        matV,  # (B, NH, S, DHHV)
        vecB,  # (B, NH, NC, L)
        vecI,  # (B, NH, NC, L)
        matC_initial,  # (B, NH, DHQK, DHHV)
        vecN_initial,  # (B, NH, DHQK)
        scaMinter_initial,  # (B, NH)
        out_shape=(matC_states, vecN_states, scaMinter_states),
        **triton_kwargs,
    )

    return matC_states, vecN_states, scaMinter_states


def _mlstm_chunkwise__parallel_fw_H(
    matQ: jax.Array,  # (B, NH, S, DHQK)
    matK: jax.Array,  # (B, NH, S, DHQK)
    matV: jax.Array,  # (B, NH, S, DHHV)
    matC_states: jax.Array,  # (B, NH, NC * DHQK, DHHV)
    vecN_states: jax.Array,  # (B, NH, NC * DHQK)
    scaMinter_states: jax.Array,  # (B, NH, NC)
    vecI: jax.Array,  # (B, NH, NC, L)
    vecB: jax.Array,  # (B, NH, NC, L)
    qk_scale: float | None = None,
    CHUNK_SIZE: int = 64,
    NUM_CHUNKS: int = 1,
    EPS: float = 1e-6,
) -> tuple[jax.Array, jax.Array]:  # matH_out (B, NH, S, DHHV), vecN_out (B, NH, S)
    """
    Execute the parallel forward kernel for the H computation in the mLSTM chunkwise formulation.

    This function defines the grid and block sizes for the kernel launch and calls the kernel. See
    the fwbw backend implementation and the triton kernels for more information.

    Args:
        matQ: Tensor containing the queries. Shape (B, NH, S, DHQK).
        matK: Tensor containing the keys. Shape (B, NH, S, DHQK).
        matV: Tensor containing the values. Shape (B, NH, S, DHHV).
        matC_states: States of the C matrix. Shape (B, NH, NC * DHQK, DHHV).
            This state and following states must be all states up to the last chunk, i.e. :-1.
        vecN_states: States of the N vector. Shape (B, NH, NC * DHQK).
        scaMinter_states: States of the M scalar. Shape (B, NH, NC).
        vecI: Tensor containing the input gate. Shape (B, NH, NC, L).
        vecB: Tensor containing the summed log forget gate activations. Shape (B, NH, NC, L).
        qk_scale: Scaling factor for the QK matrix. Defaults to None and will be inferred.
        CHUNK_SIZE: Chunk size for the kernel. Defaults to 64.
        NUM_CHUNKS: Number of chunks. Defaults to 1.
        EPS: Small value to avoid division by zero. Defaults to 1e-6.

    Returns:
        Tuple containing the output matrix H (shape (B, NH, S, DHHV)) and the N vector (shape (B, NH, S)).
    """
    B, NH, S, DHQK = matK.shape
    DHHV = matV.shape[-1]
    NC = NUM_CHUNKS
    L = CHUNK_SIZE

    assert NC == vecB.shape[2], "Number of chunks must match the number of chunks in vecB."
    assert L == vecB.shape[3], "Chunk size must match the chunk size in vecB."
    assert is_power_of_2(L), "Chunk size must be a power of 2."

    if qk_scale is None:
        qk_scale = DHQK**-0.5

    siz_b_DHQK = min(64, triton.next_power_of_2(DHQK))
    siz_b_DHHV = min(64, triton.next_power_of_2(DHHV))

    num_b_DHHV = triton.cdiv(DHHV, siz_b_DHHV)

    num_stages = 1
    num_warps = 4 if siz_b_DHQK == 64 else 2

    # Prepare the output shapes.
    matH_out = jax.ShapeDtypeStruct((B, NH, S, DHHV), matQ.dtype)
    vecN_out = jax.ShapeDtypeStruct((B, NH, S), jnp.float32)
    vecM_out = jax.ShapeDtypeStruct((B, NH, S), jnp.float32)

    # Define the grid and call the triton kernel.
    grid = (num_b_DHHV, NC, B * NH)
    matH_out, vecN_out, vecM_out = jt.triton_call(
        matQ,  # (B, NH, S, DHQK)
        matK,  # (B, NH, S, DHQK)
        matV,  # (B, NH, S, DHHV)
        matC_states,  # (B, NH, NC * DHQK, DHHV)
        vecN_states,  # (B, NH, NC * DHQK)
        scaMinter_states,  # (B, NH, NC)
        vecI,  # (B, NH, NC, L)
        vecB,  # (B, NH, NC, L)
        out_shape=(matH_out, vecN_out, vecM_out),
        qk_scale=qk_scale,
        str_matQK_B_NH=get_stride(matQ, axis=1),
        str_matQK_S=get_stride(matQ, axis=2),
        str_matQK_DHQK=get_stride(matQ, axis=3),
        str_matHV_B_NH=get_stride(matV, axis=1),
        str_matHV_S=get_stride(matV, axis=2),
        str_matHV_DHHV=get_stride(matV, axis=3),
        str_matCstates_B_NH=get_stride(matC_states, axis=1),
        str_matCstates_NCDHQK=get_stride(matC_states, axis=2),
        str_matCstates_DHHV=get_stride(matC_states, axis=3),
        str_vecNstates_B_NH=get_stride(vecN_states, axis=1),
        str_vecNstates_NCDHQK=get_stride(vecN_states, axis=2),
        str_scaMinterstates_B_NH=get_stride(scaMinter_states, axis=1),
        str_vecBI_B_NH=get_stride(vecB, axis=1),
        str_vecBI_NC=get_stride(vecB, axis=2),
        str_vecBI_L=get_stride(vecB, axis=3),
        str_vecMN_B_NH=get_stride(vecN_out, axis=1),
        str_vecMN_S=get_stride(vecN_out, axis=2),
        B=B,
        NH=NH,
        S=S,
        DHQK=DHQK,
        DHHV=DHHV,
        NC=NC,
        L=L,
        siz_b_DHQK=siz_b_DHQK,
        siz_b_DHHV=siz_b_DHHV,
        DTYPE=jax2triton_dtype(matQ.dtype),
        EPS=EPS,
        num_stages=num_stages,
        num_warps=num_warps,
        grid=grid,
        kernel=_mlstm_chunkwise_parallel_fw_H_kernel,
    )

    return matH_out, vecN_out, vecM_out


def _mlstm_chunkwise_fw(
    matQ: jax.Array,  # (B, NH, S, DHQK)
    matK: jax.Array,  # (B, NH, S, DHQK)
    matV: jax.Array,  # (B, NH, S, DHV)
    vecI: jax.Array,  # (B, NH, S)
    vecF: jax.Array,  # (B, NH, S)
    matC_initial: jax.Array | None = None,  # (B, NH, DHQK, DHV)
    vecN_initial: jax.Array | None = None,  # (B, NH, DHQK)
    scaM_initial: jax.Array | None = None,  # (B, NH)
    qk_scale: float = None,
    return_last_states: bool = False,
    return_all_states: bool = False,
    CHUNK_SIZE: int = 64,
    EPS: float = 1e-6,
    reduce_slicing: bool = False,
) -> tuple[
    jax.Array,  # matH_out (B, NH, S, DHV)
    jax.Array,  # vecN_out (B, NH, S)
    jax.Array,  # vecM_out (B, NH, S)
    None
    | (
        tuple[jax.Array, jax.Array, jax.Array]
    ),  # last_states (matC_states (B, NH, DHQK, DHV), vecN_states (B, NH, DHQK), scaMinter_states (B, NH))
    None
    | (
        tuple[jax.Array, jax.Array, jax.Array]
    ),  # all_states (matC_states (B, NH, (NC+1) * DHQK, DHV), vecN_states (B, NH, (NC+1) * DHQK),
    # scaMinter_states (B, NH, (NC+1)))
]:
    """
    Execute the forward pass of the mLSTM chunkwise formulation.

    Args:
        matQ: Tensor containing the queries. Shape (B, NH, S, DHQK).
        matK: Tensor containing the keys. Shape (B, NH, S, DHQK).
        matV: Tensor containing the values. Shape (B, NH, S, DHV).
        vecI: Tensor containing the input gate. Shape (B, NH, S).
        vecF: Tensor containing the forget gate. Shape (B, NH, S).
        matC_initial: Initial state of the C matrix. Shape (B, NH, DHQK, DHV).
            Defaults to None.
        vecN_initial: Initial state of the N vector. Shape (B, NH, DHQK).
            Defaults to None.
        scaM_initial: Initial state of the M scalar. Shape (B, NH).
            Defaults to None.
        qk_scale: Scaling factor for the QK matrix. Defaults to None and will be inferred.
        return_last_states: Whether to return the last states. Defaults to False.
        return_all_states: Whether to return all states. Defaults to False.
        CHUNK_SIZE: Chunk size for the kernel. Defaults to 64.
        EPS: Small value to avoid division by zero. Defaults to 1e-6.
        reduce_slicing: If True, reduces the slicing operations taken in the preprocessing to
            the kernel. This leads to performance improvements during training while returning
            the same results. Defaults to False.

    Returns:
        Tuple containing the output matrix H (shape (B, NH, S, DHV)), the N vector (shape (B, NH, S)),
        the M scalar (shape (B, NH)). Optionally, it might contain last states (matC_states,
        vecN_states, scaMinter_states) and optional all states (matC_states, vecN_states,
        scaMinter_states).
    """
    B, NH, S, DHQK = matQ.shape
    assert S % CHUNK_SIZE == 0, f"Sequence length {S} is not divisible by chunk size {CHUNK_SIZE}."
    NC = S // CHUNK_SIZE

    vecI = vecI.reshape(B, NH, NC, CHUNK_SIZE)
    vecF = vecF.reshape(B, NH, NC, CHUNK_SIZE).astype(jnp.float32)

    # Compute the gates, the g and the a and b vectors.
    vecF_logsig: jax.Array = jax.nn.log_sigmoid(vecF)
    vecB = vecF_logsig.cumsum(axis=-1)

    if qk_scale is None:
        qk_scale = DHQK**-0.5

    # Materialize the C_k, n_k, m_k states for each chunk.
    matC_k_states, vecN_k_states, scaMinter_k_states = _mlstm_chunkwise__recurrent_fw_C(
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

    # Compute the outputs within each chunk.
    matH_out, vecN_out, vecM_out = _mlstm_chunkwise__parallel_fw_H(
        matQ=matQ,
        matK=matK,
        matV=matV,
        # These slices are not needed in the kernel and introduces considerable overhead.
        matC_states=matC_k_states if reduce_slicing else matC_k_states[:, :, :-DHQK, :],
        vecN_states=vecN_k_states if reduce_slicing else vecN_k_states[:, :, :-DHQK],
        scaMinter_states=scaMinter_k_states if reduce_slicing else scaMinter_k_states[:, :, :-1],
        vecI=vecI,
        vecB=vecB,
        qk_scale=qk_scale,
        CHUNK_SIZE=CHUNK_SIZE,
        NUM_CHUNKS=NC,
        EPS=EPS,
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
                matC_k_states[:, :, -DHQK:, :],
                vecN_k_states[:, :, -DHQK:],
                scaMinter_k_states[:, :, -1],
            ),
        )
    else:
        ret_tuple += (None,)
    if return_all_states:
        ret_tuple += ((matC_k_states, vecN_k_states, scaMinter_k_states),)
    else:
        ret_tuple += (None,)

    return ret_tuple  # (matH_out, vecN_out, vecM_out, optional(last_states), optional(all_states))
