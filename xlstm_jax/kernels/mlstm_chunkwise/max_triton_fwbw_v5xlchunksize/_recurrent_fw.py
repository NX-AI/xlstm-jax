from mlstm_kernels.mlstm_kernels.kernel_utils import is_power_of_2
from mlstm_kernels.mlstm_kernels.mlstm.chunkwise.max_triton_fwbw_v5xlchunksize._triton_recurrent_fw import (
    _mlstm_chunkwise__recurrent_fw_C_kernel,
)

import jax
import jax.numpy as jnp
import jax_triton as jt
import triton

from xlstm_jax.kernels.kernel_utils import jax2triton_dtype
from xlstm_jax.kernels.stride_utils import get_stride


def mlstm_chunkwise__recurrent_fw_C(
    matK: jax.Array,  # (B, NH, S, DHQK)
    matV: jax.Array,  # (B, NH, S, DHHV)
    vecF: jax.Array,  # (B, NH, NC * L) = (B, NH, S)
    vecI: jax.Array,  # (B, NH, NC * L) = (B, NH, S)
    matC_initial: jax.Array | None = None,  # (B, NH, DHQK, DHHV)
    vecN_initial: jax.Array | None = None,  # (B, NH, DHQK)
    scaMinter_initial: jax.Array | None = None,  # (B, NH)
    chunk_size: int = 64,
    num_stages: int | None = None,
    num_warps: int | None = None,
    save_states_every_nth_chunk: int = 1,
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
        chunk_size: Chunk size for the kernel. Defaults to 64.
        num_stages: Number of stages of the kernel. Defaults to None.
        num_warps: Number of warps of the kernel. Defaults to None.
        save_states_every_nth_chunk: Save the states every nth chunk. Defaults to 1.

    Returns:
        Tuple containing the states of the C matrix, the N vector and the M scalar.
    """
    B, NH, S, DHQK = matK.shape
    DHHV = matV.shape[-1]

    L = chunk_size
    assert S % L == 0, "Sequence length must be divisible by chunk size."
    NC = S // L

    assert save_states_every_nth_chunk > 0, "save_states_every_nth_chunk must be positive."
    assert save_states_every_nth_chunk <= NC, "save_states_every_nth_chunk must be <= NC."

    assert is_power_of_2(
        save_states_every_nth_chunk
    ), f"save_states_every_nth_chunk must be a power of 2. Got {save_states_every_nth_chunk}."

    assert is_power_of_2(L), "Chunk size must be a power of 2."

    siz_b_DHQK = min(64, triton.next_power_of_2(DHQK))
    siz_b_DHHV = min(64, triton.next_power_of_2(DHHV))

    num_b_DHQK = triton.cdiv(DHQK, siz_b_DHQK)
    num_b_DHHV = triton.cdiv(DHHV, siz_b_DHHV)

    num_stages = 1 if num_stages is None else num_stages
    if num_warps is None:
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
        # triton_call() expects the first arguments to be the input arrays, and the last arguments
        # to be the output arrays.
        # The output arrays (whose shape is passed into out_shape argument) are allocated by the triton kernel.
        # Since the matC_initial, vecN_initial, and scaMinter_initial are optional INPUT arguments to the kernel,
        # we always need to pass them in in order for the output arrays to be always at the correct position
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

    num_chunks_saved = NC // save_states_every_nth_chunk

    # If the states are not provided, they are initialized to the correct shape in the jax-triton call.
    matC_states = jax.ShapeDtypeStruct((B, NH, (num_chunks_saved + 1) * DHQK, DHHV), dtype=jnp.float32)
    vecN_states = jax.ShapeDtypeStruct((B, NH, (num_chunks_saved + 1) * DHQK), dtype=jnp.float32)
    scaMinter_states = jax.ShapeDtypeStruct((B, NH, (num_chunks_saved + 1)), dtype=jnp.float32)

    # Shared kwargs for the triton call.
    grid = (num_b_DHQK, num_b_DHHV, B * NH)
    triton_kwargs = dict(
        str_matK_B_NH=get_stride(matK, axis=1),
        str_matK_S=get_stride(matK, axis=2),
        str_matK_DHQK=get_stride(matK, axis=3),
        str_matV_B_NH=get_stride(matV, axis=1),
        str_matV_S=get_stride(matV, axis=2),
        str_matV_DHHV=get_stride(matV, axis=3),
        str_vecFI_B_NH=get_stride(vecF, axis=1),
        str_matCstates_B_NH=get_stride(matC_states, axis=1),
        str_matCstates_NCDHQK=get_stride(matC_states, axis=2),
        str_matCstates_DHHV=get_stride(matC_states, axis=3),
        str_vecNstates_B_NH=get_stride(vecN_states, axis=1),
        str_vecNstates_NCDHQK=get_stride(vecN_states, axis=2),
        str_scaMinterstates_B_NH=get_stride(scaMinter_states, axis=1),
        str_scaMinterstates_NC=get_stride(scaMinter_states, axis=2),
        str_matCinitial_B_NH=str_matCinitial_B_NH,
        str_matCinitial_DHQK=str_matCinitial_DHQK,
        str_matCinitial_DHHV=str_matCinitial_DHHV,
        str_vecNinitial_B_NH=str_vecNinitial_B_NH,
        str_vecNinitial_DHQK=str_vecNinitial_DHQK,
        str_scaMinterinitial_B_NH=str_scaMinterinitial_B_NH,
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
        USE_INITIAL_STATE=USE_INITIAL_STATE,
        DTYPE=jax2triton_dtype(matK.dtype),
        num_stages=num_stages,
        num_warps=num_warps,
        grid=grid,
        kernel=_mlstm_chunkwise__recurrent_fw_C_kernel,
    )

    matC_states, vecN_states, scaMinter_states = jt.triton_call(
        matK,  # (B, NH, S, DHQK)
        matV,  # (B, NH, S, DHHV)
        vecF,  # (B, NH, S)
        vecI,  # (B, NH, NC, L)
        matC_initial,  # (B, NH, DHQK, DHHV)
        vecN_initial,  # (B, NH, DHQK)
        scaMinter_initial,  # (B, NH)
        out_shape=(matC_states, vecN_states, scaMinter_states),
        **triton_kwargs,
    )

    return matC_states, vecN_states, scaMinter_states
