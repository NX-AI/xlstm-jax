#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import jax
import jax.numpy as jnp

from ._parallel_fw import mlstm_chunkwise__parallel_fw_Hintra
from ._recurrent_fw import mlstm_chunkwise__recurrent_fw_C


def mlstm_chunkwise_fw(
    matQ: jax.Array,  # (B, NH, S, DHQK)
    matK: jax.Array,  # (B, NH, S, DHQK)
    matV: jax.Array,  # (B, NH, S, DHV)
    vecI: jax.Array,  # (B, NH, S)
    vecF: jax.Array,  # (B, NH, S)
    matC_initial: jax.Array | None = None,  # (B, NH, DHQK, DHV)
    vecN_initial: jax.Array | None = None,  # (B, NH, DHQK)
    scaM_initial: jax.Array | None = None,  # (B, NH)
    qk_scale: float | None = None,
    return_last_states: bool = False,
    return_all_states: bool = False,
    chunk_size_inter: int | None = None,
    chunk_size_intra: int | None = None,
    siz_b_L_parallel: int | None = None,
    siz_b_L_loop: int | None = None,
    siz_b_DH_parallel: int | None = None,
    siz_b_DH_loop: int | None = None,
    num_warps_intra: int | None = None,
    num_warps_inter: int | None = None,
    num_stages_intra: int | None = None,
    num_stages_inter: int | None = None,
    output_dtype: jnp.dtype = jnp.float32,
    eps: float = 0.0,
) -> tuple[
    jax.Array,  # matH_out (B, NH, S, DHV)
    jax.Array,  # vecN_out (B, NH, S)
    jax.Array,  # vecM_out (B, NH, S)
    None
    | (
        tuple[jax.Array, jax.Array, jax.Array]
    ),  # last_states (matC_states (B, NH, DHQK, DHV), vecN_states (B, NH, DHQK), scaMinter_states (B, NH))
    None | (tuple[jax.Array, jax.Array, jax.Array]),  # all_states (matC_states (B, NH, (NC+1) * DHQK, DHV),
    # vecN_states (B, NH, (NC+1) * DHQK), scaMinter_states (B, NH, (NC+1)))
]:
    """
    Execute the forward pass of the mLSTM chunkwise formulation.

    Args:
        matQ: Tensor containing the queries. Shape (B, NH, S, DHQK).
        matK: Tensor containing the keys. Shape (B, NH, S, DHQK).
        matV: Tensor containing the values. Shape (B, NH, S, DHV).
        vecI: Tensor containing the input gate. Shape (B, NH, S).
        vecF: Tensor containing the forget gate. Shape (B, NH, S).
        matC_initial: Initial state of the C matrix. Shape (B, NH, DHQK, DHV). Defaults to None.
        vecN_initial: Initial state of the N vector. Shape (B, NH, DHQK). Defaults to None.
        scaM_initial: Initial state of the M scalar. Shape (B, NH). Defaults to None.
        qk_scale: Scaling factor for the QK matrix. Defaults to None and will be inferred.
        return_last_states: Whether to return the last states. Defaults to False.
        return_all_states: Whether to return all states. Defaults to False.
        chunk_size_inter: Chunk size for the kernel inter chunk (recurrent) kernel. Defaults to None.
        chunk_size_intra: Chunk size for the kernel intra chunk (parallel) kernel. Defaults to None.
        siz_b_L_parallel: Size of the parallel L dimension for the parallel kernel. Defaults to None.
        siz_b_L_loop: Size of the loop L dimension for the parallel kernel. Defaults to None.
        siz_b_DH_parallel: Size of the parallel DH dimension for the parallel kernel. Defaults to None.
        siz_b_DH_loop: Size of the loop DH dimension for the parallel kernel. Defaults to None.
        num_warps_intra: Number of warps for the intra chunk kernel. Defaults to None.
        num_warps_inter: Number of warps for the inter chunk kernel. Defaults to None.
        num_stages_intra: Number of stages for the intra chunk kernel. Defaults to None.
        num_stages_inter: Number of stages for the inter chunk kernel. Defaults to None.
        output_dtype: Data type for the output. Defaults to jnp.float32.
        eps: Small value to avoid division by zero. Defaults to 1e-6.

    Returns:
        Tuple containing the output matrix H (shape (B, NH, S, DHV)), the N vector (shape (B, NH, S)),
        the M scalar (shape (B, NH)). Optionally, it might contain last states (matC_states,
        vecN_states, scaMinter_states) and optional all states (matC_states, vecN_states,
        scaMinter_states).
    """
    _B, _NH, S, DHQK = matQ.shape

    if chunk_size_inter is None:
        chunk_size_inter = min(128, S)
    if chunk_size_intra is None:
        chunk_size_intra = min(128, S)
    if siz_b_L_parallel is None:
        siz_b_L_parallel = min(64, chunk_size_intra)
    if siz_b_L_loop is None:
        siz_b_L_loop = min(64, chunk_size_intra)

    assert S % chunk_size_inter == 0, f"Sequence length {S} is not divisible by inter chunk size {chunk_size_inter}."

    assert S % chunk_size_intra == 0, f"Sequence length {S} is not divisible by intra chunk size {chunk_size_intra}."

    if qk_scale is None:
        qk_scale = DHQK**-0.5

    assert (
        chunk_size_inter <= chunk_size_intra
    ), f"chunk_size_inter {chunk_size_inter} must be >= chunk_size_intra {chunk_size_intra}"
    assert (
        chunk_size_intra % chunk_size_inter == 0
    ), f"chunk_size_intra {chunk_size_intra} must be divisible by chunk_size_inter {chunk_size_inter}"

    save_states_every_nth_chunk = chunk_size_intra // chunk_size_inter

    # materialize the  C_k, n_k, m_k states for each chunk
    matC_all, vecN_all, scaM_all = mlstm_chunkwise__recurrent_fw_C(
        matK=matK,
        matV=matV,
        vecF=vecF,
        vecI=vecI,
        matC_initial=matC_initial,
        vecN_initial=vecN_initial,
        scaMinter_initial=scaM_initial,
        chunk_size=chunk_size_inter,
        save_states_every_nth_chunk=save_states_every_nth_chunk,
        num_stages=num_stages_inter,
        num_warps=num_warps_inter,
    )

    # compute the outputs within each chunk
    matH_out, vecN_out, vecM_out = mlstm_chunkwise__parallel_fw_Hintra(
        matQ=matQ,
        matK=matK,
        matV=matV,
        vecI=vecI,
        vecF=vecF,
        matC_all=matC_all,
        vecN_all=vecN_all,
        scaM_all=scaM_all,
        qk_scale=qk_scale,
        chunk_size=chunk_size_intra,
        siz_b_LQ=siz_b_L_parallel,
        siz_b_LKV=siz_b_L_loop,
        siz_b_DHQK=siz_b_DH_loop,
        siz_b_DHHV=siz_b_DH_parallel,
        eps=eps,
        output_dtype=output_dtype,
        num_warps=num_warps_intra,
        num_stages=num_stages_intra,
    )

    ret_tuple = (
        matH_out,
        vecN_out,
        vecM_out,
    )
    if return_last_states:
        ret_tuple += (
            (
                matC_all[:, :, -DHQK:, :],
                vecN_all[:, :, -DHQK:],
                scaM_all[:, :, -1],
            ),
        )
    else:
        ret_tuple += (None,)

    if return_all_states:
        ret_tuple += ((matC_all, vecN_all, scaM_all),)
    else:
        ret_tuple += (None,)

    return ret_tuple  # (matH_out, vecN_out, vecM_out, optional(last_states), optional(all_states))
