"""This file contains the kernel that combines the recurrent and parallel
part of the forward pass of the mLSTM chunkwise formulation.
It should allow arbitrary large chunk sizes and head dimensions.
"""

import jax
import jax.numpy as jnp

from ._chunkwise_gates import compute_chunkwise_log_gates_vecB_vecA
from ._parallel_bw_dK import mlstm_chunkwise__parallel_bw_dK
from ._parallel_bw_dQ import mlstm_chunkwise__parallel_bw_dQ
from ._parallel_bw_dV import mlstm_chunkwise__parallel_bw_dV
from ._recurrent_bw import mlstm_chunkwise__recurrent_bw_dC
from ._recurrent_fw import mlstm_chunkwise__recurrent_fw_C


def mlstm_chunkwise_bw(
    # Forward arguments
    matQ: jax.Array,  # (B, NH, S, DHQK)
    matK: jax.Array,  # (B, NH, S, DHQK)
    matV: jax.Array,  # (B, NH, S, DHV)
    vecI: jax.Array,  # (B, NH, S)
    vecF: jax.Array,  # (B, NH, S)
    matC_initial: jax.Array | None = None,  # (B, NH, DHQK, DHV)
    vecN_initial: jax.Array | None = None,  # (B, NH, DHQK)
    scaM_initial: jax.Array | None = None,  # (B, NH)
    # Backward arguments
    matC_all: jax.Array | None = None,  # (B, NH, (NCsaved+1) * DHQK, DHV)
    vecN_all: jax.Array | None = None,  # (B, NH, (NCsaved+1) * DHQK)
    scaM_all: jax.Array | None = None,  # (B, NH, (NCsaved+1))
    vecN_out: jax.Array | None = None,  # (B, NH, S)
    vecM_out: jax.Array | None = None,  # (B, NH, S)
    matDeltaH: jax.Array | None = None,  # (B, NH, S, DHV)
    matDeltaC_last: jax.Array | None = None,  # (B, NH, DHQK, DHV)
    # Other arguments
    qk_scale: float | None = None,
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
    eps: float = 0.0,
):
    _B, _NH, S, DHQK = matQ.shape

    if chunk_size_inter is None:
        chunk_size_inter = min(128, S)
    if chunk_size_intra is None:
        chunk_size_intra = min(128, S)
    if siz_b_L_parallel is None:
        siz_b_L_parallel = min(64, chunk_size_intra)
    if siz_b_L_loop is None:
        siz_b_L_loop = min(64, chunk_size_intra)

    assert S % chunk_size_inter == 0, f"Sequence length {S} is not divisible by chunk size inter {chunk_size_inter}."
    assert S % chunk_size_intra == 0, f"Sequence length {S} is not divisible by chunk size intra {chunk_size_intra}."

    if qk_scale is None:
        qk_scale = DHQK**-0.5

    assert (
        chunk_size_inter <= chunk_size_intra
    ), f"chunk_size_inter {chunk_size_inter} must be >= chunk_size_intra {chunk_size_intra}"
    assert (
        chunk_size_intra % chunk_size_inter == 0
    ), f"chunk_size_intra {chunk_size_intra} must be divisible by chunk_size_inter {chunk_size_inter}"

    save_states_every_nth_chunk = chunk_size_intra // chunk_size_inter

    # recompute the "all" states if needed
    if matC_all is None:
        assert (
            (matC_all is None) and (vecN_all is None) and (scaM_all is None)
        ), "Either all or none of the states must be provided."

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

    # recurrent backward: compute the deltaC gradients
    # matDeltaC_states (B, NH, (NC+1) * DHQK, DHHV)
    matDeltaC_states = mlstm_chunkwise__recurrent_bw_dC(
        matQ=matQ,  # (B, NH, S, DHQK)
        vecF=vecF,  # (B, NH, S)
        scaM_inter=scaM_all,  # (B, NH, NCintra+1)
        vecM_combine=vecM_out,  # (B, NH, S)
        matDeltaH=matDeltaH,  # (B, NH, S, DHV)
        vecN_out=vecN_out,  # (B, NH, S)
        matDeltaC_last=matDeltaC_last,  # (B, NH, DHQK, DHV)
        qk_scale=qk_scale,
        chunk_size=chunk_size_inter,
        eps=eps,
        save_states_every_nth_chunk=save_states_every_nth_chunk,
        num_stages=num_stages_inter,
        num_warps=num_warps_inter,
    )

    # parallel backward: compute the deltaQ, deltaK, deltaV gradients
    vecB, vecA = compute_chunkwise_log_gates_vecB_vecA(
        chunk_size=chunk_size_intra, vecI=vecI, vecF=vecF, return_vecB_only=False
    )
    grad_output_dtype = matQ.dtype

    matDeltaV = mlstm_chunkwise__parallel_bw_dV(
        matQ=matQ,
        matK=matK,
        matV=matV,
        vecI=vecI,
        vecB=vecB,
        vecA=vecA,
        matC_all=matC_all,
        vecN_all=vecN_all,
        scaM_all=scaM_all,
        vecN_out=vecN_out,
        vecM_out=vecM_out,
        matDeltaH=matDeltaH,
        matDeltaC_states=matDeltaC_states,
        qk_scale=qk_scale,
        chunk_size=chunk_size_intra,
        siz_b_LQ=siz_b_L_loop,
        siz_b_LKV=siz_b_L_parallel,
        siz_b_DHQK=siz_b_DH_loop,
        siz_b_DHHV=siz_b_DH_parallel,
        num_warps=num_warps_intra,
        num_stages=num_stages_intra,
        eps=eps,
        output_dtype=grad_output_dtype,
    )

    matDeltaK = mlstm_chunkwise__parallel_bw_dK(
        matQ=matQ,
        matK=matK,
        matV=matV,
        vecI=vecI,
        vecB=vecB,
        vecA=vecA,
        matC_all=matC_all,
        vecN_all=vecN_all,
        scaM_all=scaM_all,
        vecN_out=vecN_out,
        vecM_out=vecM_out,
        matDeltaH=matDeltaH,
        matDeltaC_states=matDeltaC_states,
        qk_scale=qk_scale,
        chunk_size=chunk_size_intra,
        siz_b_LQ=siz_b_L_loop,
        siz_b_LKV=siz_b_L_parallel,
        siz_b_DHQK=siz_b_DH_parallel,
        siz_b_DHHV=siz_b_DH_loop,
        num_warps=num_warps_intra,
        num_stages=num_stages_intra,
        eps=eps,
        output_dtype=grad_output_dtype,
    )

    matDeltaQ = mlstm_chunkwise__parallel_bw_dQ(
        matQ=matQ,
        matK=matK,
        matV=matV,
        vecI=vecI,
        vecB=vecB,
        vecA=vecA,
        matC_all=matC_all,
        vecN_all=vecN_all,
        scaM_all=scaM_all,
        vecN_out=vecN_out,
        vecM_out=vecM_out,
        matDeltaH=matDeltaH,
        matDeltaC_states=matDeltaC_states,
        qk_scale=qk_scale,
        chunk_size=chunk_size_intra,
        siz_b_LQ=siz_b_L_parallel,
        siz_b_LKV=siz_b_L_loop,
        siz_b_DHQK=siz_b_DH_parallel,
        siz_b_DHHV=siz_b_DH_loop,
        num_warps=num_warps_intra,
        num_stages=num_stages_intra,
        eps=eps,
        output_dtype=grad_output_dtype,
    )

    # postprocessing: compute deltaF and deltaI gradients
    # vecF = rearrange(vecF, "b nh nc l -> b nh (nc l)")
    # compute the vecDeltaFbar values with dfbar = rev_cumsum((q*dq - k*dk).sum(-1))
    matQ = matQ.astype(jnp.float32)
    matK = matK.astype(jnp.float32)
    matDeltaQ = matDeltaQ.astype(jnp.float32)
    matDeltaK = matDeltaK.astype(jnp.float32)
    vecDeltaFbar_acc = ((matQ * matDeltaQ) - (matK * matDeltaK)).sum(-1)
    vecDeltaFbar = jnp.flip(jnp.cumsum(jnp.flip(vecDeltaFbar_acc, axis=-1).astype(jnp.float32), axis=-1), axis=-1)
    vecDeltaF = vecDeltaFbar * jax.nn.sigmoid(-vecF)

    # compute deltaI
    # both are equivalent:
    # vecDeltaI = (matV * matDeltaV).sum(-1)
    vecDeltaI = (matK * matDeltaK).sum(-1)

    # vecDeltaI = torch.zeros((B, NH, S), dtype=vecI.dtype, device=vecI.device)

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
