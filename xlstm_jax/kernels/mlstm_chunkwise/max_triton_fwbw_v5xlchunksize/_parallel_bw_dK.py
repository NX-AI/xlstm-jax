from mlstm_kernels.mlstm_kernels.kernel_utils import is_power_of_2
from mlstm_kernels.mlstm_kernels.mlstm.chunkwise.max_triton_fwbw_v5xlchunksize._triton_parallel_bw_dK import (
    _mlstm_chunkwise_parallel_bw_dK_kernel,
)

import jax
import jax.numpy as jnp
import jax_triton as jt
import triton

from ...kernel_utils import jax2triton_dtype
from ...stride_utils import get_stride


def mlstm_chunkwise__parallel_bw_dK(
    # Forward arguments
    matQ: jax.Array,  # (B, NH, S, DHQK)
    matK: jax.Array,  # (B, NH, S, DHQK)
    matV: jax.Array,  # (B, NH, S, DHHV)
    vecI: jax.Array,  # (B, NH, NC, L)
    vecA: jax.Array,  # (B, NH, NC, L)
    vecB: jax.Array,  # (B, NH, NC, L)
    # Backward arguments
    matC_all: jax.Array,  # (B, NH, (NC+1) * DHQK, DHHV)
    vecN_all: jax.Array,  # (B, NH, (NC+1) * DHQK)
    scaM_all: jax.Array,  # (B, NH, (NC+1))
    vecN_out: jax.Array,  # (B, NH, S) # vecN_combine
    vecM_out: jax.Array,  # (B, NH, S) # vecM_combine
    matDeltaH: jax.Array,  # (B, NH, S, DHHV)
    matDeltaC_states: jax.Array,  # (B, NH, (NC+1) * DHQK, DHHV)
    # Other arguments
    qk_scale: float | None = None,
    chunk_size: int = 64,
    siz_b_LQ: int = 32,
    siz_b_LKV: int = 32,
    siz_b_DHQK: int | None = None,
    siz_b_DHHV: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
    eps: float = 0.0,
    output_dtype: jnp.dtype = jnp.float32,
) -> jax.Array:  # matDeltaK (B, NH, S, DHQK)
    """
    Computes only the deltaK gradients for the backward pass.
    The other gradients are computed in the other (kernel) function.

    This function defines the grid and block sizes for the kernel launch and calls the kernel.
    chunk parallel size:        siz_b_LKV
    chunk loop size:            siz_b_LQ
    head dim parallel size:     siz_b_DHQK
    head dim loop size:         siz_b_DHHV

    Args:
        matQ: Tensor containing the query vectors. Shape (B, NH, S, DHQK).
        matK: Tensor containing the key vectors. Shape (B, NH, S, DHQK).
        matV: Tensor containing the value vectors. Shape (B, NH, S, DHHV).
        vecI: Tensor containing the input gate pre-activations. Shape (B, NH, NC, L).
        vecA: Tensor containing the summed input and cumulative forget gate pre-activations. Shape (B, NH, NC, L).
        vecB: Tensor containing the cumulative forget gate pre-activations. Shape (B, NH, NC, L).
        matCstate_all: Tensor containing the C states at the chunk borders. Shape (B, NH, (NC+1) * DHQK, DHHV).
        vecNstate_all: Tensor containing the N states at the chunk borders. Shape (B, NH, (NC+1) * DHQK).
        scaMstate_all: Tensor containing the M states at the chunk borders. Shape (B, NH, (NC+1)).
        matH_out: Tensor containing the output H. Shape (B, NH, S, DHHV).
        vecN_out: Tensor containing the normalizer output N. Shape (B, NH, S).
        vecM_out: Tensor containing the max state M. Shape (B, NH, S).
        matDeltaH_out: Tensor containing the incoming H gradients. Shape (B, NH, S, DHHV).
        matDeltaC_states: Tensor containing the incoming C gradients. Shape (B, NH, (NC+1) * DHQK, DHHV).
        vecDeltaN_states: Tensor containing the incoming N gradients. Shape (B, NH, (NC+1) * DHQK).
        qk_scale: Scale factor for the QK matrix. Defaults to None.
        chunk_size: Chunk size. Defaults to 64.
        siz_b_LQ: Block size for the chunk dimension LQ. Defaults to 32.
        siz_b_LKV: Block size for the chunk dimension LKV. Defaults to 32.
        siz_b_DHQK: Block size for the head dimension DHQK. Defaults to None.
        siz_b_DHHV: Block size for the head dimension DHHV. Defaults to None.
        num_warps: Number of warps. Defaults to None.
        num_stages: Number of stages. Defaults to None.
        eps: Epsilon value. Defaults to 1e-6.
        output_dtype: Output data type. Defaults to jnp.float32.

    Returns:
        Tensor containing the K gradients. Shape (B, NH, S, DHQK).
    """
    B, NH, S, DHQK = matQ.shape
    DHHV = matV.shape[-1]

    assert S % chunk_size == 0, f"Sequence length {S} must be divisible by chunk size {chunk_size}"
    NC = S // chunk_size
    L = chunk_size

    assert is_power_of_2(L), "Chunk size must be a power of 2."

    if qk_scale is None:
        qk_scale = DHQK**-0.5

    siz_b_DHQK = min(128, triton.next_power_of_2(DHQK)) if siz_b_DHQK is None else siz_b_DHQK
    siz_b_DHHV = min(64, triton.next_power_of_2(DHHV)) if siz_b_DHHV is None else siz_b_DHHV

    assert siz_b_LQ <= L, "siz_b_LQ must be less than or equal to chunk size L"
    assert siz_b_LKV <= L, "siz_b_LKV must be less than or equal to chunk size L"
    assert siz_b_LQ <= siz_b_LKV, "siz_b_LQ must be less than or equal to siz_b_LKV"
    assert siz_b_LKV % siz_b_LQ == 0, "siz_b_LKV must be divisible by siz_b_LQ"

    num_b_DHQK = triton.cdiv(DHQK, siz_b_DHQK)
    num_b_LKV = triton.cdiv(L, siz_b_LKV)

    num_stages = 1 if num_stages is None else num_stages
    if num_warps is None:
        num_warps = 4 if (siz_b_DHQK >= 64 or siz_b_DHHV >= 64) else 2

    matDeltaK = jax.ShapeDtypeStruct(shape=(B, NH, S, DHQK), dtype=output_dtype)

    grid = (num_b_DHQK, num_b_LKV, NC * B * NH)
    matDeltaK = jt.triton_call(
        matQ,
        matK,
        matV,
        vecI,
        vecB,
        vecA,
        matC_all,
        vecN_all,
        scaM_all,
        vecN_out,
        vecM_out,
        matDeltaH,
        matDeltaC_states,
        out_shape=(matDeltaK),
        qk_scale=qk_scale,
        str_matQK_B_NH=get_stride(matQ, axis=1),
        str_matQK_S=get_stride(matQ, axis=2),
        str_matQK_DHQK=get_stride(matQ, axis=3),
        str_matHV_B_NH=get_stride(matV, axis=1),
        str_matHV_S=get_stride(matV, axis=2),
        str_matHV_DHHV=get_stride(matV, axis=3),
        str_vecABI_B_NH=get_stride(vecB, axis=1),
        str_vecABI_NC=get_stride(vecB, axis=2),
        str_matCstate_B_NH=get_stride(matC_all, axis=1),
        str_matCstate_NCDHQK=get_stride(matC_all, axis=2),
        str_matCstate_DHHV=get_stride(matC_all, axis=3),
        str_vecNstate_B_NH=get_stride(vecN_all, axis=1),
        str_scaMstate_B_NH=get_stride(scaM_all, axis=1),
        str_vecMN_B_NH=get_stride(vecN_out, axis=1),
        str_vecMN_S=get_stride(vecN_out, axis=2),
        B=B,
        NH=NH,
        S=S,
        DHQK=DHQK,
        DHHV=DHHV,
        NC=NC,
        L=L,
        siz_b_LQ=siz_b_LQ,
        siz_b_LKV=siz_b_LKV,
        siz_b_DHQK=siz_b_DHQK,
        siz_b_DHHV=siz_b_DHHV,
        DTYPE=jax2triton_dtype(matQ.dtype),
        OUTPUT_DTYPE=jax2triton_dtype(output_dtype),
        EPS=eps,
        num_stages=num_stages,
        num_warps=num_warps,
        grid=grid,
        kernel=_mlstm_chunkwise_parallel_bw_dK_kernel,
    )

    return matDeltaK
