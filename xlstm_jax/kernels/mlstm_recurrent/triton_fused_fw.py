import jax
import jax_triton as jt
import triton

from mlstm_kernels.mlstm_kernels.kernel_utils import is_power_of_2
from mlstm_kernels.mlstm_kernels.mlstm.recurrent.triton_fused_fw import _recurrent_step_fw_kernel

from xlstm_jax.kernels.kernel_utils import jax2triton_dtype
from xlstm_jax.kernels.stride_utils import get_stride


def recurrent_step_fw(
    matC_state: jax.Array,  # (B, NH, DHQK, DHV)
    vecN_state: jax.Array,  # (B, NH, DHQK)
    scaM_state: jax.Array,  # (B, NH, 1)
    vecQ: jax.Array,  # (B, NH, DHQK)
    vecK: jax.Array,  # (B, NH, DHQK)
    vecV: jax.Array,  # (B, NH, DHV)
    scaI: jax.Array,  # (B, NH, 1)
    scaF: jax.Array,  # (B, NH, 1)
    matC_new: jax.Array | None = None,  # (B, NH, DHQK, DHV)
    vecN_new: jax.Array | None = None,  # (B, NH, DHQK)
    scaM_new: jax.Array | None = None,  # (B, NH, 1)
    qk_scale: float | None = None,
    eps: float = 1e-6,
):
    B, NH, DHQK, DHHV = matC_state.shape

    DTYPE = matC_state.dtype

    if qk_scale is None:
        qk_scale = DHQK**-0.5

    if matC_new is None:
        assert vecN_new is None and scaM_new is None, "Initial states must be provided together."
        matC_new = jax.ShapeDtypeStruct(shape=matC_state.shape, dtype=matC_state.dtype)
        vecN_new = jax.ShapeDtypeStruct(shape=vecN_state.shape, dtype=vecN_state.dtype)
        scaM_new = jax.ShapeDtypeStruct(shape=scaM_state.shape, dtype=scaM_state.dtype)
    else:
        assert vecN_new is not None and scaM_new is not None, "Initial states must be provided together."

    min_siz_b_DHQK = 64
    min_siz_b_DHHV = 64

    assert (
        is_power_of_2(DHQK) or DHQK % min_siz_b_DHQK == 0
    ), f"DHQK must be a power of 2 or multiple of {min_siz_b_DHQK}. Got {DHQK}."
    assert (
        is_power_of_2(DHHV) or DHHV % min_siz_b_DHHV == 0
    ), f"DHHV must be a power of 2 or multiple of {min_siz_b_DHHV}. Got {DHHV}."

    siz_b_DHQK = min(min_siz_b_DHQK, triton.next_power_of_2(DHQK))
    siz_b_DHHV = min(min_siz_b_DHHV, triton.next_power_of_2(DHHV))

    # num_b_DHQK = triton.cdiv(DHQK, siz_b_DHQK)
    num_b_DHHV = triton.cdiv(DHHV, siz_b_DHHV)

    grid = (1, num_b_DHHV, B * NH)
    num_warps = 4 if siz_b_DHQK >= 64 else 2
    num_stages = 1

    # create output tensors
    vecH = jax.ShapeDtypeStruct(shape=vecV.shape, dtype=vecV.dtype)

    vecH, matC_new, vecN_new, scaM_new = jt.triton_call(
        matC_state,
        vecN_state,
        scaM_state,
        vecQ,
        vecK,
        vecV,
        scaI,
        scaF,
        out_shape=(vecH, matC_new, vecN_new, scaM_new),
        qk_scale=qk_scale,
        str_matC_B_NH=get_stride(matC_state, 1),
        str_matC_DHQK=get_stride(matC_state, 2),
        str_matC_DHHV=get_stride(matC_state, 3),
        str_vecN_B_NH=get_stride(vecN_state, 1),
        str_vecN_DHQK=get_stride(vecN_state, 2),
        str_scaM_B_NH=get_stride(scaM_state, 1),
        str_vecQK_NH=get_stride(vecQ, 1),
        str_vecQK_DHQK=get_stride(vecQ, 2),
        str_vecVH_B_NH=get_stride(vecV, 1),
        str_vecVH_DHHV=get_stride(vecV, 2),
        str_scaIF_B_NH=get_stride(scaI, 1),
        B=B,
        NH=NH,
        DHQK=DHQK,
        DHHV=DHHV,
        siz_b_DHQK=siz_b_DHQK,
        siz_b_DHHV=siz_b_DHHV,
        EPS=eps,
        DTYPE=jax2triton_dtype(DTYPE),
        grid=grid,
        kernel=_recurrent_step_fw_kernel,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return vecH, (matC_new, vecN_new, scaM_new)
