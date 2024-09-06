# Copyright (c) NXAI GmbH and its affiliates 2024
# Korbinian Pöppel
#
# Adapted from flash-linear-attention

import torch
import triton
import triton.language as tl
from torch.cuda.amp import custom_bwd, custom_fwd

from .tl_utils import contiguous


@triton.jit
def chunk_mlstm_fwd_kernel_C(
    k,
    v,
    C,
    n,
    m,
    i,  # log igates
    f,  # accumulated log fgate
    initial_C,  # initial state of the chunk [B, H, D_head_K, D_head_V]
    initial_n,
    initial_m,
    final_C,  # final state of the chunk [B, H, D_head_K, D_head_V]
    final_n,
    final_m,
    s_qk_h,
    s_qk_t,
    s_qk_d,
    s_vh_h,
    s_vh_t,
    s_vh_d,
    s_C_h,
    s_C_t,
    s_n_h,
    H: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
):
    i_k, i_v, i_bC = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    if USE_INITIAL_STATE:
        p_C0 = tl.make_block_ptr(
            initial_C + i_bC * K * V,
            (K, V),
            (V, 1),
            (i_k * BK, i_v * BV),
            (BK, BV),
            (1, 0),
        )
        p_n0 = tl.make_block_ptr(
            initial_n + i_bC * K,
            (K,),
            (1,),
            (i_k * BK,),
            (BK,),
            (0,),
        )
        p_m0 = initial_m

        b_C = tl.load(p_C0, boundary_check=(0, 1)).to(tl.float32)
        b_n = tl.load(p_n0, boundary_check=(0,)).to(tl.float32)
        b_m = tl.load(p_m0).to(tl.float32)
    else:
        b_C = tl.zeros([BK, BV], dtype=tl.float32)
        b_n = tl.zeros([BK], dtype=tl.float32)
        b_m = 0.0

    b_m_next = 0.0
    for i_t in range(NT):
        p_k = tl.make_block_ptr(
            k + i_bC * s_qk_h,
            (K, T),
            (s_qk_d, s_qk_t),
            (i_k * BK, i_t * BT),
            (BK, BT),
            (0, 1),
        )
        p_v = tl.make_block_ptr(
            v + i_bC * s_vh_h,
            (T, V),
            (s_vh_t, s_vh_d),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        p_C = tl.make_block_ptr(
            C + i_bC * s_C_h + i_t * K * V,
            (K, V),
            (s_C_t, 1),
            (i_k * BK, i_v * BV),
            (BK, BV),
            (1, 0),
        )
        p_n = tl.make_block_ptr(
            n + i_bC * s_n_h + i_t * K, (K,), (1,), (i_k * BK,), (BK,), (0,)
        )
        tl.store(p_C, b_C.to(p_C.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_n, b_n.to(p_n.dtype.element_ty), boundary_check=(0,))
        tl.store(m + i_bC * (NT + 1) + i_t, b_m)
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BK, BV]
        b_f_last = tl.load(f + i_bC * T + i_t * BT + BT - 1)
        b_f = tl.load(f + i_bC * T + i_t * BT + tl.arange(0, BT))
        b_i = tl.load(i + i_bC * T + i_t * BT + tl.arange(0, BT))
        b_g = b_i + b_f_last - b_f
        b_m_next, _ = tl.max(b_g)
        b_m_next = tl.maximum(b_f_last + b_m, b_m_next)

        b_C *= tl.math.exp2(b_f_last - b_m_next + b_m)
        b_n *= tl.math.exp2(b_f_last - b_m_next + b_m)
        b_C += tl.dot(
            b_k,
            (b_v * tl.math.exp2(b_g - b_m_next)[:, None]).to(b_k.dtype),
            allow_tf32=False,
        )
        b_n += tl.sum(b_k * tl.math.exp2(b_g - b_m_next), axis=1)
        b_m = b_m_next

    if STORE_FINAL_STATE:
        p_Ct = tl.make_block_ptr(
            final_C + i_bC * K * V,
            (K, V),
            (V, 1),
            (i_k * BK, i_v * BV),
            (BK, BV),
            (1, 0),
        )
        p_n = tl.make_block_ptr(
            final_n + i_bC * K, (K,), (1,), (i_k * BK,), (BK,), (0,)
        )
        tl.store(p_Ct, b_C.to(p_Ct.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_n, b_n.to(p_n.dtype.element_ty), boundary_check=(0,))
        tl.store(final_m + i_bC, b_m)
        tl.store(m + i_bC * (NT + 1) + NT, b_m)


@triton.jit
def chunk_mlstm_fwd_kernel_h(
    q,
    k,
    v,
    C,
    n,
    m,
    m_total,
    i,
    f,
    h,
    norm,
    s_qk_h,
    s_qk_t,
    s_qk_d,
    s_vh_h,
    s_vh_t,
    s_vh_d,
    s_C_h,
    s_C_t,
    s_n_h,
    scale,
    H: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
):
    i_v, i_t, i_bC = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    h_i = tl.arange(0, BT)
    m_s = h_i[:, None] >= h_i[None, :]

    b_h = tl.zeros([BT, BV], dtype=tl.float32)
    b_s = tl.zeros([BT, BT], dtype=tl.float32)
    b_norm = tl.zeros([BT, BV], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(
            q + i_bC * s_qk_h,
            (T, K),
            (s_qk_t, s_qk_d),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_k = tl.make_block_ptr(
            k + i_bC * s_qk_h,
            (K, T),
            (s_qk_d, s_qk_t),
            (i_k * BK, i_t * BT),
            (BK, BT),
            (0, 1),
        )
        p_C = tl.make_block_ptr(
            C + i_bC * s_C_h + i_t * K * V,
            (K, V),
            (s_C_t, 1),
            (i_k * BK, i_v * BV),
            (BK, BV),
            (1, 0),
        )
        p_n = tl.make_block_ptr(
            n + i_bC * s_n_h + i_t * K,
            (K, BV),
            (1, 0),
            (i_k * BK, 0),
            (BK, BV),
            (0, 1),
        )

        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BT]

        # [BK, BV]
        b_C = tl.load(p_C, boundary_check=(0, 1))
        b_n = tl.load(p_n, boundary_check=(0,))
        b_h += tl.dot(b_q, b_C, allow_tf32=False)
        b_s += tl.dot(b_q, b_k, allow_tf32=False)
        b_n2 = tl.dot(b_q, b_n, allow_tf32=False)
        b_norm += b_n2

    p_f = f + i_bC * T + i_t * BT + tl.arange(0, BT)
    b_f = tl.load(p_f)
    p_i = i + i_bC * T + i_t * BT + tl.arange(0, BT)
    b_i = tl.load(p_i)
    b_m = tl.load(m + i_bC * (NT + 1) + i_t)

    # # TODO revise this to the stabilized version of Mamba2
    b_logD = b_i[None, :] + b_f[:, None] - b_f[None, :]
    b_logD = tl.where(m_s, b_logD, -float("inf"))
    b_mlogD = tl.max(b_logD, axis=1)

    b_m_total = tl.maximum(b_f + b_m, b_mlogD)
    p_m_total = tl.make_block_ptr(
        m_total + T * i_bC, (T,), (1,), (i_t * BT,), (BT,), (0,)
    )
    tl.store(p_m_total, b_m_total.to(p_m_total.dtype.element_ty), boundary_check=(0,))

    b_D = tl.math.exp2(b_logD - b_m_total[:, None])

    b_h = b_h * tl.math.exp2(b_f + b_m - b_m_total)[:, None] * scale
    b_s = b_s * b_D * scale
    b_norm = b_norm * tl.math.exp2(b_f + b_m - b_m_total)[:, None] * scale

    b_s = tl.where(m_s, b_s, 0)
    b_norm += tl.sum(b_s, axis=1)[:, None]
    b_norm = tl.abs(b_norm)
    b_norm = tl.maximum(b_norm, tl.math.exp2(-b_m_total)[:, None])

    tl.store(norm + i_bC * T + i_t * BT + tl.arange(0, BT), tl.max(b_norm, axis=1))

    p_v = tl.make_block_ptr(
        v + i_bC * s_vh_h,
        (T, V),
        (s_vh_t, s_vh_d),
        (i_t * BT, i_v * BV),
        (BT, BV),
        (1, 0),
    )
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_h = (b_h + tl.dot(b_s.to(b_v.dtype), b_v, allow_tf32=False)) / b_norm
    p_h = tl.make_block_ptr(
        h + i_bC * s_vh_h,
        (T, V),
        (s_vh_t, s_vh_d),
        (i_t * BT, i_v * BV),
        (BT, BV),
        (1, 0),
    )
    tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_mlstm_bwd_kernel_dC(
    q,
    f,
    m,
    m_total,
    norm,
    dh,
    dC,
    final_dC,
    final_m,
    initial_dC,
    initial_m,
    s_qk_h,
    s_qk_t,
    s_qk_d,
    s_vh_h,
    s_vh_t,
    s_vh_d,
    s_C_h,
    s_C_t,
    scale,
    H: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
):
    i_k, i_v, i_bC = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    # [BK, BV]
    p_dC = tl.make_block_ptr(
        final_dC + i_bC * K * V,
        (K, V),
        (s_C_t, 1),
        (i_k * BK, i_v * BV),
        (BK, BV),
        (1, 0),
    )
    b_dC = tl.load(p_dC, boundary_check=(0, 1))
    b_m = tl.load(final_m + i_bC)
    for i_t in range(NT - 1, -1, -1):
        p_q = tl.make_block_ptr(
            q + i_bC * s_qk_h,
            (K, T),
            (s_qk_d, s_qk_t),
            (i_k * BK, i_t * BT),
            (BK, BT),
            (0, 1),
        )
        p_dh = tl.make_block_ptr(
            dh + i_bC * s_vh_h,
            (T, V),
            (s_vh_t, s_vh_d),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        p_dC = tl.make_block_ptr(
            dC + i_bC * s_C_h + i_t * K * V,
            (K, V),
            (s_C_t, 1),
            (i_k * BK, i_v * BV),
            (BK, BV),
            (1, 0),
        )

        tl.store(p_dC, b_dC.to(p_dC.dtype.element_ty), boundary_check=(0, 1))
        b_f_last = tl.load(f + i_bC * T + i_t * BT + BT - 1)
        b_f = tl.load(f + i_bC * T + i_t * BT + tl.arange(0, BT))
        b_m_p = tl.load(m + i_bC * (NT + 1) + i_t)
        b_m_total = tl.load(m_total + i_bC * T + i_t * BT + tl.arange(0, BT))
        b_norm = tl.load(norm + i_bC * T + i_t * BT + tl.arange(0, BT))

        # [BK, BT]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale * tl.math.exp2(b_f + b_m_p - b_m_total)[None, :]).to(
            b_q.dtype
        )
        # [BT, V]
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        b_dh /= b_norm[:, None]
        # [BK, BV]
        b_dC *= tl.math.exp2(b_f_last + b_m_p - b_m)
        # print("DC", b_dC)
        b_dC += tl.dot(b_q, b_dh.to(b_q.dtype), allow_tf32=False)
        b_m = b_m_p

    p_initial_dC = tl.make_block_ptr(
        initial_dC + i_bC * K * V,
        (K, V),
        (V, 1),
        (i_k * BK, i_v * BV),
        (BK, BV),
        (1, 0),
    )
    tl.store(
        p_initial_dC, b_dC.to(p_initial_dC.dtype.element_ty), boundary_check=(0, 1)
    )
    tl.store(initial_m + i_bC, b_m)


@triton.jit
def chunk_mlstm_bwd_kernel_dqkvif(
    q,
    k,
    v,
    C,
    m,
    m_total,
    norm,
    i,
    f,
    dh,
    dC,
    dq,
    dk,
    dv,
    s_qk_h,
    s_qk_t,
    s_qk_d,
    s_vh_h,
    s_vh_t,
    s_vh_d,
    s_C_h,
    s_C_t,
    scale,
    B: tl.constexpr,
    H: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
):
    i_k, i_t, i_bC = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    n_bh = tl.num_programs(2)
    o_i = tl.arange(0, BT)

    p_q = tl.make_block_ptr(
        q + i_bC * s_qk_h,
        (K, T),
        (s_qk_d, s_qk_t),
        (i_k * BK, i_t * BT),
        (BK, BT),
        (0, 1),
    )
    p_k = tl.make_block_ptr(
        k + i_bC * s_qk_h,
        (T, K),
        (s_qk_t, s_qk_d),
        (i_t * BT, i_k * BK),
        (BT, BK),
        (1, 0),
    )

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_s = tl.dot(b_k, b_q, allow_tf32=False)
    p_f = f + i_bC * T + i_t * BT + tl.arange(0, BT)
    p_i = i + i_bC * T + i_t * BT + tl.arange(0, BT)
    b_f = tl.load(p_f)
    b_f_last = tl.load(f + i_bC * T + i_t * BT + BT - 1)
    b_m = tl.load(m + i_bC * (NT + 1) + i_t)
    b_m_total = tl.load(m_total + i_bC * T + i_t * BT + tl.arange(0, BT))
    b_norm = tl.load(norm + i_bC * T + i_t * BT + tl.arange(0, BT))
    b_i = tl.load(p_i)

    # TODO: update to stable version of Mamba2
    mask = tl.math.exp2(b_i[:, None] + b_f[None, :] - b_f[:, None] - b_m_total[None, :])
    mask = tl.where(o_i[:, None] <= o_i[None, :], mask * scale, 0)
    b_s = b_s * mask

    b_m_next = tl.load(m + i_bC * (NT + 1) + i_t + 1)

    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_ds = tl.zeros([BT, BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(
            v + i_bC * s_vh_h,
            (T, V),
            (s_vh_t, s_vh_d),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        p_C = tl.make_block_ptr(
            C + i_bC * s_C_h,
            (V, NT * K),
            (1, s_C_t),
            (i_v * BV, i_t * K + i_k * BK),
            (BV, BK),
            (0, 1),
        )
        p_dh = tl.make_block_ptr(
            dh + i_bC * s_vh_h,
            (T, V),
            (s_vh_t, s_vh_d),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        p_dC = tl.make_block_ptr(
            dC + i_bC * s_C_h,
            (NT * K, V),
            (s_C_t, 1),
            (i_t * K + i_k * BK, i_v * BV),
            (BK, BV),
            (1, 0),
        )
        p_dv = tl.make_block_ptr(
            dv + (i_k * n_bh + i_bC) * s_vh_h,
            (T, V),
            (s_vh_t, s_vh_d),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        # [BV, BK]
        b_C = tl.load(p_C, boundary_check=(0, 1))
        # [BK, BV]
        b_dC = tl.load(p_dC, boundary_check=(0, 1))
        # [BT, BT]
        b_ds += tl.dot(b_dh, tl.trans(b_v), allow_tf32=False)
        # [BT, BK]
        b_dq += tl.dot(b_dh, b_C, allow_tf32=False) * scale
        b_dk += tl.dot(b_v, tl.trans(b_dC), allow_tf32=False)
        # [BT, BV]
        b_dv = tl.dot(b_k, b_dC, allow_tf32=False) * tl.math.exp2(
            b_i - b_f + b_f_last - b_m_next
        )[:, None] + tl.dot(b_s.to(b_q.dtype) / b_norm[None, :], b_dh, allow_tf32=False)

        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

    b_dq = b_dq * tl.math.exp2(b_f + b_m - b_m_total)[:, None] / b_norm[:, None]
    b_dk = b_dk * tl.math.exp2(b_i - b_f + b_f_last - b_m_next)[:, None]

    b_ds = b_ds * tl.trans(mask)
    b_ds = b_ds.to(b_k.dtype)
    # [BT, BK]
    b_dq += tl.dot(b_ds, b_k, allow_tf32=False) / b_norm[:, None]
    b_dk += tl.trans(tl.dot(b_q / b_norm[None, :], b_ds, allow_tf32=False))

    p_dq = tl.make_block_ptr(
        dq + i_bC * s_qk_h,
        (T, K),
        (s_qk_t, s_qk_d),
        (i_t * BT, i_k * BK),
        (BT, BK),
        (1, 0),
    )
    p_dk = tl.make_block_ptr(
        dk + i_bC * s_qk_h,
        (T, K),
        (s_qk_t, s_qk_d),
        (i_t * BT, i_k * BK),
        (BT, BK),
        (1, 0),
    )
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))


def mLSTMFunc(chunk_size, save_states: bool = True):
    class mLSTMFunction(torch.autograd.Function):
        @staticmethod
        @custom_fwd
        @contiguous
        def forward(
            ctx, q, k, v, i, f, initial_C, initial_n, initial_m, output_final_state
        ):
            B, H, T, K, V = *q.shape, v.shape[-1]
            BT = chunk_size  # more does not work due to shared memory constraints
            BK, BV = (
                min(64, triton.next_power_of_2(K)),
                min(64, triton.next_power_of_2(V)),
            )
            NT, NK, NV = triton.cdiv(T, BT), triton.cdiv(K, BK), triton.cdiv(V, BV)
            num_stages = 1
            num_warps = 4 if BK == 64 else 2
            scale = K**-0.5

            assert T % BT == 0, "sequence length must be divisible by BT"
            f_orig = f
            f = torch.nn.functional.logsigmoid(f)
            f = f.reshape(B, H, -1, BT)
            f = f.cumsum(-1) * 1.44269504
            f = f.reshape(B, H, -1)
            i = i.reshape(B, H, -1) * 1.44269504

            final_C, final_n, final_m = None, None, None
            if output_final_state:
                final_C = q.new_empty(
                    B, H, K, V, dtype=torch.float32, requires_grad=False
                )
                final_n = q.new_empty(B, H, K, dtype=torch.float32, requires_grad=False)
                final_m = q.new_empty(B, H, dtype=torch.float32, requires_grad=False)

            C = q.new_empty(B, H, NT * K, V)
            n = q.new_empty(B, H, NT, K)
            m = q.new_empty(B, H, NT + 1)
            m_total = q.new_empty(B, H, NT, BT)
            norm = q.new_empty(B, H, NT, BT)
            grid = (NK, NV, B * H)
            chunk_mlstm_fwd_kernel_C[grid](
                k,
                v,
                C,
                n,
                m,
                i,
                f,
                initial_C,
                initial_n,
                initial_m,
                final_C,
                final_n,
                final_m,
                q.stride(1),
                q.stride(2),
                q.stride(3),
                v.stride(1),
                v.stride(2),
                v.stride(3),
                C.stride(1),
                C.stride(2),
                n.stride(1),
                H=H,
                T=T,
                K=K,
                V=V,
                BT=BT,
                BK=BK,
                BV=BV,
                NT=NT,
                USE_INITIAL_STATE=initial_C is not None,
                STORE_FINAL_STATE=output_final_state,
                num_warps=num_warps,
                num_stages=num_stages,
            )
            grid = (NV, NT, B * H)
            h = torch.empty_like(v)

            chunk_mlstm_fwd_kernel_h[grid](
                q,
                k,
                v,
                C,
                n,
                m,
                m_total,
                i,
                f,
                h,
                norm,
                q.stride(1),
                q.stride(2),
                q.stride(3),
                v.stride(1),
                v.stride(2),
                v.stride(3),
                C.stride(1),
                C.stride(2),
                n.stride(1),
                scale,
                H=H,
                T=T,
                K=K,
                V=V,
                BT=BT,
                BK=BK,
                BV=BV,
                NT=NT,
                num_warps=num_warps,
                num_stages=num_stages,
            )
            if save_states:
                ctx.save_for_backward(
                    q,
                    k,
                    v,
                    C,
                    f,
                    i,
                    m,
                    m_total,
                    norm,
                    final_m,
                    initial_C,
                    initial_n,
                    f_orig,
                )
            else:
                ctx.save_for_backward(
                    q,
                    k,
                    v,
                    initial_n,
                    f,
                    i,
                    m,
                    m_total,
                    norm,
                    final_m,
                    initial_C,
                    initial_n,
                    f_orig,
                )
            return h.to(q.dtype), final_C, final_n, final_m

        @staticmethod
        @custom_bwd
        @contiguous
        def backward(ctx, dh, final_dC=None, final_dn=None, final_dm=None):
            if save_states:
                (
                    q,
                    k,
                    v,
                    C,
                    f,
                    i,
                    m,
                    m_total,
                    norm,
                    final_m,
                    initial_C,
                    initial_n,
                    f_orig,
                ) = ctx.saved_tensors
            else:
                (
                    q,
                    k,
                    v,
                    initial_n,
                    f,
                    i,
                    m,
                    m_total,
                    norm,
                    final_m,
                    initial_C,
                    initial_n,
                    f_orig,
                ) = ctx.saved_tensors

            B, H, T, K, V = *q.shape, v.shape[-1]
            BT = chunk_size  # more does not work due to shared memory constraints
            BK, BV = (
                min(32 if q.dtype == torch.float32 else 64, triton.next_power_of_2(K)),
                min(32 if q.dtype == torch.float32 else 64, triton.next_power_of_2(V)),
            )
            NT, NK, NV = triton.cdiv(T, BT), triton.cdiv(K, BK), triton.cdiv(V, BV)
            num_stages = 1
            num_warps = 4 if BK == 64 else 2
            scale = K**-0.5
            dC = q.new_empty(B, H, NT * K, V)

            initial_dC = q.new_empty(
                B, H, K, V, dtype=torch.float32, requires_grad=False
            )
            initial_m = q.new_empty(B, H, dtype=torch.float32, requires_grad=False)

            if final_dC is None:
                final_dC = q.new_full(initial_dC.shape, 0.0)
                final_m = q.new_full(initial_m.shape, 0.0)

            grid = (NK, NV, B * H)
            chunk_mlstm_bwd_kernel_dC[grid](
                q,
                f,
                m,
                m_total,
                norm,
                dh,
                dC,
                final_dC,
                final_m,
                initial_dC,
                initial_m,
                q.stride(1),
                q.stride(2),
                q.stride(3),
                v.stride(1),
                v.stride(2),
                v.stride(3),
                dC.stride(1),
                dC.stride(2),
                scale,
                H=H,
                T=T,
                K=K,
                V=V,
                BT=BT,
                BK=BK,
                BV=BV,
                NT=NT,
                # num_warps=num_warps,
                # num_stages=num_stages,
            )
            if not save_states:
                C = q.new_empty(B, H, NT * K, V)
                n = q.new_empty(B, H, NT, K)
                m_tmp = q.new_empty(B, H, NT + 1)
                grid = (NK, NV, B * H)
                final_C, final_n, final_m = None, None, None
                chunk_mlstm_fwd_kernel_C[grid](
                    k,
                    v,
                    C,
                    n,
                    m_tmp,
                    i,
                    f,
                    initial_C,
                    initial_n,
                    initial_m,
                    final_C,
                    final_n,
                    final_m,
                    q.stride(1),
                    q.stride(2),
                    q.stride(3),
                    v.stride(1),
                    v.stride(2),
                    v.stride(3),
                    C.stride(1),
                    C.stride(2),
                    n.stride(1),
                    H=H,
                    T=T,
                    K=K,
                    V=V,
                    BT=BT,
                    BK=BK,
                    BV=BV,
                    NT=NT,
                    USE_INITIAL_STATE=initial_C is not None,
                    STORE_FINAL_STATE=False,
                    num_warps=num_warps,
                    num_stages=num_stages,
                )

            grid = (NK, NT, B * H)
            dq = torch.empty_like(q)
            dk = torch.empty_like(k)
            dv = v.new_empty(NK, *v.shape)
            num_stages = 1
            num_warps = 4 if BK == 64 else 2
            chunk_mlstm_bwd_kernel_dqkvif[grid](
                q,
                k,
                v,
                C,
                m,
                m_total,
                norm,
                i,
                f,
                dh,
                dC,
                dq,
                dk,
                dv,
                q.stride(1),
                q.stride(2),
                q.stride(3),
                v.stride(1),
                v.stride(2),
                v.stride(3),
                dC.stride(1),
                dC.stride(2),
                scale,
                B=B,
                H=H,
                T=T,
                K=K,
                V=V,
                BT=BT,
                BK=BK,
                BV=BV,
                NT=NT,
                # num_warps=num_warps,
                # num_stages=num_stages,
            )
            dv = dv.sum(0)
            df = (dq * q - dk * k).sum(-1)
            di = (dv * v).sum(-1)

            def rev_cumsum(x):
                return x.flip(dims=(-1,)).cumsum(-1).flip(dims=(-1,))

            df = rev_cumsum(df)
            df = df * torch.nn.functional.sigmoid(-f_orig)
            return (
                dq.to(q.dtype),
                dk.to(k.dtype),
                dv.to(v.dtype),
                di.to(i.dtype),
                df.to(f.dtype).view(f.shape),
                initial_dC if initial_C is not None else None,
                None,
                None,
                None,
            )

    return mLSTMFunction


# other chunk sizes do not work as patches are forced to be in shared memory and are limited therefore
# need to update parallel kernel (h kernels) with one more for loop
mLSTMFunction = mLSTMFunc(chunk_size=64)


def mlstm_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    i: torch.Tensor,  # input gate
    f: torch.Tensor,  # forget gate
    initial_C: torch.Tensor = None,
    initial_n: torch.Tensor = None,
    initial_m: torch.Tensor = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if initial_C is not None:
        initial_C = initial_C.detach()
        initial_n = initial_n.detach()
        initial_m = initial_m.detach()
    f = f.float()
    i = i.float()
    h, final_C, final_n, final_m = mLSTMFunction(chunk_size=64).apply(
        q, k, v, i, f, initial_C, initial_n, initial_m, output_final_state
    )
    if output_final_state:
        return h, (final_C, final_n, final_m)
    else:
        return h
