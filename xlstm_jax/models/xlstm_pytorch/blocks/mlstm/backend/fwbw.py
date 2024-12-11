#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import math
from dataclasses import asdict, dataclass

import torch
from torch.amp import custom_bwd, custom_fwd

# from torch.amp import custom_fwd, custom_bwd
from torch.autograd.function import once_differentiable


def rev_cumsum_off(x):
    y = torch.zeros_like(x)
    y[..., :-1] = x[..., 1:]
    return y.flip(dims=(-1,)).cumsum(dim=-1).flip(dims=(-1,))


def rev_cumsum(x):
    return x.flip(dims=(-1,)).cumsum(-1).flip(dims=(-1,))


def causal_forget_matrix(forget_gates):
    S = forget_gates.shape[-1]

    fgates = forget_gates.view(-1, S)  # B, S   (batch dimension can incorporate others)
    mask = torch.tril(torch.ones([1, S, S], dtype=torch.bool, device=forget_gates.device), diagonal=-1)
    forget_matrix = fgates[:, :, None].repeat((1, 1, S))
    forget_matrix = forget_matrix.masked_fill(~mask, 0).cumsum(dim=1)
    mask = torch.tril(torch.ones(S, S, device=forget_gates.device, dtype=bool), diagonal=0)
    forget_matrix = forget_matrix.masked_fill(~mask, -torch.inf)
    return forget_matrix.view(*forget_gates.shape[:-1], S, S)


@dataclass
class mLSTMfwbwConfig:
    chunk_size: int | None = 512  # to be tuned
    return_state: bool = False
    use_initial_state: bool = False
    keep_G: bool = False
    keep_gates: bool = True
    keep_M: bool = False
    keep_c: bool = False
    stabilize_correctly: bool = False  # this is only needed, if the GroupNorm is omitted
    scale = None
    device_type: str = "cuda"

    def assign_model_config_params(self, model_config):
        pass


def mLSTMTorchFunction(config: mLSTMfwbwConfig):
    """
    Returns an autograd function that computes the gradient itself.

    config: mLSTMfwbwConfig     Configuration for mLSTMTorchFunc
    """
    _cfg = asdict(config)

    class mLSTMTorchFunc(torch.autograd.Function):
        @staticmethod
        @custom_fwd(device_type=_cfg["device_type"])
        def forward(ctx, q, k, v, i, f, initial_C, initial_n, initial_m):
            B, H, T, K, V = *q.shape, v.shape[-1]
            S = _cfg["chunk_size"] if _cfg["chunk_size"] is not None else T
            C = T // S
            use_chunkwise = _cfg["chunk_size"] is not None or _cfg["return_state"] or _cfg["use_initial_state"]

            # form chunks
            scale = 1 / math.sqrt(K)
            q = q.contiguous().view(B, H, C, S, K) * scale
            k = k.contiguous().view(B, H, C, S, K)
            v = v.contiguous().view(B, H, C, S, V)

            f = f.contiguous().view(B, H, C, S)
            log_fgates = torch.nn.functional.logsigmoid(f).view(B, H, C, S)
            i = i.contiguous().view(B, H, C, S)

            if use_chunkwise:
                lfacc = log_fgates.cumsum(dim=3)
                lflast = lfacc[:, :, :, -1]
                forward_gates_unstabilized = i + rev_cumsum_off(log_fgates)
                m_intra, _ = torch.max(forward_gates_unstabilized, dim=3)
                forward_gates = forward_gates_unstabilized - m_intra[..., None]
                backward_gates = lfacc

                kv = torch.einsum("bhctk,bhctd,bhct->bhckd", k, v, forward_gates.exp())
                ksum = torch.einsum("bhctk,bhct->bhck", k, forward_gates.exp())

                c = q.new_empty((B, H, C + 1, K, V))
                n = q.new_empty((B, H, C + 1, K))
                m = q.new_empty((B, H, C + 1))

                if initial_C is not None:
                    c[:, :, 0] = initial_C
                    n[:, :, 0] = initial_n
                    m[:, :, 0] = initial_m
                else:
                    c[:, :, 0] = 0.0
                    n[:, :, 0] = 0.0
                    m[:, :, 0] = 0.0

                for j in range(1, C + 1):
                    m[:, :, j] = torch.maximum(
                        (lflast[:, :, j - 1] + m[:, :, j - 1]),
                        m_intra[:, :, j - 1],
                    )
                    c[:, :, j] = torch.einsum(
                        "bhkd,bh->bhkd",
                        c[:, :, j - 1],
                        (lflast[:, :, j - 1] + m[:, :, j - 1] - m[:, :, j]).exp(),
                    ) + torch.einsum(
                        "bhkd,bh->bhkd",
                        kv[:, :, j - 1],
                        (m_intra[:, :, j - 1] - m[:, :, j]).exp(),
                    )
                    n[:, :, j] = torch.einsum(
                        "bhk,bh->bhk",
                        n[:, :, j - 1],
                        (lflast[:, :, j - 1] + m[:, :, j - 1] - m[:, :, j]).exp(),
                    ) + torch.einsum(
                        "bhk,bh->bhk",
                        ksum[:, :, j - 1],
                        (m_intra[:, :, j - 1] - m[:, :, j]).exp(),
                    )
            G_unstab = causal_forget_matrix(log_fgates) + i[:, :, :, None, :]
            G_stabilizer, _ = torch.max(G_unstab, dim=-1)

            if use_chunkwise:
                G_stabilizer = torch.maximum(G_stabilizer, m[:, :, :-1, None] + backward_gates)

            G = (G_unstab - G_stabilizer[..., None]).exp()
            M = torch.einsum("bhctk,bhcsk,bhcst->bhcst", k, q, G)
            if use_chunkwise:
                inter_n = torch.einsum(
                    "bhcsk,bhcs,bhck->bhcs",
                    q,
                    (backward_gates + m[:, :, :-1, None] - G_stabilizer).exp(),
                    n[:, :, :-1],
                )
            else:
                inter_n = 0.0

            M_norm = torch.maximum(
                (M.sum(dim=-1) + inter_n).abs(),
                (-G_stabilizer).exp() if _cfg["stabilize_correctly"] else torch.ones_like(G_stabilizer),
            )
            h = torch.einsum("bhctv,bhcst,bhcs->bhcsv", v, M, 1.0 / M_norm)

            if use_chunkwise:
                h += torch.einsum(
                    "bhckv,bhcsk,bhcs,bhcs->bhcsv",
                    c[:, :, :-1],
                    q,
                    (backward_gates + m[:, :, :-1, None] - G_stabilizer).exp(),
                    1.0 / M_norm,
                )

            if use_chunkwise:
                c_saved, n_saved = c[:, :, :-1], n[:, :, :-1]
                if not _cfg["keep_c"]:
                    c_saved = None
                if not _cfg["keep_gates"]:
                    forward_gates_unstabilized, backward_gates = None, None
            else:
                c_saved, n_saved, lflast, m, forward_gates_unstabilized, backward_gates = (
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
            if not _cfg["keep_G"]:
                G, G_stabilizer = None, None
            if not _cfg["keep_M"]:
                M = None
                if not use_chunkwise:
                    n_saved = None
            context = [
                q,
                k,
                v,
                i,
                f,
                log_fgates,
                M_norm,
                lflast,
                c_saved,
                n_saved,
                m,
                initial_C,
                initial_n,
                forward_gates_unstabilized,
                backward_gates,
                G,
                G_stabilizer,
            ]
            ctx.save_for_backward(*context)
            h = h.reshape(B, H, T, V)
            if _cfg["return_state"]:
                m = m[:, :, -1]
                n = n[:, :, -1]
                c = c[:, :, -1].requires_grad_(True)
                ctx.mark_non_differentiable(n, m)
                return h, c, n, m
            return h

        @staticmethod
        @once_differentiable
        @custom_bwd(device_type=_cfg["device_type"])
        def backward(ctx, dh, dc_last=None, dn_last=None, dm_last=None):
            use_chunkwise = _cfg["chunk_size"] is not None or _cfg["return_state"] or _cfg["use_initial_state"]
            _ = dn_last, dm_last
            # get stored tensors
            (
                q,
                k,
                v,
                i,
                f,
                log_fgates,
                M_norm,
                lflast,
                _c_saved,
                _n_saved,
                m,
                initial_C,
                initial_n,
                forward_gates_unstabilized,
                backward_gates,
                G,
                G_stabilizer,
            ) = ctx.saved_tensors

            B, H, C, S, K, V = *q.shape, v.shape[-1]
            T = S * C
            scale = 1 / math.sqrt(K)
            dh = dh.view(B, H, C, S, V)

            if not _cfg["keep_G"]:
                G_unstab = causal_forget_matrix(log_fgates) + i[:, :, :, None, :]
                G_stabilizer, _ = torch.max(G_unstab, dim=-1)

                if use_chunkwise:
                    G_stabilizer = torch.maximum(G_stabilizer, m[:, :, :-1, None] + backward_gates)
                # print(G_stabilizer.shape, m.shape, backward_gates.shape)

                G = (G_unstab - G_stabilizer[..., None]).exp()

            if not _cfg["keep_M"]:
                M = torch.einsum("bhctk,bhcsk,bhcst->bhcst", k, q, G)

            # inter chunk
            if use_chunkwise:
                if not _cfg["keep_c"]:
                    m_intra, _ = torch.max(forward_gates_unstabilized, dim=3)
                    forward_gates = forward_gates_unstabilized - m_intra[..., None]

                    kv = torch.einsum("bhctk,bhctd,bhct->bhckd", k, v, forward_gates.exp())
                    ksum = torch.einsum("bhctk,bhct->bhck", k, forward_gates.exp())

                    c = q.new_empty((B, H, C, K, V))
                    n = q.new_empty((B, H, C, K))

                    if initial_C is not None:
                        c[:, :, 0] = initial_C
                        n[:, :, 0] = initial_n
                    else:
                        c[:, :, 0] = 0.0
                        n[:, :, 0] = 0.0

                    for j in range(1, C):
                        c[:, :, j] = torch.einsum(
                            "bhkd,bh->bhkd",
                            c[:, :, j - 1],
                            (lflast[:, :, j - 1] + m[:, :, j - 1] - m[:, :, j]).exp(),
                        ) + torch.einsum(
                            "bhkd,bh->bhkd",
                            kv[:, :, j - 1],
                            (m_intra[:, :, j - 1] - m[:, :, j]).exp(),
                        )
                        n[:, :, j] = torch.einsum(
                            "bhk,bh->bhk",
                            n[:, :, j - 1],
                            (lflast[:, :, j - 1] + m[:, :, j - 1] - m[:, :, j]).exp(),
                        ) + torch.einsum(
                            "bhk,bh->bhk",
                            ksum[:, :, j - 1],
                            (m_intra[:, :, j - 1] - m[:, :, j]).exp(),
                        )

                if not _cfg["keep_gates"]:
                    backward_gates = log_fgates.cumsum(dim=3)
                    forward_gates_unstabilized = i + rev_cumsum_off(log_fgates)

                dc = q.new_empty(B, H, C + 1, K, V)
                if dc_last is not None:
                    dc[:, :, -1] = dc_last
                else:
                    dc[:, :, -1] = 0.0

                for j in range(C, 0, -1):
                    dc[:, :, j - 1] = (lflast[:, :, j - 1] + m[:, :, j - 1] - m[:, :, j]).exp()[..., None, None] * dc[
                        :, :, j
                    ] + torch.einsum(
                        "bhs,bhse,bhsd,bhs->bhde",
                        (backward_gates[:, :, j - 1] + m[:, :, j - 1, None] - G_stabilizer[:, :, j - 1]).exp(),
                        dh[:, :, j - 1],
                        q[:, :, j - 1],
                        1.0 / M_norm[:, :, j - 1],
                    )

            # intra chunk
            dQK = torch.einsum("bhcse,bhcst,bhcte,bhcs->bhcst", dh, G, v, 1 / M_norm)
            dq = torch.einsum("bhcst,bhctd->bhcsd", dQK, k * scale)
            dk = torch.einsum(
                "bhcsd,bhcst->bhctd",
                q,
                dQK,
            )
            dv = torch.einsum("bhcse,bhcst,bhcs->bhcte", dh, M, 1 / M_norm)

            if use_chunkwise:
                dq += torch.einsum(
                    "bhcs,bhcse,bhcde,bhcs->bhcsd",
                    (backward_gates + m[:, :, :-1, None] - G_stabilizer).exp(),
                    dh,
                    c * scale,
                    1.0 / M_norm,
                )
                dk += torch.einsum(
                    "bhct,bhcde,bhcte->bhctd",
                    (forward_gates_unstabilized - m[:, :, 1:, None]).exp(),
                    dc[:, :, 1:],
                    v,
                )
                dv += torch.einsum(
                    "bhct,bhcde,bhctd->bhcte",
                    (forward_gates_unstabilized - m[:, :, 1:, None]).exp(),
                    dc[:, :, 1:],
                    k,
                )
            dflacc = (dq * q - dk * k).sum(dim=-1).view(B, H, T)
            di = (dv * v).sum(dim=-1).view(B, H, T)

            dfl = rev_cumsum(dflacc)
            # print("dfl", dfl)
            df = dfl * torch.sigmoid(-f.view(B, H, T))

            dq = dq.reshape(B, H, T, K)
            dk = dk.reshape(B, H, T, K)
            dv = dv.reshape(B, H, T, V)

            if _cfg["use_initial_state"]:
                return (
                    dq,
                    dk,
                    dv,
                    di,
                    df,
                    dc[:, :, 0],
                    q.new_empty((B, H, K)).zero_(),
                    q.new_empty((B, H)).zero_(),
                )
            return dq, dk, dv, di.view(B, H, T, 1), df.view(B, H, T, 1), None, None, None

    return mLSTMTorchFunc


class mLSTMfwbw(torch.nn.Module):
    config_class = mLSTMfwbwConfig

    def __init__(self, config: mLSTMfwbwConfig):
        super().__init__()
        self.config = config
        self.func = mLSTMTorchFunction(config=config)

    def forward(self, *args):
        if self.config.use_initial_state:
            return self.func.apply(*args)
        return self.func.apply(*args, None, None, None)


# def mLSTM_parallel(q, k, v, i, f, initial_C=None, initial_n=None, initial_m=None):
#     return _mLSTMParallel.apply(q, k, v, i, f, initial_C, initial_n, initial_m)
