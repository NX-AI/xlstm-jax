import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
from flax import linen as nn

from .config import mLSTMBackend


@dataclass
class mLSTMBackendFwbwConfig:
    """
    Configuration object for the mLSTM fwbw backend.
    """

    chunk_size: int | None = 512  # to be tuned
    """Chunk size for the kernel computation."""
    return_state: bool = False
    """Whether to return the last state. USeful for inference."""
    use_initial_state: bool = False
    """Whether to start from an initial state or zeros."""
    keep_G: bool = False
    """Whether to save the G matrix for the backward pass."""
    keep_gates: bool = True
    """Whether to save the gates for the backward pass."""
    keep_M: bool = False
    """Whether to save the M matrix for the backward pass."""
    keep_c: bool = False
    """Whether to save the c matrix for the backward pass."""
    stabilize_correctly: bool = False  # this is only needed, if the GroupNorm is omitted
    """Whether to stabilize the output correctly. This is only needed if no GroupNorm is applied after the mLSTM.
    If GroupNorm is applied, this can be set to False, as results after GroupNorm will be the same."""

    def assign_model_config_params(self, *args, **kwargs):
        pass


def rev_cumsum_off(x: jax.Array):
    """Compute the reverse cumulative sum of a tensor with an offset."""
    y = jnp.concatenate(
        [
            x[..., 1:],
            jnp.zeros_like(x[..., :1]),
        ],
        axis=-1,
    )
    y = jnp.flip(y, axis=(-1,))
    y = y.cumsum(axis=-1)
    y = jnp.flip(y, axis=(-1,))
    return y


def rev_cumsum(x: jax.Array):
    """Compute the reverse cumulative sum of a tensor."""
    # NOTE: This may not be the most efficient way to compute the reverse cumulative sum.
    # A faster alternative can be (x.sum(axis=-1, keepdims=True) - x.cumsum(axis=-1) + x).
    # However, in low precision, this may lead to different numerical results. Thus,
    # the current implementation is used.
    x = jnp.flip(x, axis=(-1,))
    x = x.cumsum(axis=-1)
    x = jnp.flip(x, axis=(-1,))
    return x


def causal_forget_matrix(forget_gates: jax.Array):
    """Compute the causal forget matrix from the forget gates."""
    S = forget_gates.shape[-1]

    fgates = forget_gates.reshape(-1, S)  # B, S   (batch dimension can incorporate others)
    forget_matrix = fgates[:, :, None].repeat(repeats=S, axis=-1)
    forget_matrix = jnp.tril(forget_matrix, k=-1)
    forget_matrix = forget_matrix.cumsum(axis=1)
    mask = jnp.tril(jnp.ones(shape=(S, S), dtype=jnp.bool_), k=0)
    forget_matrix = jnp.where(mask, forget_matrix, -float("inf"))
    return forget_matrix.reshape(*forget_gates.shape[:-1], S, S)


def fwbw_forward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    i: jax.Array,
    f: jax.Array,
    config: mLSTMBackendFwbwConfig,
    initial_C: jax.Array | None = None,
    initial_n: jax.Array | None = None,
    initial_m: jax.Array | None = None,
) -> tuple[jax.Array, Sequence[jax.Array]]:
    """
    Forward pass of the mLSTM fwbw backend.

    Args:
        q: query tensor
        k: key tensor
        v: value tensor
        i: input gate tensor
        f: forget gate tensor
        config: configuration object
        initial_C: initial chunk tensor. Defaults to None.
        initial_n: initial n tensor. Defaults to None.
        initial_m: initial m tensor. Defaults to None.

    Returns:
        Output tensor and context for backward.
    """
    B, T, K, V = *q.shape, v.shape[-1]
    S = config["chunk_size"] if config["chunk_size"] is not None else T
    C = T // S
    use_chunkwise = config["chunk_size"] is not None or config["return_state"] or config["use_initial_state"]

    # form chunks
    scale = 1 / math.sqrt(K)
    q = q.reshape(B, C, S, K) * scale
    k = k.reshape(B, C, S, K)
    v = v.reshape(B, C, S, V)

    f = f.reshape(B, C, S)
    log_fgates = jax.nn.log_sigmoid(f).reshape(B, C, S)
    i = i.reshape(B, C, S)

    if use_chunkwise:
        lfacc = log_fgates.cumsum(axis=-1)
        lflast = lfacc[:, :, -1]
        forward_gates_unstabilized = i + rev_cumsum_off(log_fgates)
        m_intra = jnp.max(forward_gates_unstabilized, axis=-1)
        forward_gates = forward_gates_unstabilized - m_intra[..., None]
        backward_gates = lfacc

        kv = jnp.einsum("bctk,bctd,bct->bckd", k, v, jnp.exp(forward_gates))
        ksum = jnp.einsum("bctk,bct->bck", k, jnp.exp(forward_gates))

        def _inner_step(carry, xs):
            m_last, c_last, n_last, j = carry
            m_new = jnp.maximum(
                (lflast[:, j - 1] + m_last),
                m_intra[:, j - 1],
            )
            lfast_m_up = jnp.exp(lflast[:, j - 1] + m_last - m_new)
            m_intra_new = jnp.exp(m_intra[:, j - 1] - m_new)
            c_new = jnp.einsum(
                "bkd,b->bkd",
                c_last,
                lfast_m_up,
            ) + jnp.einsum(
                "bkd,b->bkd",
                kv[:, j - 1],
                m_intra_new,
            )
            n_new = jnp.einsum(
                "bk,b->bk",
                n_last,
                lfast_m_up,
            ) + jnp.einsum(
                "bk,b->bk",
                ksum[:, j - 1],
                m_intra_new,
            )
            return (m_new, c_new, n_new, j + 1), (m_new, c_new, n_new)

        if initial_C is None:
            # Initialized in float32 following the kernel dtypes.
            initial_C = jnp.zeros_like(q, shape=(B, K, V), dtype=jnp.float32)
            initial_n = jnp.zeros_like(q, shape=(B, K), dtype=jnp.float32)
            initial_m = jnp.zeros_like(q, shape=(B,), dtype=jnp.float32)

        _, (m, c, n) = jax.lax.scan(
            _inner_step,
            init=(initial_m, initial_C, initial_n, 1),
            length=C,
        )
        c = jnp.concatenate([initial_C[None], c], axis=0).swapaxes(0, 1)
        n = jnp.concatenate([initial_n[None], n], axis=0).swapaxes(0, 1)
        m = jnp.concatenate([initial_m[None], m], axis=0).swapaxes(0, 1)

    G_unstab = causal_forget_matrix(log_fgates) + i[:, :, None, :]
    G_stabilizer = jnp.max(G_unstab, axis=-1)

    if use_chunkwise:
        G_stabilizer = jnp.maximum(G_stabilizer, m[:, :-1, None] + backward_gates)

    G = jnp.exp(G_unstab - G_stabilizer[..., None])
    M = jnp.einsum("bctk,bcsk,bcst->bcst", k, q, G)
    if use_chunkwise:
        inter_n = jnp.einsum(
            "bcsk,bcs,bck->bcs",
            q,
            jnp.exp(backward_gates + m[:, :-1, None] - G_stabilizer),
            n[:, :-1],
        )
    else:
        inter_n = 0.0

    M_norm = jnp.maximum(
        jnp.abs(M.sum(axis=-1) + inter_n),
        jnp.exp(-G_stabilizer) if config["stabilize_correctly"] else jnp.ones_like(G_stabilizer),
    )
    h = jnp.einsum("bctv,bcst,bcs->bcsv", v, M, 1.0 / M_norm)

    if use_chunkwise:
        h += jnp.einsum(
            "bckv,bcsk,bcs,bcs->bcsv",
            c[:, :-1],
            q,
            jnp.exp(backward_gates + m[:, :-1, None] - G_stabilizer),
            1.0 / M_norm,
        )

    if use_chunkwise:
        c_saved, n_saved = c[:, :-1], n[:, :-1]
        if not config["keep_c"]:
            c_saved = None
        if not config["keep_gates"]:
            forward_gates_unstabilized, backward_gates = None, None
    else:
        c_saved, n_saved, lflast, m, forward_gates_unstabilized, backward_gates = (None,) * 6
    if not config["keep_G"]:
        G, G_stabilizer = None, None
    if not config["keep_M"]:
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
    h = h.reshape(B, T, V)
    return h, context


def fwbw_backward(
    ctx: Sequence[jax.Array],
    dh: jax.Array,
    config: mLSTMBackendFwbwConfig,
    dc_last: jax.Array = None,
    dn_last: jax.Array = None,
    dm_last: jax.Array = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array | None, jax.Array | None, jax.Array | None]:
    """
    Backward pass of the mLSTM fwbw backend.

    Args:
        ctx (Sequence[jax.Array]): context from forward pass.
        dh (jax.Array): gradient tensor.
        config (mLSTMfwbwConfig): configuration object.
        dc_last (jax.Array, optional): last chunk tensor. Defaults to None.
        dn_last (jax.Array, optional): last n tensor. Defaults to None.
        dm_last (jax.Array, optional): last m tensor. Defaults to None.

    Returns:
        tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array,
              jax.Array | None, jax.Array | None, jax.Array | None]: gradients.
    """
    use_chunkwise = config["chunk_size"] is not None or config["return_state"] or config["use_initial_state"]
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
        c_saved,
        n_saved,
        m,
        initial_C,
        initial_n,
        forward_gates_unstabilized,
        backward_gates,
        G,
        G_stabilizer,
    ) = ctx

    B, C, S, K, V = *q.shape, v.shape[-1]
    T = S * C
    scale = 1 / math.sqrt(K)
    dh = dh.reshape(B, C, S, V)

    if not config["keep_G"]:
        G_unstab = causal_forget_matrix(log_fgates) + i[:, :, None, :]
        G_stabilizer = jnp.max(G_unstab, axis=-1)

        if use_chunkwise:
            G_stabilizer = jnp.maximum(G_stabilizer, m[:, :-1, None] + backward_gates)

        G = jnp.exp(G_unstab - G_stabilizer[..., None])

    if not config["keep_M"]:
        M = jnp.einsum("bctk,bcsk,bcst->bcst", k, q, G)

    # inter chunk
    if use_chunkwise:
        if not config["keep_c"]:
            m_intra = jnp.max(forward_gates_unstabilized, axis=-1)
            forward_gates = forward_gates_unstabilized - m_intra[..., None]

            kv = jnp.einsum("bctk,bctd,bct->bckd", k, v, jnp.exp(forward_gates))
            ksum = jnp.einsum("bctk,bct->bck", k, jnp.exp(forward_gates))

            c = jnp.empty_like(q, shape=(B, C, K, V))
            n = jnp.empty_like(q, shape=(B, C, K))

            if initial_C is not None:
                c = c.at[:, 0].set(initial_C)
                n = n.at[:, 0].set(initial_n)
            else:
                c = c.at[:, 0].set(0.0)
                n = n.at[:, 0].set(0.0)

            # This can be rewritten as a scan, but it is not clear if it gives speed benefits.
            for j in range(1, C):
                c_new = jnp.einsum(
                    "bkd,b->bkd",
                    c[:, j - 1],
                    jnp.exp(lflast[:, j - 1] + m[:, j - 1] - m[:, j]),
                ) + jnp.einsum(
                    "bkd,b->bkd",
                    kv[:, j - 1],
                    jnp.exp(m_intra[:, j - 1] - m[:, j]),
                )
                c = c.at[:, j].set(c_new)
                n_new = jnp.einsum(
                    "bk,b->bk",
                    n[:, j - 1],
                    jnp.exp(lflast[:, j - 1] + m[:, j - 1] - m[:, j]),
                ) + jnp.einsum(
                    "bk,b->bk",
                    ksum[:, j - 1],
                    jnp.exp(m_intra[:, j - 1] - m[:, j]),
                )
                n = n.at[:, j].set(n_new)

        if not config["keep_gates"]:
            backward_gates = log_fgates.cumsum(axis=-1)
            forward_gates_unstabilized = i + rev_cumsum_off(log_fgates)

        dc = jnp.empty_like(q, shape=(B, C + 1, K, V))
        if dc_last is not None:
            dc = dc.at[:, -1].set(dc_last)
        else:
            dc = dc.at[:, -1].set(0.0)

        # This can be rewritten as a scan, but it is not clear if it gives speed benefits.
        for j in range(C, 0, -1):
            d_new = jnp.exp(lflast[:, j - 1] + m[:, j - 1] - m[:, j])[..., None, None] * dc[:, j] + jnp.einsum(
                "bs,bse,bsd,bs->bde",
                jnp.exp(backward_gates[:, j - 1] + m[:, j - 1, None] - G_stabilizer[:, j - 1]),
                dh[:, j - 1],
                q[:, j - 1],
                1.0 / M_norm[:, j - 1],
            )
            dc = dc.at[:, j - 1].set(d_new)

    # intra chunk
    dQK = jnp.einsum("bcse,bcst,bcte,bcs->bcst", dh, G, v, 1 / M_norm)
    dq = jnp.einsum("bcst,bctd->bcsd", dQK, k * scale)
    dk = jnp.einsum(
        "bcsd,bcst->bctd",
        q,
        dQK,
    )
    dv = jnp.einsum("bcse,bcst,bcs->bcte", dh, M, 1 / M_norm)

    if use_chunkwise:
        dq += jnp.einsum(
            "bcs,bcse,bcde,bcs->bcsd",
            jnp.exp(backward_gates + m[:, :-1, None] - G_stabilizer),
            dh,
            c * scale,
            1.0 / M_norm,
        )
        dk += jnp.einsum(
            "bct,bcde,bcte->bctd",
            jnp.exp(forward_gates_unstabilized - m[:, 1:, None]),
            dc[:, 1:],
            v,
        )
        dv += jnp.einsum(
            "bct,bcde,bctd->bcte",
            jnp.exp(forward_gates_unstabilized - m[:, 1:, None]),
            dc[:, 1:],
            k,
        )
    dflacc = (dq * q - dk * k).sum(axis=-1).reshape(B, T)
    di = (dv * v).sum(axis=-1).reshape(B, T)

    dfl = rev_cumsum(dflacc)
    df = dfl * jax.nn.sigmoid(-f.reshape(B, T))

    dq = dq.reshape(B, T, K)
    dk = dk.reshape(B, T, K)
    dv = dv.reshape(B, T, V)

    if config["use_initial_state"]:
        return (
            dq,
            dk,
            dv,
            di,
            df,
            dc[:, 0],
            jnp.zeros_like(q, shape=(B, K)),
            jnp.zeros_like(q, shape=(B,)),
        )
    else:
        return dq, dk, dv, di.reshape(B, T, 1), df.reshape(B, T, 1), None, None, None


def mlstm_fwbw_custom_grad(
    config: mLSTMBackendFwbwConfig,
) -> Callable[
    [jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array | None, jax.Array | None, jax.Array | None],
    jax.Array,
]:
    """
    Returns an autograd function that computes the gradient itself.

    Args:
        config (mLSTMfwbwConfig): configuration object.

    Returns:
        function: autograd function.
    """
    config = asdict(config)

    @jax.custom_gradient
    def mlstm_jax_fn(
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        i: jax.Array,
        f: jax.Array,
        initial_C: jax.Array | None = None,
        initial_n: jax.Array | None = None,
        initial_m: jax.Array | None = None,
    ):
        h, ctx = fwbw_forward(q, k, v, i, f, config, initial_C, initial_n, initial_m)
        h_dtype = h.dtype
        h = h.astype(q.dtype)

        def backward(dh):
            dh = dh.astype(h_dtype)
            dq, dk, dv, di, df, dC, dn, dm = fwbw_backward(ctx, dh, config)
            dq = dq.astype(q.dtype)
            dk = dk.astype(k.dtype)
            dv = dv.astype(v.dtype)
            di = di.astype(i.dtype)
            df = df.astype(f.dtype)
            if dC is not None and initial_C is not None:
                dC = dC.astype(initial_C.dtype)
            if dn is not None and initial_n is not None:
                dn = dn.astype(initial_n.dtype)
            if dm is not None and initial_m is not None:
                dm = dm.astype(initial_m.dtype)
            return dq, dk, dv, di, df, dC, dn, dm

        return h, backward

    return mlstm_jax_fn


class mLSTMBackendFwbw(mLSTMBackend):
    config_class = mLSTMBackendFwbwConfig

    @nn.compact
    def __call__(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        i: jax.Array,
        f: jax.Array,
    ) -> jax.Array:
        """Forward pass of the mLSTM fwbw backend."""
        return mlstm_fwbw_custom_grad(self.config)(q, k, v, i, f)

    @property
    def can_vmap_over_heads(self) -> bool:
        """
        Whether the backend can be vmaped over the heads dimension.

        The backend is written independent of the heads dimension, and thus can be vmapped.

        Returns:
            bool: True
        """
        return True
