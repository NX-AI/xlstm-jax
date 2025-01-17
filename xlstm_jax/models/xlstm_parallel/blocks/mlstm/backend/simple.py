#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

# pylint: disable=invalid-name
import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import linen as nn

from .config import mLSTMBackend


def parallel_stabilized_simple(
    queries: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    igate_preact: jax.Array,
    fgate_preact: jax.Array,
    lower_triangular_matrix: jax.Array = None,
    stabilize_rowwise: bool = True,
    eps: float = 1e-6,
    qkv_dtype: jnp.dtype | None = None,
    gate_dtype: jnp.dtype | None = None,
) -> jax.Array:
    """
    This is the mLSTM cell in parallel form.

    This version is stabilized. We control the range of exp() arguments by
    ensuring that they are always smaller than 0.0 by subtracting the maximum.

    Args:
        queries (jax.Array): (B, NH, S, DHQK)
        keys (jax.Array): (B, NH, S, DHQK)
        values (jax.Array): (B, NH, S, DHV)
        igate_preact (jax.Array): (B, NH, S, 1)
        fgate_preact (jax.Array): (B, NH, S, 1)
        lower_triangular_matrix (jax.Array, optional): (S,S). Defaults to None.
        stabilize_rowwise (bool, optional): Wether to stabilize the combination matrix C rowwise (take maximum per row).
            Alternative: Subtract the maximum over all rows. Defaults to True.
        eps (float, optional): Small value to avoid division by zero. Defaults to 1e-6.
        qkv_dtype (jnp.dtype, optional): dtype of queries, keys and values. Defaults to None,
            which infers the dtype from the inputs.
        gate_dtype (jnp.dtype, optional): dtype of igate_preact and fgate_preact. Defaults to None,
            which infers the dtype from the inputs.

    Returns:
        jax.Array: (B, NH, S, DH), h_tilde_state
    """

    B, S, DHQK = queries.shape
    _, _, DHV = values.shape
    if gate_dtype is None:
        gate_dtype = igate_preact.dtype
        assert gate_dtype == fgate_preact.dtype, (
            f"igate_preact and fgate_preact must have the same dtype, got {igate_preact.dtype}"
            f" for igate_preact and {fgate_preact.dtype} for fgate_preact."
        )
    else:
        igate_preact = igate_preact.astype(gate_dtype)
        fgate_preact = fgate_preact.astype(gate_dtype)
    if qkv_dtype is None:
        qkv_dtype = queries.dtype
        assert qkv_dtype == keys.dtype == values.dtype, (
            f"queries, keys and values must have the same dtype, got {queries.dtype} for queries,"
            f" {keys.dtype} for keys and {values.dtype} for values."
        )
    else:
        queries = queries.astype(qkv_dtype)
        keys = keys.astype(qkv_dtype)
        values = values.astype(qkv_dtype)
    assert (
        queries.shape == keys.shape == (B, S, DHQK)
    ), f"queries and keys must have shape (B, S, DHQK), got {queries.shape} and {keys.shape}."
    assert values.shape == (B, S, DHV), f"values must have shape (B, S, DHV), got {values.shape}."
    assert (
        igate_preact.shape == fgate_preact.shape == (B, S, 1)
    ), f"igate_preact and fgate_preact must have shape (B, S, 1), got {igate_preact.shape}, {fgate_preact.shape}."
    if lower_triangular_matrix is not None:
        assert lower_triangular_matrix.shape == (
            S,
            S,
        ), f"lower_triangular_matrix must have shape (S, S), got {lower_triangular_matrix.shape}."

    # forget gate matrix
    log_fgates = jax.nn.log_sigmoid(fgate_preact)  # (B, NH, S, 1)
    if lower_triangular_matrix is None or S < lower_triangular_matrix.shape[-1]:
        ltr = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
    else:
        ltr = lower_triangular_matrix
    assert ltr.dtype == jnp.bool_, f"lower_triangular_matrix must be of dtype bool, got {ltr.dtype}"

    log_fgates_cumsum = jnp.concat(
        [
            jnp.zeros((B, 1, 1), dtype=gate_dtype),
            jnp.cumsum(log_fgates, axis=-2),
        ],
        axis=-2,
    )  # (B, NH, S+1, 1)
    # for each batch/head this is a matrix of shape (S+1, S+1) containing the cumsum of the log forget gate values
    # in the second dimension (colum dimension). Each row has the same is a copy of the first row.
    # First entry of each row is zero.
    rep_log_fgates_cumsum = log_fgates_cumsum.repeat(repeats=S + 1, axis=-1)  # (B, NH, S+1, S+1)
    # Now in each row cut off / subtract the forgetgate values of the later timesteps
    # where col j > row i
    _log_fg_matrix = rep_log_fgates_cumsum - rep_log_fgates_cumsum.swapaxes(-2, -1)  # (B, NH, S+1, S+1)
    # Causal masking & selection of the correct submatrix, such that forgetgate at timestep t is not applied
    # to the input at timestep t
    log_fg_matrix = jnp.where(ltr, _log_fg_matrix[:, 1:, 1:], -float("inf"))  # (B, NH, S, S)

    # gate decay matrix D (combination of forget gate and input gate)
    log_D_matrix = log_fg_matrix + igate_preact.swapaxes(-2, -1)  # (B, NH, S, S)
    # D matrix stabilization
    if stabilize_rowwise:
        max_log_D = jnp.max(log_D_matrix, axis=-1, keepdims=True)  # (B, NH, S, 1)
    else:
        max_log_D = jnp.max(log_D_matrix.reshape(B, -1), axis=-1, keepdims=True)[..., None]
        # (B, NH, 1, 1)
    log_D_matrix_stabilized = log_D_matrix - max_log_D  # (B, NH, S, S)
    D_matrix = jnp.exp(log_D_matrix_stabilized)  # (B, NH, S, S)
    D_matrix = D_matrix.astype(qkv_dtype)

    keys_scaled = keys / math.sqrt(DHQK)

    # combination matrix C
    qk_matrix = queries @ keys_scaled.swapaxes(-2, -1)  # (B, NH, S, S)
    C_matrix = qk_matrix * D_matrix  # (B, NH, S, S)
    normalizer = jnp.maximum(
        jnp.abs(C_matrix.sum(axis=-1, keepdims=True)), jnp.exp(-max_log_D).astype(qkv_dtype)
    )  # (B, NH, S, 1)
    # (B, NH, S, S)
    C_matrix_normalized = C_matrix / (normalizer + eps)

    # retrieved values
    h_tilde_state = C_matrix_normalized @ values  # (B, NH, S, DH)

    return h_tilde_state


@dataclass
class mLSTMBackendParallelConfig:
    context_length: int = -1

    def assign_model_config_params(self, model_config):
        self.context_length = model_config.context_length


class mLSTMBackendParallel(mLSTMBackend):
    config_class = mLSTMBackendParallelConfig

    @nn.compact
    def __call__(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        i: jax.Array,
        f: jax.Array,
        c_initial: jax.Array | None = None,
        n_initial: jax.Array | None = None,
        m_initial: jax.Array | None = None,
        return_last_states: bool = False,
    ) -> jax.Array | tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
        """Forward pass of the parallel stabilized backend."""
        assert not return_last_states, "return_last_states is not supported for the parallel backend yet."
        assert (
            c_initial is None or n_initial is None or m_initial is None
        ), "Initial states are not supported for the parallel backend yet."
        causal_mask = jnp.tril(jnp.ones((self.config.context_length, self.config.context_length), dtype=jnp.bool_))
        return parallel_stabilized_simple(q, k, v, i, f, lower_triangular_matrix=causal_mask)

    @property
    def can_vmap_over_heads(self) -> bool:
        """
        Whether the backend can be vmaped over the heads dimension.

        The backend is written independent of the heads dimension, and thus can be vmapped.

        Returns:
            bool: True
        """
        return True
