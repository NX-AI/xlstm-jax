from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn

from .backend import create_mlstm_backend, mLSTMBackend


def run_backend(
    parent: nn.Module,
    cell_config: Any,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    igate_preact: jax.Array,
    fgate_preact: jax.Array,
) -> jax.Array:
    """
    Execute the mLSTM backend for the given input tensors.

    This function handles the caching of intermediate states, if enabled, and the vmap over the heads dimension.
    The caching follows the setup of the cache in the Attention module:
    https://github.com/google/flax/blob/main/flax/linen/attention.py#L570. During decoding, if the cache is not
    initialized, the cache is initialized with zeros and we *do not* update the cache, following the setup in the
    Attention module. If the cache is provided, we update the cache with the new states.

    Args:
        parent: The parent module.
        cell_config: The mLSTM cell configuration.
        q: The query tensor, shape (batch_size, seq_len, num_heads, qk_dim).
        k: The key tensor, shape (batch_size, seq_len, num_heads, qk_dim).
        v: The value tensor, shape (batch_size, seq_len, num_heads, v_dim).
        igate_preact: The input gate preactivation, shape (batch_size, seq_len, num_heads, 1).
        fgate_preact: The forget gate preactivation, shape (batch_size, seq_len, num_heads, 1).

    Returns:
        The output tensor, shape (batch_size, seq_len, num_heads, v_dim).
    """
    batch_size, _, num_heads, qk_dim = q.shape
    _, _, _, v_dim = v.shape
    backend_fn: mLSTMBackend = create_mlstm_backend(cell_config)

    # Check whether to use the cache.
    use_cache = parent.is_mutable_collection("cache") and not parent.is_initializing()
    if use_cache:
        cache_initialized = parent.has_variable("cache", "cached_c")
        # Infer state dtype from backend config, or use the dtype of the forget gate preactivation.
        if hasattr(cell_config.backend.kwargs, "state_dtype"):
            state_dtype = cell_config.backend.kwargs.state_dtype
        else:
            state_dtype = fgate_preact.dtype
        c_initial = parent.variable("cache", "cached_c", jnp.zeros, (batch_size, num_heads, qk_dim, v_dim), state_dtype)
        n_initial = parent.variable("cache", "cached_n", jnp.zeros, (batch_size, num_heads, qk_dim), state_dtype)
        m_initial = parent.variable("cache", "cached_m", jnp.zeros, (batch_size, num_heads), state_dtype)
    else:
        cache_initialized = False
        c_initial, n_initial, m_initial = None, None, None

    if backend_fn.can_vmap_over_heads:
        # Vmap over the heads dimension without needing to transpose the input tensors.
        backend_fn = partial(backend_fn, return_last_states=use_cache)
        if use_cache:
            backend_fn = jax.vmap(backend_fn, in_axes=(2, 2, 2, 2, 2, 1, 1, 1), out_axes=(2, 1, 1, 1))
            with jax.named_scope("mlstm_backend"):
                h_state, (c_updated, n_updated, m_updated) = backend_fn(
                    q,
                    k,
                    v,
                    igate_preact,
                    fgate_preact,
                    c_initial.value,
                    n_initial.value,
                    m_initial.value,
                )
        else:
            backend_fn = partial(backend_fn, return_last_states=False)
            backend_fn = jax.vmap(backend_fn, in_axes=(2, 2, 2, 2, 2), out_axes=2)
            with jax.named_scope("mlstm_backend"):
                h_state = backend_fn(q, k, v, igate_preact, fgate_preact)
            c_updated, n_updated, m_updated = None, None, None
    else:
        # Manual transpose to work over heads.
        q = q.transpose(0, 2, 1, 3)  # (B, NH, S, DHQK)
        k = k.transpose(0, 2, 1, 3)  # (B, NH, S, DHQK)
        v = v.transpose(0, 2, 1, 3)  # (B, NH, S, DHV)
        igate_preact = igate_preact.transpose(0, 2, 1, 3)  # (B, NH, S, 1)
        fgate_preact = fgate_preact.transpose(0, 2, 1, 3)  # (B, NH, S, 1)
        args = (q, k, v, igate_preact, fgate_preact)
        if use_cache:
            args += (c_initial.value, n_initial.value, m_initial.value)
        with jax.named_scope("mlstm_backend"):
            out = backend_fn(*args, return_last_states=use_cache)
        if use_cache:
            h_state, (c_updated, n_updated, m_updated) = out
        else:
            h_state = out
        h_state = h_state.transpose(0, 2, 1, 3)  # (B, S, NH, DHV)

    if use_cache and cache_initialized:
        # Update the cache.
        c_initial.value = c_updated
        n_initial.value = n_updated
        m_initial.value = m_updated

    return h_state
