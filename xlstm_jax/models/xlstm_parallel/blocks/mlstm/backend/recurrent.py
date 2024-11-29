import math
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .config import mLSTMBackend


def recurrent_step_fw(
    matC_state: jax.Array,  # (B, NH, DHQK, DHV)
    vecN_state: jax.Array,  # (B, NH, DHQK)
    scaM_state: jax.Array,  # (B, NH, 1)
    vecQ: jax.Array,  # (B, NH, DHQK)
    vecK: jax.Array,  # (B, NH, DHQK)
    vecV: jax.Array,  # (B, NH, DHV)
    scaI: jax.Array,  # (B, NH, 1)
    scaF: jax.Array,  # (B, NH, 1)
    eps: float = 1e-6,
) -> tuple[
    jax.Array, tuple[jax.Array, jax.Array, jax.Array]
]:  # vecH, (matC_state_new (B, NH, DHQK, DHV), vecN_state_new (B, NH, DHQK), vecM_state_new (B, NH, 1))
    """This is a single step of the mLSTM operation in recurrent form.

    Args:
        matC_state: Memory state tensor of shape (B, NH, DHQK, DHV).
        vecN_state: Normalizer state tensor of shape (B, NH, DHQK).
        scaM_state: Max state tensor of shape (B, NH, 1).
        vecQ: Queries tensor of shape (B, NH, DHQK).
        vecK: Keys tensor of shape (B, NH, DHQK).
        vecV: Values tensor of shape (B, NH, DHV).
        scaI: Input gate tensor of shape (B, NH, 1).
        scaF: Forget gate tensor of shape (B, NH, 1).
        eps: Used for building the forgetgate matrix. Defaults to 1e-6.

    Returns:
        The hidden state and the new states (matC_state_new, vecN_state_new, vecM_state_new).
    """
    DHQK = vecQ.shape[-1]
    state_dtype = matC_state.dtype
    qkv_dtype = vecQ.dtype

    # gates
    scaF_log = jax.nn.log_sigmoid(scaF)
    scaF_log = scaF_log.astype(state_dtype)
    scaI = scaI.astype(state_dtype)

    # update rule
    scaM_state_new = jnp.maximum(scaF_log + scaM_state, scaI)  # (B, NH, 1)

    scaF_act = jnp.exp(scaF_log + scaM_state - scaM_state_new)  # (B, NH, 1)
    scaI_act = jnp.exp(scaI - scaM_state_new)  # (B, NH, 1)

    vecQ_scaled = vecQ / math.sqrt(DHQK)  # (B, NH, DHQK)

    matC_state_new = scaF_act[:, :, :, None] * matC_state + scaI_act[:, :, :, None] * (
        vecK[:, :, :, None] @ vecV[:, :, None, :]
    )  # (B, NH, DHQK, DHV)
    vecN_state_new = scaF_act * vecN_state + scaI_act * vecK  # (B, NH, DHQK)

    h_num = vecQ_scaled[:, :, None, :] @ matC_state_new.astype(qkv_dtype)  # (B, NH, 1, DHV)
    h_num = h_num.squeeze(2)  # (B, NH, DHV)

    qn_dotproduct = vecQ_scaled[:, :, None, :] @ vecN_state_new[:, :, :, None].astype(qkv_dtype)  # (B, NH, 1, 1)
    qn_dotproduct = qn_dotproduct.squeeze(2)  # (B, NH, 1)
    max_val = jnp.exp(-scaM_state_new)  # (B, NH, 1)
    h_denom = jnp.maximum(jnp.abs(qn_dotproduct).astype(state_dtype), max_val) + eps  # (B, NH, 1)
    h = h_num.astype(state_dtype) / h_denom  # (B, NH, DHV) / (B, NH, 1) = (B, NH, DHV)
    h = h.astype(qkv_dtype)  # Division is upcasted for numerical stability.

    return h, (matC_state_new, vecN_state_new, scaM_state_new)


def recurrent_sequence_fw(
    queries: jax.Array,  # (B, NH, S, DHQK)
    keys: jax.Array,  # (B, NH, S, DHQK)
    values: jax.Array,  # (B, NH, S, DHV)
    igate_preact: jax.Array,  # (B, NH, S)
    fgate_preact: jax.Array,  # (B, NH, S)
    c_initial: jax.Array | None = None,  # (B, NH, DHQK, DHV)
    n_initial: jax.Array | None = None,  # (B, NH, DHQK)
    m_initial: jax.Array | None = None,  # (B, NH)
    return_last_states: bool = False,
    eps: float = 1e-6,
    state_dtype: jnp.dtype | None = None,
    use_scan: bool = False,
    mlstm_step_fn: Callable = recurrent_step_fw,
) -> jax.Array | tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
    """
    Forward pass of the mLSTM cell in recurrent form on a full sequence.

    Args:
        queries: Queries tensor of shape (B, NH, S, DHQK).
        keys: Keys tensor of shape (B, NH, S, DHQK).
        values: Values tensor of shape (B, NH, S, DHV).
        igate_preact: Input gate pre-activation tensor of shape (B, NH, S, 1).
        fgate_preact: Forget gate pre-activation tensor of shape (B, NH, S, 1).
        c_initial: Initial memory state tensor of shape (B, NH, DHQK, DHV). If None, initialized to zeros.
        n_initial: Initial normalizer state tensor of shape (B, NH, DHQK). If None, initialized to zeros.
        m_initial: Initial max state tensor of shape (B, NH). If None, initialized to zeros.
        return_last_states: Whether to return the last states. Defaults to False.
        eps: Epsilon value for numerical stability. Defaults to 1e-6.
        state_dtype: Dtype of the states. If None, uses the dtype of the initial states if provided, or other
            the dtypes of the pre-activations. If initial states are provided, the return dtype will be the same as the
            initial states. Defaults to None.
        use_scan: Whether to use `jax.lax.scan` for the loop. The scan reduces compilation time, but may be slower for
            kernels without XLA compiler support and introduces memory copy overhead.
        mlstm_step_fn: Function to compute a single mLSTM step. By default, set to `recurrent_step_fw` in this backend.

    Returns:
        Hidden states tensor of shape (B, NH, S, DHV) if `return_last_states` is False.
        Tuple of hidden states tensor and tuple of last states tensors if `return_last_states` is True.
    """
    B, NH, S, DHQK = queries.shape
    DHV = values.shape[-1]
    if igate_preact.ndim == 3:
        igate_preact = igate_preact[:, :, :, None]
    if fgate_preact.ndim == 3:
        fgate_preact = fgate_preact[:, :, :, None]

    # Set up initial states.
    if c_initial is not None:
        assert n_initial is not None and m_initial is not None, "Initial states must be provided together."
        assert m_initial.ndim == 2, "Initial states must be 2D."
        matC_state, vecN_state, vecM_state = (
            c_initial,
            n_initial,
            m_initial[:, :, None],
        )
        if state_dtype is not None:
            matC_state = matC_state.astype(state_dtype)
            vecN_state = vecN_state.astype(state_dtype)
            vecM_state = vecM_state.astype(state_dtype)
    else:
        if state_dtype is None:
            state_dtype = fgate_preact.dtype
        # memory state
        matC_state = jnp.zeros((B, NH, DHQK, DHV), dtype=state_dtype)
        # normalizer state
        vecN_state = jnp.zeros((B, NH, DHQK), dtype=state_dtype)
        # max state
        vecM_state = jnp.zeros((B, NH, 1), dtype=state_dtype)

    if S == 1:
        # Single step can skip the loop and other operations.
        matH, (matC_state, vecN_state, vecM_state) = mlstm_step_fn(
            matC_state=matC_state,
            vecN_state=vecN_state,
            scaM_state=vecM_state,
            vecQ=queries[:, :, 0],
            vecK=keys[:, :, 0],
            vecV=values[:, :, 0],
            scaI=igate_preact[:, :, 0],
            scaF=fgate_preact[:, :, 0],
            eps=eps,
        )
        matH = matH[:, :, None]
    elif use_scan:
        # Recurrent loop step function. The states are in the carry,
        # and the inputs are the tensors at the current time step.
        def _scan_fn(carry, inputs):
            matC_state, vecN_state, vecM_state = carry
            vecQ_t, vecK_t, vecV_t, vecI_t, vecF_t = inputs

            vecH, (matC_state, vecN_state, vecM_state) = mlstm_step_fn(
                matC_state=matC_state,
                vecN_state=vecN_state,
                scaM_state=vecM_state,
                vecQ=vecQ_t,
                vecK=vecK_t,
                vecV=vecV_t,
                scaI=vecI_t,
                scaF=vecF_t,
                eps=eps,
            )
            return (matC_state, vecN_state, vecM_state), vecH

        # Run the recurrent loop.
        carry = (matC_state, vecN_state, vecM_state)
        inputs = (queries, keys, values, igate_preact, fgate_preact)
        inputs = jax.tree.map(lambda x: jnp.moveaxis(x, 2, 0), inputs)  # Scan expects time as the first dimension.
        (matC_state, vecN_state, vecM_state), matH = jax.lax.scan(_scan_fn, carry, inputs)
        matH = jnp.moveaxis(matH, 0, 2)  # Scan returns time as the first dimension.
    else:
        # Use loop to iterate over the sequence.
        # Recurrent loop step function. The states are in the carry,
        # and the inputs are the tensors at the current time step.
        inputs = {
            "vecQ": queries,
            "vecK": keys,
            "vecV": values,
            "scaI": igate_preact,
            "scaF": fgate_preact,
        }
        inputs = jax.tree.map(
            lambda x: jnp.moveaxis(x, 2, 0), inputs
        )  # Move time to first dimension for easier slicing.
        outputs = []
        for t in range(S):
            inputs_t = jax.tree_map(lambda x: x[t], inputs)
            vecH_t, (matC_state, vecN_state, vecM_state) = mlstm_step_fn(
                matC_state=matC_state,
                vecN_state=vecN_state,
                scaM_state=vecM_state,
                **inputs_t,
                eps=eps,
            )
            outputs.append(vecH_t)

        matH = jnp.stack(outputs, axis=2)

    if return_last_states:
        if c_initial is not None:
            matC_state = matC_state.astype(c_initial.dtype)
        if n_initial is not None:
            vecN_state = vecN_state.astype(n_initial.dtype)
        if m_initial is not None:
            vecM_state = vecM_state.astype(m_initial.dtype)
        return matH, (matC_state, vecN_state, vecM_state.squeeze(-1))
    return matH


@dataclass
class mLSTMBackendRecurrentConfig:
    context_length: int = -1
    eps: float = 1e-6
    state_dtype: str | None = None
    use_scan: bool = False

    def assign_model_config_params(self, model_config):
        self.context_length = model_config.context_length


class mLSTMBackendRecurrent(mLSTMBackend):
    config_class = mLSTMBackendRecurrentConfig

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
        """Forward pass of the recurrent backend."""
        return recurrent_sequence_fw(
            mlstm_step_fn=recurrent_step_fw,
            queries=q,
            keys=k,
            values=v,
            igate_preact=i,
            fgate_preact=f,
            c_initial=c_initial,
            n_initial=n_initial,
            m_initial=m_initial,
            return_last_states=return_last_states,
            eps=self.config.eps,
            state_dtype=self.config.state_dtype,
            use_scan=self.config.use_scan,
        )

    @property
    def can_vmap_over_heads(self) -> bool:
        """
        Whether the backend can be vmaped over the heads dimension.

        The backend was not written to be independent of the heads dimension, and thus cannot be vmapped.

        Returns:
            bool: False
        """
        return False
