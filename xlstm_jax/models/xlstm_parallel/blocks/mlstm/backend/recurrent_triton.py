from dataclasses import dataclass

import jax

from xlstm_jax.kernels import mlstm_recurrent_step_triton_fused

from .config import mLSTMBackend
from .recurrent import recurrent_sequence_fw


@dataclass
class mLSTMBackendRecurrentTritonConfig:
    """
    Configuration class for the mLSTM recurrent backend using Triton kernels.
    """

    eps: float = 1e-6
    """Epsilon value used in the kernel."""
    state_dtype: str | None = None
    """Data type for the state tensors. If None, the data type is inferred from the input tensors."""
    use_scan: bool = False
    """Whether to use scan for the recurrent sequence."""

    def assign_model_config_params(self, *args, **kwargs):
        pass


class mLSTMBackendRecurrentTriton(mLSTMBackend):
    """
    mLSTM recurrent backend using Triton kernels.

    This backend uses Triton kernels for the mLSTM recurrent cell.

    Args:
        config: Configuration object for the backend.
        config_class: Configuration class for the backend.
    """

    config_class = mLSTMBackendRecurrentTritonConfig

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
    ) -> jax.Array:
        """
        Forward pass of the mLSTM cell using triton kernels.

        Args:
            q: Query tensor of shape (B, NH, S, DHQK).
            k: Key tensor of shape (B, NH, S, DHQK).
            v: Value tensor of shape (B, NH, S, DHV).
            i: Input gate tensor of shape (B, NH, S, 1) or (B, NH, S).
            f: Forget gate tensor of shape (B, NH, S, 1) or (B, NH, S).
            c_initial: Initial cell state tensor of shape (B, NH, DHQK, DHV).
            n_initial: Initial norm state tensor of shape (B, NH, DHQK).
            m_initial: Initial maximizer state tensor of shape (B, NH).
            return_last_states: Whether to return the last states.

        Returns:
            Output tensor of shape (B, NH, S, DH).
        """
        return recurrent_sequence_fw(
            mlstm_step_fn=mlstm_recurrent_step_triton_fused,
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

        Triton kernels already handle the head dimension, hence not to be vmaped over.

        Returns:
            bool: False
        """
        return False
