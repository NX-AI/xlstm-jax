from dataclasses import dataclass, field
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
from flax import linen as nn

from xlstm_jax.models.configs import ParallelConfig, SubModelConfig
from xlstm_jax.models.shared import soft_cap_logits

from ...components.init import bias_linspace_init
from ...components.linear_headwise import LinearHeadwiseExpand, LinearHeadwiseExpandConfig
from ...components.normalization import MultiHeadNormLayer, NormLayer
from .backend import mLSTMBackendNameAndKwargs
from .backend_utils import run_backend


@dataclass
class mLSTMCellConfig(SubModelConfig):
    context_length: int = -1
    embedding_dim: int = -1
    num_heads: int = -1
    backend: mLSTMBackendNameAndKwargs = field(
        default_factory=lambda: mLSTMBackendNameAndKwargs(name="parallel_stabilized")
    )
    norm_eps: float = 1e-6
    """Epsilon value for numerical stability in layer norm."""
    norm_type: Literal["layernorm", "rmsnorm"] = "layernorm"
    """Type of normalization layer to use."""
    norm_type_v1: Literal["layernorm", "rmsnorm"] = "layernorm"
    """Type of normalization layer to use. NOTE: this is only used in the 'mlstm_v1' layer_type. Due to a bug, the
    'norm_type' was not used correctly in the v1 version. To keep the same behavior, we introduce a separate parameter
    for the normalization layer."""
    dtype: str = "bfloat16"
    gate_dtype: str = "float32"
    gate_soft_cap: float | None = None
    """Soft cap for the gate pre-activations. If None, no cap is applied."""
    gate_linear_headwise: bool = False
    """If True, the gate pre-activations are computed with a linear headwise layer, similar to QKV.
    Otherwise, each gate head takes as input the full features across all heads."""
    igate_bias_init_range: tuple[float, float] | float | None = None
    """Input gate bias initialization. If a tuple, the bias is initialized with a linspace in the given range.
    If a float, the bias is initialized with the given value. If None, the bias is initialized with normal(0.1)."""
    fgate_bias_init_range: tuple[float, float] | float | None = (3.0, 6.0)
    """Forget gate bias initialization. If a tuple, the bias is initialized with a linspace in the given range.
    If a float, the bias is initialized with the given value. If None, the bias is initialized with normal(0.1)."""
    add_qk_norm: bool = False
    """If True, adds a normalization layer on the query and key vectors before the mLSTM cell."""
    reset_at_document_boundaries: bool = False
    """If True, the memory is reset at the beginning of each document."""
    reset_fgate_value: float = -25.0
    """Value to set the forget gate to at document boundaries."""
    parallel: ParallelConfig | None = None
    """Parallel configuration for the mLSTM cell."""

    @property
    def _dtype(self) -> jnp.dtype:
        """
        Returns the real dtype instead of the str from configs.

        Returns:
            The jnp dtype corresponding to the string value.
        """
        return getattr(jnp, self.dtype)

    @property
    def _gate_dtype(self) -> jnp.dtype:
        return getattr(jnp, self.gate_dtype)


class mLSTMCell(nn.Module):
    config: mLSTMCellConfig

    @nn.compact
    def __call__(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        gate_input: jax.Array | tuple[jax.Array, ...] | None = None,
        document_borders: jax.Array | None = None,
        **kwargs,
    ):
        """mLSTM cell implementation.

        Args:
            q: Query tensor. Should be of shape (batch_size, context_length, num_heads * key_dim).
            k: Key tensor. Should be of shape (batch_size, context_length, num_heads * key_dim).
            v: Value tensor. Should be of shape (batch_size, context_length, num_heads * value_dim).
            gate_input: Input to the gate layers that predict the gate pre-activations. If None, the input
                is (q, k, v). If a tuple, the inputs are concatenated along the last axis. For linear
                headwise layers, the inputs are reshaped to (batch_size, context_length, num_heads, -1) and
                concatenated along the last axis.
            document_borders: Boolean mask tensor indicating the document boundaries. If provided, the forget
                gate is set to 0 at the document boundaries, i.e. the memory is reset at the beginning of each
                document.
            kwargs: Additional arguments for the mLSTM backend.
        """
        B, S, _ = q.shape
        tp_size = jax.lax.psum(1, self.config.parallel.model_axis_name)
        assert tp_size == 1 or self.config.gate_linear_headwise, "Only headwise gate layers are supported with TP > 1."

        # Prepare gate input.
        if gate_input is None:
            gate_input = (q, k, v)
        if isinstance(gate_input, tuple):
            if self.config.gate_linear_headwise:
                # If headwise, we need to restructure the features such that the headwise features of each input are
                # concatenated on the last axis.
                gate_input = [inp.reshape(*inp.shape[:-1], self.config.num_heads, -1) for inp in gate_input]
                gate_input = jnp.concatenate(gate_input, axis=-1)
                gate_input = gate_input.reshape(*gate_input.shape[:-2], -1)
            else:
                # For a dense layer, the order of the inputs does not matter.
                gate_input = jnp.concatenate(gate_input, axis=-1)

        # Compute input and forget gate pre-activations.
        with jax.named_scope("mlstm_gates"):
            # Layer is either a linear headwise layer (inputs per-head features) or a dense layer (inputs all features).
            def gate_layer(bias_init, name):
                if self.config.gate_linear_headwise:
                    return LinearHeadwiseExpand(
                        config=LinearHeadwiseExpandConfig(
                            in_features=gate_input.shape[-1],
                            num_heads=self.config.num_heads,
                            _out_features=self.config.num_heads,
                            bias=True,
                            dtype=self.config.gate_dtype,
                        ),
                        bias_init=bias_init,
                        kernel_init=nn.initializers.zeros,
                        name=name,
                    )
                else:
                    return nn.Dense(
                        features=self.config.num_heads,
                        dtype=self.config._gate_dtype,
                        bias_init=bias_init,
                        kernel_init=nn.initializers.zeros,
                        name=name,
                    )

            # Initialization function for the gate biases.
            def gate_init(init_range):
                if init_range is None:
                    # Previous default behavior for input gate.
                    init_fn = nn.initializers.normal(stddev=0.1)
                elif isinstance(init_range, tuple):
                    init_fn = bias_linspace_init(*init_range, axis_name=self.config.parallel.model_axis_name)
                else:
                    init_fn = nn.initializers.constant(init_range)
                return init_fn

            # Compute the gate pre-activations.
            igate_preact = gate_layer(
                bias_init=gate_init(self.config.igate_bias_init_range),
                name="igate",
            )(gate_input)
            fgate_preact = gate_layer(
                bias_init=gate_init(self.config.fgate_bias_init_range),
                name="fgate",
            )(gate_input)

            # Apply soft cap to the gate pre-activations.
            igate_preact = soft_cap_logits(igate_preact, self.config.gate_soft_cap)
            fgate_preact = soft_cap_logits(fgate_preact, self.config.gate_soft_cap)

            # Reset memory at document boundaries.
            if document_borders is not None and self.config.reset_at_document_boundaries:
                # Set forget gate to 0 at document boundaries.
                # NOTE: If a convolution was applied to QKV, the state may not be fully independent of the
                # previous documents. This is a limitation of the v2 mLSTM implementation.
                fgate_preact = jnp.where(document_borders[..., None], self.config.reset_fgate_value, fgate_preact)

            # Log intermediate statistics of the gates.
            self.sow("intermediates", "max_igate_preact", jnp.max(igate_preact))
            self.sow("intermediates", "max_fgate_preact", jnp.max(fgate_preact))
            self.sow("intermediates", "min_igate_preact", jnp.min(igate_preact))
            self.sow("intermediates", "min_fgate_preact", jnp.min(fgate_preact))
            self.sow("intermediates", "mean_igate_preact", jnp.mean(igate_preact))
            self.sow("intermediates", "mean_fgate_preact", jnp.mean(fgate_preact))
            self.sow("intermediates", "igate_bias", self.get_variable("params", "igate")["bias"].mean())
            self.sow("intermediates", "fgate_bias", self.get_variable("params", "fgate")["bias"].mean())

        q = q.reshape(B, S, self.config.num_heads, -1)  # (B, S, NH, DHQK)
        k = k.reshape(B, S, self.config.num_heads, -1)  # (B, S, NH, DHQK)
        v = v.reshape(B, S, self.config.num_heads, -1)  # (B, S, NH, DHV)
        igate_preact = igate_preact[..., None]  # (B, S, NH, 1)
        fgate_preact = fgate_preact[..., None]  # (B, S, NH, 1)

        # Normalize q and k.
        if self.config.add_qk_norm:
            norm_fn = partial(
                NormLayer,
                weight=True,
                bias=False,
                eps=self.config.norm_eps,
                dtype=self.config._dtype,
                norm_type=self.config.norm_type,
            )
            q = norm_fn(name="q_norm")(q)
            k = norm_fn(name="k_norm")(k)

        h_state = run_backend(
            parent=self,
            cell_config=self.config,
            q=q,
            k=k,
            v=v,
            igate_preact=igate_preact,
            fgate_preact=fgate_preact,
        )

        h_state_norm = MultiHeadNormLayer(
            weight=True,
            bias=False,
            dtype=self.config._dtype,
            name="outnorm",
            axis=2,
            eps=self.config.norm_eps,
            norm_type=self.config.norm_type,
        )(h_state)
        h_state_norm = h_state_norm.reshape(B, S, -1)
        return h_state_norm
