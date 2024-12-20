#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from functools import partial

import jax
import jax.numpy as jnp
from flax import linen as nn

from xlstm_jax.distributed.tensor_parallel import TPAsyncDense, TPDense
from xlstm_jax.models.shared import create_common_init_fn, small_init, soft_cap_logits

from ...components.init import bias_linspace_init
from ...components.normalization import MultiHeadNormLayer, NormLayer
from .backend_utils import run_backend
from .layer import mLSTMLayerConfig


class mLSTMLayerV1(nn.Module):
    """
    mLSTM layer with a Transformer-style projection up and down.
    """

    config: mLSTMLayerConfig

    @nn.compact
    def __call__(
        self, x: jax.Array, document_borders: jax.Array | None = None, train: bool = True, **kwargs
    ) -> jax.Array:
        """
        Forward pass of the mLSTM layer.

        Args:
            x: input tensor of shape (batch_size, context_length, embedding_dim).
            document_borders: Optional boolean tensor indicating which input tokens represent document borders (True)
                and which don't (False). For document border tokens, the mLSTM memory will be reset if selected in
                config (see mlstm_cell). Shape (batch_size, context_length).
            train: Whether the model is in training mode. If True, may apply dropout.

        Returns:
            The output tensor of the mLSTM layer, shape (batch_size, context_length, embedding_dim).
        """
        B, S, _ = x.shape
        tp_size = jax.lax.psum(1, self.config.parallel.model_axis_name)
        assert self.config.num_heads % tp_size == 0, "num_heads must be divisible by the number of model replicas"
        num_local_heads = self.config.num_heads // tp_size
        embedding_dim = x.shape[-1] * tp_size

        tp_dense_fn = (
            partial(TPAsyncDense, use_bidirectional_gather=True, use_bidirectional_scatter=True)
            if self.config.parallel.tp_async_dense
            else TPDense
        )
        # up-projection
        if self.config.parallel.tp_async_dense:
            raise NotImplementedError("Async dense not supported for mLSTMLayerSimple")
        # split projection up into two smaller matrices, as splitting large feature vector costs time.
        x = jax.lax.all_gather(x, self.config.parallel.model_axis_name, axis=-1, tiled=True)

        # Projection up layers
        def proj_up_layer(name, kernel_init, scale: float = 1.0):
            return TPDense(
                dense_fn=partial(
                    nn.Dense,
                    dtype=self.config._dtype,
                    features=int(embedding_dim * scale) // tp_size,
                ),
                model_axis_name=self.config.parallel.model_axis_name,
                tp_mode="gather",
                kernel_init=kernel_init,
                use_bias=self.config.bias,
                skip_communication=True,
                name=name,
            )

        # QKV layers
        qkv_init = small_init(embedding_dim, distribution=self.config.init_distribution)
        q = proj_up_layer(name="dense_q", kernel_init=qkv_init, scale=self.config.qk_dim_factor)(x)
        k = proj_up_layer(name="dense_k", kernel_init=qkv_init, scale=self.config.qk_dim_factor)(x)
        v = proj_up_layer(name="dense_v", kernel_init=qkv_init, scale=self.config.v_dim_factor)(x)

        # Output layer
        o = proj_up_layer(name="dense_o", kernel_init=nn.initializers.zeros, scale=self.config.v_dim_factor)(x)

        # Gates
        def gate_layer(name, bias_init):
            return TPDense(
                dense_fn=partial(
                    nn.Dense,
                    dtype=self.config.mlstm_cell.gate_dtype,
                    features=num_local_heads,
                    bias_init=bias_init,
                ),
                model_axis_name=self.config.parallel.model_axis_name,
                tp_mode="gather",
                kernel_init=nn.initializers.zeros,
                use_bias=True,
                skip_communication=True,
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
            bias_init=gate_init(self.config.mlstm_cell.igate_bias_init_range),
            name="igate",
        )(x)
        fgate_preact = gate_layer(
            bias_init=gate_init(self.config.mlstm_cell.fgate_bias_init_range),
            name="fgate",
        )(x)

        # Apply soft cap to the gate pre-activations.
        igate_preact = soft_cap_logits(igate_preact, self.config.mlstm_cell.gate_soft_cap)
        fgate_preact = soft_cap_logits(fgate_preact, self.config.mlstm_cell.gate_soft_cap)

        # Reset memory at document boundaries.
        if document_borders is not None and self.config.mlstm_cell.reset_at_document_boundaries:
            # Set forget gate to 0 at document boundaries.
            fgate_preact = jnp.where(
                document_borders[..., None], self.config.mlstm_cell.reset_fgate_value, fgate_preact
            )

        # mlstm branch
        q = q.reshape(B, S, num_local_heads, -1)  # (B, S, NH, DHQK)
        k = k.reshape(B, S, num_local_heads, -1)  # (B, S, NH, DHQK)
        v = v.reshape(B, S, num_local_heads, -1)  # (B, S, NH, DHV)
        igate_preact = igate_preact[..., None]  # (B, S, NH, 1)
        fgate_preact = fgate_preact[..., None]  # (B, S, NH, 1)

        # Normalize q and k.
        if self.config.mlstm_cell.add_qk_norm:
            norm_fn = partial(
                NormLayer,
                weight=True,
                bias=False,
                eps=self.config.mlstm_cell.norm_eps,
                dtype=self.config._dtype,
                norm_type=self.config.norm_type,
            )
            q = norm_fn(name="q_norm")(q)
            k = norm_fn(name="k_norm")(k)

        h_state = run_backend(
            parent=self,
            cell_config=self.config.mlstm_cell,
            q=q,
            k=k,
            v=v,
            igate_preact=igate_preact,
            fgate_preact=fgate_preact,
        )

        # Normalize output of mLSTM cell
        h_state_norm = MultiHeadNormLayer(
            weight=True,
            bias=False,
            dtype=self.config._dtype,
            name="outnorm",
            axis=2,
            eps=self.config.mlstm_cell.norm_eps,
            norm_type=self.config.mlstm_cell.norm_type_v1,
            model_axis_name=self.config.parallel.model_axis_name,
        )(h_state)
        h_state_norm = h_state_norm.reshape(B, S, -1)

        # output / z branch. Fixed to sigmoid for now as experiments show it works best.
        h_state_out = nn.sigmoid(o) * h_state_norm

        # down-projection
        y = tp_dense_fn(
            dense_fn=partial(
                nn.Dense,
                dtype=self.config._dtype,
                features=self.config.embedding_dim // tp_size
                if self.config.parallel.tp_async_dense
                else self.config.embedding_dim,
            ),
            model_axis_name=self.config.parallel.model_axis_name,
            tp_mode="scatter",
            kernel_init=create_common_init_fn(
                fn_name=self.config.output_init_fn,
                dim=embedding_dim,
                num_blocks=self.config._num_blocks,
                distribution=self.config.init_distribution,
            ),
            use_bias=self.config.bias,
            name="proj_down",
        )(h_state_out)
        y = nn.Dropout(rate=self.config.dropout, deterministic=not train)(y)

        # Log intermediate statistics of the mLSTM layer.
        self.sow("intermediates", "q_std", jnp.std(q, axis=-1).mean())
        self.sow("intermediates", "q_abs_max", jnp.abs(q).max(axis=-1).mean())
        self.sow("intermediates", "q_mean", q.mean())
        self.sow("intermediates", "k_std", jnp.std(k, axis=-1).mean())
        self.sow("intermediates", "k_abs_max", jnp.abs(k).max(axis=-1).mean())
        self.sow("intermediates", "k_mean", k.mean())
        self.sow("intermediates", "v_std", jnp.std(v, axis=-1).mean())
        self.sow("intermediates", "v_abs_max", jnp.abs(v).max(axis=-1).mean())
        self.sow("intermediates", "v_mean", v.mean())
        self.sow("intermediates", "max_igate_preact", jnp.max(igate_preact))
        self.sow("intermediates", "max_fgate_preact", jnp.max(fgate_preact))
        self.sow("intermediates", "min_igate_preact", jnp.min(igate_preact))
        self.sow("intermediates", "min_fgate_preact", jnp.min(fgate_preact))
        self.sow("intermediates", "mean_igate_preact", jnp.mean(igate_preact))
        self.sow("intermediates", "mean_fgate_preact", jnp.mean(fgate_preact))
        self.sow("intermediates", "mean_output_gate", nn.sigmoid(o).mean())
        self.sow("intermediates", "h_state_std", jnp.std(h_state, axis=-1).mean())
        self.sow("intermediates", "h_state_abs_max", jnp.abs(h_state).max())
        self.sow("intermediates", "h_state_norm_std", jnp.std(h_state_norm, axis=-1).mean())
        self.sow("intermediates", "h_state_norm_abs_max", jnp.abs(h_state_norm).max())
        self.sow("intermediates", "h_state_out_std", jnp.std(h_state_out, axis=-1).mean())
        self.sow("intermediates", "h_state_out_abs_max", jnp.abs(h_state_out).max())
        self.sow("intermediates", "block_output_std", jnp.std(y, axis=-1).mean())
        self.sow("intermediates", "block_output_abs_max", jnp.abs(y).max())

        return y
