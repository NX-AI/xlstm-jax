from functools import partial

import jax
import jax.numpy as jnp
from flax import linen as nn

from xlstm_jax.distributed.tensor_parallel import TPAsyncDense, TPDense

from ...components.init import bias_linspace_init, small_init, wang_init
from ...components.normalization import MultiHeadNormLayer
from ...utils import soft_cap_logits
from .backend import create_mlstm_backend, mLSTMBackend
from .layer import mLSTMLayerConfig


class mLSTMLayerV1(nn.Module):
    config: mLSTMLayerConfig

    @nn.compact
    def __call__(self, x: jax.Array, train: bool = True, **kwargs) -> jax.Array:
        B, S, _ = x.shape
        tp_size = jax.lax.psum(1, self.config.parallel.model_axis_name)
        assert self.config.num_heads % tp_size == 0, "num_heads must be divisible by the number of model replicas"
        embedding_dim = x.shape[-1] * tp_size

        tp_dense_fn = (
            partial(TPAsyncDense, use_bidirectional_gather=True, use_bidirectional_scatter=True)
            if self.config.parallel.tp_async_dense
            else TPDense
        )
        # up-projection
        if self.config.parallel.tp_async_dense:
            raise NotImplementedError("Async dense not supported for mLSTMLayerSimple")
        else:
            # split projection up into two smaller matrices, as splitting large feature vector costs time.
            x = jax.lax.all_gather(x, self.config.parallel.model_axis_name, axis=-1, tiled=True)

            # Projection up layers
            def proj_up_layer(name, kernel_init):
                return TPDense(
                    dense_fn=partial(
                        nn.Dense,
                        dtype=self.config.dtype,
                        use_bias=self.config.bias,
                        features=embedding_dim // tp_size,
                    ),
                    model_axis_name=self.config.parallel.model_axis_name,
                    tp_mode="gather",
                    kernel_init=kernel_init,
                    skip_communication=True,
                    name=name,
                )

            # QKV layers
            q = proj_up_layer(name="dense_q", kernel_init=small_init(embedding_dim))(x)
            k = proj_up_layer(name="dense_k", kernel_init=small_init(embedding_dim))(x)
            v = proj_up_layer(name="dense_v", kernel_init=small_init(embedding_dim))(x)

            # Output layer
            o = proj_up_layer(name="dense_o", kernel_init=nn.initializers.zeros)(x)

            # Gates
            def gate_layer(name, bias_init):
                return TPDense(
                    dense_fn=partial(
                        nn.Dense,
                        dtype=self.config.mlstm_cell.gate_dtype,
                        use_bias=True,
                        features=self.config.num_heads // tp_size,
                        bias_init=bias_init,
                    ),
                    model_axis_name=self.config.parallel.model_axis_name,
                    tp_mode="gather",
                    kernel_init=nn.initializers.zeros,
                    skip_communication=True,
                    name=name,
                )

            # Initialization function for the gate biases.
            def gate_init(init_range):
                if init_range is None:
                    # Previous default behavior for input gate.
                    init_fn = nn.initializers.normal(stddev=0.1)
                elif isinstance(init_range, tuple):
                    init_fn = bias_linspace_init(*init_range)
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

        # mlstm branch
        q = q.reshape(B, S, self.config.num_heads, -1)  # (B, S, NH, DH)
        k = k.reshape(B, S, self.config.num_heads, -1)  # (B, S, NH, DH)
        v = v.reshape(B, S, self.config.num_heads, -1)  # (B, S, NH, DH)
        igate_preact = igate_preact[..., None]  # (B, S, NH, 1)
        fgate_preact = fgate_preact[..., None]  # (B, S, NH, 1)

        backend_fn: mLSTMBackend = create_mlstm_backend(self.config.mlstm_cell)

        if backend_fn.can_vmap_over_heads:
            # Vmap over the heads dimension without needing to transpose the input tensors.
            backend_fn = jax.vmap(backend_fn, in_axes=(2, 2, 2, 2, 2), out_axes=2)
            with jax.named_scope("mlstm_backend"):
                h_state = backend_fn(q, k, v, igate_preact, fgate_preact)
        else:
            # Manual transpose to work over heads.
            q = q.transpose(0, 2, 1, 3)  # (B, NH, S, DH)
            k = k.transpose(0, 2, 1, 3)  # (B, NH, S, DH)
            v = v.transpose(0, 2, 1, 3)  # (B, NH, S, DH)
            igate_preact = igate_preact.transpose(0, 2, 1, 3)  # (B, NH, S, 1)
            fgate_preact = fgate_preact.transpose(0, 2, 1, 3)  # (B, NH, S, 1)
            with jax.named_scope("mlstm_backend"):
                h_state = backend_fn(q, k, v, igate_preact, fgate_preact)
            h_state = h_state.transpose(0, 2, 1, 3)

        # Normalize output of mLSTM cell
        h_state_norm = MultiHeadNormLayer(
            weight=True,
            bias=False,
            dtype=self.config.dtype,
            name="outnorm",
            axis=2,
            eps=self.config.mlstm_cell.norm_eps,
        )(h_state)
        h_state_norm = h_state_norm.reshape(B, S, -1)

        # output / z branch. Fixed to sigmoid for now as experiments show it works best.
        h_state_out = nn.sigmoid(o) * h_state_norm

        # down-projection
        y = tp_dense_fn(
            dense_fn=partial(
                nn.Dense,
                dtype=self.config.dtype,
                use_bias=self.config.bias,
                features=self.config.embedding_dim // tp_size
                if self.config.parallel.tp_async_dense
                else self.config.embedding_dim,
            ),
            model_axis_name=self.config.parallel.model_axis_name,
            tp_mode="scatter",
            kernel_init=wang_init(embedding_dim, num_blocks=self.config._num_blocks),
            name="proj_down",
        )(h_state_out)
        y = nn.Dropout(rate=self.config.dropout, deterministic=not train)(y)

        # Log intermediate statistics of the mLSTM layer.
        self.sow("intermediates", "q_std", jnp.std(q, axis=-1).mean())
        self.sow("intermediates", "k_std", jnp.std(k, axis=-1).mean())
        self.sow("intermediates", "v_std", jnp.std(v, axis=-1).mean())
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
