from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
from flax import linen as nn

from xlstm_jax.distributed.tensor_parallel import TPAsyncDense, TPDense
from xlstm_jax.models.configs import ParallelConfig
from xlstm_jax.models.shared import InitDistribution, InitFnName, create_common_init_fn, small_init

from ..utils import UpProjConfigMixin

_act_fn_registry = {
    "gelu": nn.gelu,
    "relu": nn.relu,
    "relu^2": lambda x: jnp.square(nn.relu(x)),
    "sigmoid": nn.sigmoid,
    "swish": nn.silu,
    "selu": nn.selu,
}


def get_act_fn(act_fn_name: str) -> Callable[[jax.Array], jax.Array]:
    if act_fn_name in _act_fn_registry:
        return _act_fn_registry[act_fn_name]
    else:
        assert False, (
            f"Unknown activation function name '{act_fn_name}'. "
            f"Available activation functions are: {str(_act_fn_registry.keys())}"
        )


@dataclass
class FeedForwardConfig(UpProjConfigMixin):
    proj_factor: float = 1.3
    act_fn: str = "gelu"
    embedding_dim: int = -1
    dropout: float = 0.0
    bias: bool = False
    init_distribution: InitDistribution = "normal"
    """Distribution type from which to sample the weights."""
    output_init_fn: InitFnName = "wang"
    """Initialization function for the output projection layer."""
    ff_type: Literal["ffn_gated", "ffn"] = "ffn_gated"
    dtype: str = "bfloat16"
    parallel: ParallelConfig | None = None

    _num_blocks: int = 1

    def __post_init__(self):
        self._set_proj_up_dim(embedding_dim=self.embedding_dim)
        assert self.act_fn in _act_fn_registry, f"Unknown activation function {self.act_fn}"

    @property
    def _dtype(self) -> jnp.dtype:
        """
        Returns the real dtype instead of the str from configs.

        Returns:
            The jnp dtype corresponding to the string value.
        """
        return getattr(jnp, self.dtype)


class GatedFeedForward(nn.Module):
    config: FeedForwardConfig

    @nn.compact
    def __call__(self, x: jax.Array, train: bool = True, **kwargs) -> jax.Array:
        tp_size = jax.lax.psum(1, self.config.parallel.model_axis_name)
        assert self.config._proj_up_dim % tp_size == 0, "proj_up_dim must be divisible by the number of model replicas"
        embedding_dim = x.shape[-1] * tp_size

        # Helper function to create a dense layer for up-projection.
        # We force it to not async, as the async version would not support the splitting of the projection as well
        # by default. Could be implemented with extra splitting logic, but for now we keep it simple.
        def up_layer_fn(name):
            return TPDense(
                dense_fn=partial(
                    nn.Dense,
                    dtype=self.config._dtype,
                    features=self.config._proj_up_dim // tp_size,
                ),
                model_axis_name=self.config.parallel.model_axis_name,
                tp_mode="gather",
                skip_communication=True,
                kernel_init=small_init(embedding_dim, distribution=self.config.init_distribution),
                use_bias=self.config.bias,
                name=name,
            )

        # Up-projection.
        x = jax.lax.all_gather(x, self.config.parallel.model_axis_name, axis=-1, tiled=True)
        up_proj = up_layer_fn(name="proj_up")(x)
        gate_preact = up_layer_fn(name="proj_up_gate")(x)
        gate_act = get_act_fn(self.config.act_fn)(gate_preact)
        x_interm = gate_act * up_proj

        # Down-projection.
        tp_dense_fn = (
            partial(TPAsyncDense, use_bidirectional_gather=True, use_bidirectional_scatter=True)
            if self.config.parallel.tp_async_dense
            else TPDense
        )
        out = tp_dense_fn(
            dense_fn=partial(
                nn.Dense,
                dtype=self.config._dtype,
                features=embedding_dim // tp_size if self.config.parallel.tp_async_dense else embedding_dim,
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
        )(x_interm)
        out = nn.Dropout(rate=self.config.dropout, deterministic=not train)(out)
        return out


class FeedForward(nn.Module):
    config: FeedForwardConfig

    @nn.compact
    def __call__(self, x: jax.Array, train: bool = True, **kwargs) -> jax.Array:
        tp_size = jax.lax.psum(1, self.config.parallel.model_axis_name)
        assert self.config._proj_up_dim % tp_size == 0, "proj_up_dim must be divisible by the number of model replicas"
        embedding_dim = x.shape[-1] * tp_size

        # Configure Tensor Parallel dense layer.
        tp_dense_fn = (
            partial(TPAsyncDense, use_bidirectional_gather=True, use_bidirectional_scatter=True)
            if self.config.parallel.tp_async_dense
            else TPDense
        )

        # Up-projection.
        x = tp_dense_fn(
            dense_fn=partial(
                nn.Dense,
                dtype=self.config._dtype,
                features=self.config._proj_up_dim // tp_size,
            ),
            model_axis_name=self.config.parallel.model_axis_name,
            tp_mode="gather",
            kernel_init=small_init(embedding_dim, distribution=self.config.init_distribution),
            use_bias=self.config.bias,
            name="proj_up",
        )(x)
        x = get_act_fn(self.config.act_fn)(x)

        # Down-projection.
        x = tp_dense_fn(
            dense_fn=partial(
                nn.Dense,
                dtype=self.config._dtype,
                features=embedding_dim // tp_size if self.config.parallel.tp_async_dense else embedding_dim,
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
        )(x)
        x = nn.Dropout(rate=self.config.dropout, deterministic=not train)(x)
        return x


def create_feedforward(config: FeedForwardConfig, name: str = "ffn") -> nn.Module:
    if config.ff_type == "ffn_gated":
        return GatedFeedForward(config, name=name)
    elif config.ff_type == "ffn":
        return FeedForward(config, name=name)
    else:
        raise ValueError(f"Unknown feedforward type {config.ff_type}")
