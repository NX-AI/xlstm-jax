from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
from flax import linen as nn

from ..utils import UpProjConfigMixin
from .init import small_init, wang_init

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
        assert False, f'Unknown activation function name "{act_fn_name}". Available activation functions are: {str(_act_fn_registry.keys())}'


@dataclass
class FeedForwardConfig(UpProjConfigMixin):
    proj_factor: float = 1.3
    act_fn: str = "gelu"
    embedding_dim: int = -1
    dropout: float = 0.0
    bias: bool = False
    ff_type: Literal["ffn_gated"] = "ffn_gated"
    dtype: jnp.dtype = jnp.bfloat16

    _num_blocks: int = 1

    def __post_init__(self):
        self._set_proj_up_dim(embedding_dim=self.embedding_dim)
        assert self.act_fn in _act_fn_registry, f"Unknown activation function {self.act_fn}"


class GatedFeedForward(nn.Module):
    config: FeedForwardConfig

    @nn.compact
    def __call__(self, x: jax.Array, train: bool = True, **kwargs) -> jax.Array:
        up_out = nn.Dense(
            features=2 * self.config._proj_up_dim,
            kernel_init=small_init(x.shape[-1]),
            use_bias=self.config.bias,
            dtype=self.config.dtype,
            name="proj_up",
        )(x)
        gate_preact, up_proj = jnp.split(up_out, 2, axis=-1)
        gate_act = get_act_fn(self.config.act_fn)(gate_preact)
        out = nn.Dense(
            features=self.config.embedding_dim,
            kernel_init=wang_init(x.shape[-1], num_blocks=self.config._num_blocks),
            use_bias=self.config.bias,
            dtype=self.config.dtype,
            name="proj_down",
        )(gate_act * up_proj)
        out = nn.Dropout(rate=self.config.dropout, deterministic=not train)(out)
        return out


def create_feedforward(config: FeedForwardConfig) -> nn.Module:
    if config.ff_type == "ffn_gated":
        return GatedFeedForward(config)
    else:
        raise ValueError(f"Unknown feedforward type {config.ff_type}")
