# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck, Korbinian Pöppel
import jax
import jax.numpy as jnp
from flax import linen as nn


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False."""

    weight: bool = True
    bias: bool = False
    eps: float = 1e-5
    residual_weight: bool = True
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        return nn.LayerNorm(
            epsilon=self.eps,
            use_bias=self.bias,
            use_scale=self.weight,
            dtype=self.dtype,
        )(x)


def MultiHeadLayerNorm(*args, axis: int = 1, **kwargs):
    return nn.vmap(
        LayerNorm,
        variable_axes={"params": None},
        in_axes=axis,
        out_axes=axis,
        split_rngs={"params": False},
    )(*args, **kwargs)


def test_ln():
    rng = jax.random.PRNGKey(42)
    inp_rng, model_rng = jax.random.split(rng)
    x = jax.random.normal(inp_rng, (2, 3, 4))
    model = LayerNorm(dtype=jnp.float32)
    params = model.init(model_rng, x)
    y = model.apply(params, x)
    assert params["params"]["LayerNorm_0"]["scale"].shape == (4,)
    assert len(params["params"]["LayerNorm_0"]) == 1
    assert y.shape == (2, 3, 4)
    assert jnp.allclose(y.std(axis=-1), 1, atol=1e-3), y.std(axis=-1)
    assert jnp.allclose(y.mean(axis=-1), 0, atol=1e-3), y.mean(axis=-1)
    print("All tests for LayerNorm passed successfully.")

    x = jax.random.normal(inp_rng, (2, 8, 3, 4))
    model = MultiHeadLayerNorm(dtype=jnp.float32)
    params = model.init(model_rng, x)
    y = model.apply(params, x)
    assert params["params"]["LayerNorm_0"]["scale"].shape == (4,)
    assert len(params["params"]["LayerNorm_0"]) == 1
    assert y.shape == (2, 8, 3, 4)
    assert jnp.allclose(y.std(axis=-1), 1, atol=1e-3), y.std(axis=-1)
    assert jnp.allclose(y.mean(axis=-1), 0, atol=1e-3), y.mean(axis=-1)
    print("All tests for MultiHeadLayerNorm passed successfully.")
