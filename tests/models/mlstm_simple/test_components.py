from functools import partial

try:
    from mlstm_simple_torch.mlstm_simple.components import MultiHeadLayerNorm, MultiHeadRMSNorm
except ImportError:
    MultiHeadLayerNorm = None
    MultiHeadRMSNorm = None

import jax
import numpy as np
import pytest
import torch

from xlstm_jax.models.xlstm_parallel.components.normalization import MultiHeadNormLayer


def get_jax_norm_layer(norm_type: str):
    if norm_type == "rmsnorm":
        return partial(MultiHeadNormLayer, norm_type="rmsnorm")
    elif norm_type == "layernorm":
        return partial(MultiHeadNormLayer, norm_type="layernorm")
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")


@pytest.mark.skip(reason="This tests requires imports submodules, which does not work currently.")
@pytest.mark.parametrize("jax_and_torch_norm_types", [["rmsnorm", MultiHeadRMSNorm], ["layernorm", MultiHeadLayerNorm]])
@pytest.mark.parametrize("B", [1, 8])
@pytest.mark.parametrize("S", [1, 16])
@pytest.mark.parametrize("NH", [1, 8])
@pytest.mark.parametrize("DH", [384])
def test_multihead_norm(jax_and_torch_norm_types, B, S, NH, DH):
    rng = jax.random.PRNGKey(42)
    inp_rng, model_rng = jax.random.split(rng)
    x = jax.random.normal(inp_rng, (B, S, NH, DH))
    x_np = jax.device_get(x)

    jax_norm_type, torch_norm_layer = jax_and_torch_norm_types

    jax_norm_layer = get_jax_norm_layer(jax_norm_type)

    jax_mh_norm = jax_norm_layer(weight=True, bias=False, eps=1e-6, axis=2)
    params = jax_mh_norm.init(model_rng, x)
    y = jax_mh_norm.apply(params, x)
    assert params["params"]["scale"].shape == (x.shape[2], x.shape[-1])
    assert len(params["params"]) == 1
    assert y.shape == x.shape

    norm_params_rand = jax.random.normal(model_rng, params["params"]["scale"].shape)
    norm_params_rand_np = jax.device_get(norm_params_rand)

    params["params"]["scale"] = norm_params_rand

    y_jax = jax_mh_norm.apply(params, x)

    torch_mh_norm = torch_norm_layer(num_heads=NH, head_dim=DH, eps=1e-6, use_weight=True)
    # with flatten()
    norm_params_rand_torch = torch.from_numpy(norm_params_rand_np).clone()
    state_dict_from_jax = {"weight": norm_params_rand_torch.flatten()}

    torch_mh_norm.load_state_dict(state_dict_from_jax)

    x_torch = torch.from_numpy(x_np).clone()

    y_torch = torch_mh_norm(x_torch)

    y_jax_np = jax.device_get(y_jax.reshape(B, S, -1))
    y_torch_np = y_torch.detach().numpy()

    np.testing.assert_allclose(y_jax_np, y_torch_np, atol=1e-5)
    # with reshape()
    norm_params_rand_torch = torch.from_numpy(norm_params_rand_np).clone()
    # state_dict_from_jax = {"weight": norm_params_rand_torch.reshape(-1)}

    torch_mh_norm.load_state_dict(state_dict_from_jax)

    x_torch = torch.from_numpy(x_np).clone()

    y_torch = torch_mh_norm(x_torch)

    y_jax_np = jax.device_get(y_jax.reshape(B, S, -1))
    y_torch_np = y_torch.detach().numpy()

    np.testing.assert_allclose(y_jax_np, y_torch_np, atol=1e-5)
