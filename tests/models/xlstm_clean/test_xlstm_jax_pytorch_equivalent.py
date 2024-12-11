#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from typing import Any

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import torch

from xlstm_jax.common_types import PyTree
from xlstm_jax.models.xlstm_clean.blocks.mlstm.block import mLSTMBlockConfig as mLSTMBlockConfig_jax
from xlstm_jax.models.xlstm_clean.blocks.mlstm.layer import mLSTMLayerConfig as mLSTMLayerConfig_jax
from xlstm_jax.models.xlstm_clean.components.ln import LayerNorm as LayerNorm_jax
from xlstm_jax.models.xlstm_clean.xlstm_lm_model import (
    xLSTMLMModel as xLSTMLMModel_jax,
    xLSTMLMModelConfig as xLSTMLMModelConfig_jax,
)
from xlstm_jax.models.xlstm_pytorch.blocks.mlstm.block import mLSTMBlockConfig as mLSTMBlockConfig_torch
from xlstm_jax.models.xlstm_pytorch.blocks.mlstm.layer import mLSTMLayerConfig as mLSTMLayerConfig_torch
from xlstm_jax.models.xlstm_pytorch.components.ln import LayerNorm as LayerNorm_torch
from xlstm_jax.models.xlstm_pytorch.xlstm_lm_model import (
    xLSTMLMModel as xLSTMLMModel_torch,
    xLSTMLMModelConfig as xLSTMLMModelConfig_torch,
)
from xlstm_jax.utils import flatten_dict

# Define configuration
MODEL_CONFIGS = [
    xLSTMLMModelConfig_torch(
        vocab_size=100,
        embedding_dim=16,
        num_blocks=1,
        context_length=128,
        tie_weights=False,
        add_embedding_dropout=True,
        add_post_blocks_norm=True,
        mlstm_block=mLSTMBlockConfig_torch(
            mlstm=mLSTMLayerConfig_torch(
                proj_factor=2.0,
                conv1d_kernel_size=4,
                num_heads=4,
                dropout=0.2,
                embedding_dim=16,
                context_length=128,
            )
        ),
    )
]
LARGE_MODEL_CONFIGS = [
    xLSTMLMModelConfig_torch(
        vocab_size=10_000,
        embedding_dim=1024,
        num_blocks=2,
        context_length=32,
        tie_weights=False,
        add_embedding_dropout=True,
        add_post_blocks_norm=True,
        mlstm_block=mLSTMBlockConfig_torch(
            mlstm=mLSTMLayerConfig_torch(
                proj_factor=2.0,
                conv1d_kernel_size=4,
                num_heads=4,
                dropout=0.2,
                embedding_dim=1024,
                context_length=32,
            )
        ),
    ),
    xLSTMLMModelConfig_torch(
        vocab_size=8192,
        embedding_dim=768,
        num_blocks=4,
        context_length=64,
        tie_weights=False,
        add_embedding_dropout=True,
        add_post_blocks_norm=True,
        mlstm_block=mLSTMBlockConfig_torch(
            mlstm=mLSTMLayerConfig_torch(
                proj_factor=3.0,
                conv1d_kernel_size=6,
                num_heads=8,
                dropout=0.0,
                embedding_dim=768,
                context_length=64,
            )
        ),
    ),
]


@pytest.mark.parametrize("config_torch", MODEL_CONFIGS)
def test_xLSTMLMModel(config_torch):
    """Test that PyTorch and JAX xLSTMLMModel produce the same output."""
    if jax.default_backend() != "cpu":
        pytest.skip("PyTorch backend can only be run on CPU so far, difference are not representative with GPU.")
    torch.manual_seed(0)
    np_rng = np.random.default_rng(0)
    model_torch = xLSTMLMModel_torch(config_torch)
    model_torch.reset_parameters()
    model_torch.eval()
    input_tensor = np_rng.integers(0, config_torch.vocab_size, (2, 128))
    with torch.no_grad():
        logits_torch = model_torch(torch.from_numpy(input_tensor))
    assert logits_torch.shape == (2, 128, config_torch.vocab_size)
    assert logits_torch.dtype == torch.float32

    config_jax = mLSTMLayerConfig_jax(
        proj_factor=config_torch.mlstm_block.mlstm.proj_factor,
        conv1d_kernel_size=config_torch.mlstm_block.mlstm.conv1d_kernel_size,
        num_heads=config_torch.mlstm_block.mlstm.num_heads,
        dropout=config_torch.mlstm_block.mlstm.dropout,
        embedding_dim=config_torch.mlstm_block.mlstm.embedding_dim,
        context_length=config_torch.mlstm_block.mlstm.context_length,
        vmap_qk=False,
        dtype="float32",
    )
    config_jax = mLSTMBlockConfig_jax(
        mlstm=config_jax,
    )
    config_jax = xLSTMLMModelConfig_jax(
        vocab_size=config_torch.vocab_size,
        embedding_dim=config_torch.embedding_dim,
        num_blocks=config_torch.num_blocks,
        context_length=config_torch.context_length,
        tie_weights=config_torch.tie_weights,
        add_embedding_dropout=config_torch.add_embedding_dropout,
        add_post_blocks_norm=config_torch.add_post_blocks_norm,
        mlstm_block=config_jax,
        dtype="float32",
    )
    model_jax = xLSTMLMModel_jax(config_jax)
    params_jax = model_jax.init(jax.random.PRNGKey(0), input_tensor, train=False)["params"]
    params_jax = jax.device_get(params_jax)

    # Check equal number of parameters.
    num_torch_params = sum(p.numel() for p in model_torch.parameters())
    num_jax_params = sum(p.size for p in jax.tree.leaves(params_jax))
    assert num_torch_params == num_jax_params, f"Number of parameters differ: {num_torch_params} vs {num_jax_params}."

    # Convert PyTorch parameters to JAX parameters.
    params_jax = _convert_params_torch_to_jax(
        params_torch=dict(model_torch.named_parameters()), params_jax=params_jax, config=config_torch
    )
    logits_jax = model_jax.apply({"params": params_jax}, input_tensor, train=False)
    assert logits_jax.shape == (2, 128, 100)
    assert logits_jax.dtype == jnp.float32
    np.testing.assert_allclose(logits_torch.numpy(), logits_jax, atol=1e-5, rtol=1e-5)

    # Add a bit of noise to PyTorch parameters and check equivalence again.
    torch.manual_seed(0)
    for p in model_torch.parameters():
        p.data.add_(torch.randn_like(p) * 1e-1)
    with torch.no_grad():
        logits_torch = model_torch(torch.from_numpy(input_tensor))
    params_jax = _convert_params_torch_to_jax(
        params_torch=dict(model_torch.named_parameters()), params_jax=params_jax, config=config_torch
    )
    logits_jax = model_jax.apply({"params": params_jax}, input_tensor, train=False)
    np.testing.assert_allclose(logits_torch.numpy(), logits_jax, atol=1e-5, rtol=1e-5)

    # Verify equal gradients.
    logits_torch = model_torch(torch.from_numpy(input_tensor))
    loss_torch = torch.nn.functional.cross_entropy(
        logits_torch.view(-1, config_torch.vocab_size), torch.from_numpy(input_tensor).flatten()
    )
    loss_torch.backward()
    grads_torch = {name: p.grad for name, p in model_torch.named_parameters()}

    def loss_fn(params: PyTree, input_tensor: jnp.ndarray) -> jnp.ndarray:
        logits = model_jax.apply({"params": params}, input_tensor, train=True)
        return optax.softmax_cross_entropy_with_integer_labels(
            logits.reshape(-1, config_jax.vocab_size), input_tensor.flatten()
        ).mean()

    grads_jax = jax.grad(loss_fn)(params_jax, jnp.array(input_tensor))
    grads_jax = jax.device_get(grads_jax)
    grads_torch_to_jax = _convert_params_torch_to_jax(
        params_torch=grads_torch, params_jax=grads_jax, config=config_torch, is_grad=True
    )
    grads_jax = flatten_dict(grads_jax)
    grads_torch_to_jax = flatten_dict(grads_torch_to_jax)
    for key in grads_torch_to_jax:
        g_torch = grads_torch_to_jax[key]
        g_jax = grads_jax[key]
        np.testing.assert_allclose(g_torch, g_jax, atol=1e-5, rtol=1e-5, err_msg=f"Gradient deviates for key: {key}")


@pytest.mark.parametrize("config_torch", LARGE_MODEL_CONFIGS)
def test_xLSTMLMModel_initialization(config_torch):
    """Test that the parameters of the PyTorch and JAX models are similar after initialization."""
    if jax.default_backend() != "cpu":
        pytest.skip("PyTorch backend can only be run on CPU so far, difference are not representative with GPU.")
    torch.manual_seed(0)
    np_rng = np.random.default_rng(0)
    model_torch = xLSTMLMModel_torch(config_torch)
    model_torch.reset_parameters()
    model_torch.eval()
    input_tensor = np_rng.integers(0, config_torch.vocab_size, (2, config_torch.context_length))

    config_jax = mLSTMLayerConfig_jax(
        proj_factor=config_torch.mlstm_block.mlstm.proj_factor,
        conv1d_kernel_size=config_torch.mlstm_block.mlstm.conv1d_kernel_size,
        num_heads=config_torch.mlstm_block.mlstm.num_heads,
        dropout=config_torch.mlstm_block.mlstm.dropout,
        embedding_dim=config_torch.mlstm_block.mlstm.embedding_dim,
        context_length=config_torch.mlstm_block.mlstm.context_length,
        vmap_qk=False,
        dtype="float32",
    )
    config_jax = mLSTMBlockConfig_jax(
        mlstm=config_jax,
    )
    config_jax = xLSTMLMModelConfig_jax(
        vocab_size=config_torch.vocab_size,
        embedding_dim=config_torch.embedding_dim,
        num_blocks=config_torch.num_blocks,
        context_length=config_torch.context_length,
        tie_weights=config_torch.tie_weights,
        add_embedding_dropout=config_torch.add_embedding_dropout,
        add_post_blocks_norm=config_torch.add_post_blocks_norm,
        mlstm_block=config_jax,
        dtype="float32",
    )
    model_jax = xLSTMLMModel_jax(config_jax)
    params_jax = model_jax.init(jax.random.PRNGKey(0), input_tensor, train=False)["params"]
    params_jax = jax.device_get(params_jax)
    params_torch_to_jax = _convert_params_torch_to_jax(
        params_torch=dict(model_torch.named_parameters()), params_jax=params_jax, config=config_torch
    )
    params_jax = flatten_dict(params_jax)
    params_torch_to_jax = flatten_dict(params_torch_to_jax)
    for key in params_torch_to_jax:
        p_torch = params_torch_to_jax[key]
        p_jax = params_jax[key]
        p_torch_mean, p_jax_mean = p_torch.mean(), p_jax.mean()
        p_torch_std, p_jax_std = p_torch.std(), p_jax.std()
        if np.prod(p_jax.shape) < 16:
            atol_mean, atol_std = 1e-1, 1e-1
        else:
            atol_mean, atol_std = 1e-2, 1e-2
        np.testing.assert_allclose(
            p_torch_mean, p_jax_mean, atol=atol_mean, err_msg=f"Mean deviates for key: {key} / shape: {p_jax.shape}"
        )
        np.testing.assert_allclose(
            p_torch_std,
            p_jax_std,
            atol=atol_std,
            rtol=0.1,
            err_msg=f"Std deviates for key: {key} / shape: {p_jax.shape}",
        )


def test_pytorch_jax_LN():
    """Test that PyTorch and JAX LayerNorm produce the same output."""
    if jax.default_backend() != "cpu":
        pytest.skip("PyTorch backend can only be run on CPU so far, difference are not representative with GPU.")
    rng = np.random.default_rng(0)
    input_tensor = rng.normal(size=(2, 100, 128)).astype(np.float32)
    torch_layernorm = torch.nn.LayerNorm(128, eps=1e-5)
    with torch.no_grad():
        torch_out = torch_layernorm(torch.from_numpy(input_tensor)).numpy()
    jax_layernorm = flax.linen.LayerNorm(epsilon=1e-5)
    jax_out, _ = jax_layernorm.init_with_output({"params": jax.random.PRNGKey(0)}, jnp.array(input_tensor))
    np.testing.assert_allclose(jax_out, torch_out, atol=1e-5, rtol=1e-5)

    torch_self_layernorm = LayerNorm_torch(128, eps=1e-5)
    with torch.no_grad():
        torch_self_out = torch_self_layernorm(torch.from_numpy(input_tensor)).numpy()
    np.testing.assert_allclose(torch_out, torch_self_out, atol=1e-5, rtol=1e-5)

    jax_self_layernorm = LayerNorm_jax(eps=1e-5)
    jax_self_out, _ = jax_self_layernorm.init_with_output({"params": jax.random.PRNGKey(0)}, jnp.array(input_tensor))
    np.testing.assert_allclose(jax_out, jax_self_out, atol=1e-5, rtol=1e-5)


def _convert_params_torch_to_jax(
    params_torch: dict[str, Any],
    params_jax: PyTree | None = None,
    config: xLSTMLMModelConfig_torch | None = None,
    is_grad: bool = False,
) -> object:
    """Convert PyTorch parameters to JAX parameters."""
    params_jax_new = {}
    for key, p_torch in params_torch.items():
        param = p_torch.data.numpy()
        if key == "token_embedding.weight":
            key = "token_embedding.embedding"
        elif key.endswith("norm.weight"):
            key = key[: -len(".weight")] + ".scale"
            if not is_grad:
                param = param + 1.0
            if key.endswith("mlstm_cell.outnorm.scale"):
                assert config is not None, "Need config to configure multi-head layer norm params correctly."
                param = param.reshape(config.mlstm_block.mlstm.num_heads, -1)
        elif key.endswith(".weight"):
            key = key[: -len(".weight")] + ".kernel"
            if not any(key.endswith(f".{proj}_proj.kernel") for proj in ["q", "k", "v"]):
                param = np.swapaxes(param, 0, -1)
            else:
                pass
                # param = np.swapaxes(param, 1, 2)
        _add_nested_param_to_dict(params_jax_new, key, param)
    if params_jax is not None:
        _check_equal_pytree_struct(params_jax_new, params_jax)
    return params_jax_new


def _add_nested_param_to_dict(param_dict: dict[str, Any], key: str, param: Any) -> None:
    """Add a nested parameter to a dictionary."""
    if "." in key:
        sub_key, key = key.split(".", 1)
        if sub_key not in param_dict:
            param_dict[sub_key] = {}
        _add_nested_param_to_dict(param_dict[sub_key], key, param)
    else:
        if key in param_dict:
            raise RuntimeError(f"Found already parameter at key {key}.")
        param_dict[key] = param


def _check_equal_pytree_struct(tree1: PyTree, tree2: PyTree, full_key: str = ""):
    """Check that two PyTrees have the same structure."""
    if isinstance(tree1, dict):
        assert isinstance(
            tree2, dict
        ), f"[Key {full_key}] Found tree-1 to be a dict with keys {list(tree1.keys())}, but tree-2 is {type(tree2)}."
        tree_keys_1 = sorted(list(tree1.keys()))
        tree_keys_2 = sorted(list(tree2.keys()))
        assert (
            tree_keys_1 == tree_keys_2
        ), f"[Key {full_key}] Found unmatching keys in tree: {tree_keys_1} vs {tree_keys_2}."
        for key in tree_keys_1:
            _check_equal_pytree_struct(
                tree1[key], tree2[key], full_key=full_key + ("." if len(full_key) > 0 else "") + str(key)
            )
    else:
        assert isinstance(
            tree2, type(tree1)
        ), f"[Key {full_key}] Found tree-1 to be a {type(tree1)}, but tree-2 is a {type(tree2)}."
        assert tree1.shape == tree2.shape, f"[Key {full_key}] Found different shapes: {tree1.shape} vs {tree2.shape}."
        assert tree1.dtype == tree2.dtype, f"[Key {full_key}] Found different dtypes: {tree1.dtype} vs {tree2.dtype}."
