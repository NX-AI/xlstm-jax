import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import jax
import jax.numpy as jnp
import numpy as np
import torch
import pytest
from typing import Any
from model.xlstm_lm_model import xLSTMLMModel as xLSTMLMModel_jax
from model.xlstm_lm_model import xLSTMLMModelConfig as xLSTMLMModelConfig_jax
from model.blocks.mlstm.block import mLSTMBlockConfig as mLSTMBlockConfig_jax
from model.blocks.mlstm.layer import mLSTMLayerConfig as mLSTMLayerConfig_jax
from model_pytorch.xlstm_lm_model import xLSTMLMModel as xLSTMLMModel_torch
from model_pytorch.xlstm_lm_model import xLSTMLMModelConfig as xLSTMLMModelConfig_torch
from model_pytorch.blocks.mlstm.block import mLSTMBlockConfig as mLSTMBlockConfig_torch
from model_pytorch.blocks.mlstm.layer import mLSTMLayerConfig as mLSTMLayerConfig_torch

PyTree = Any

# Define configuration
MODEL_CONFIGS = [
    xLSTMLMModelConfig_torch(
        vocab_size=100,
        embedding_dim=16,
        num_blocks=2,
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
        )
    )
]

@pytest.mark.parametrize(
    "torch_config", MODEL_CONFIGS
)
def test_xLSTMLMModel(torch_config):
    torch.manual_seed(0)
    np_rng = np.random.default_rng(0)
    model_torch = xLSTMLMModel_torch(torch_config)
    model_torch.reset_parameters()
    model_torch.eval()
    input_tensor = np_rng.integers(0, 100, (2, 128))
    with torch.no_grad():
        logits_torch = model_torch(torch.from_numpy(input_tensor))
    assert logits_torch.shape == (2, 128, 100)
    assert logits_torch.dtype == torch.float32

    print("Parameters", {key: value.shape for key, value in model_torch.named_parameters()})
    config_jax = mLSTMLayerConfig_jax(
        proj_factor=torch_config.mlstm_block.mlstm.proj_factor,
        conv1d_kernel_size=torch_config.mlstm_block.mlstm.conv1d_kernel_size,
        num_heads=torch_config.mlstm_block.mlstm.num_heads,
        dropout=torch_config.mlstm_block.mlstm.dropout,
        embedding_dim=torch_config.mlstm_block.mlstm.embedding_dim,
        context_length=torch_config.mlstm_block.mlstm.context_length,
        vmap_qk=False,
        dtype=jnp.float32,
    )
    config_jax = mLSTMBlockConfig_jax(
        mlstm=config_jax,
    )
    config_jax = xLSTMLMModelConfig_jax(
        vocab_size=torch_config.vocab_size,
        embedding_dim=torch_config.embedding_dim,
        num_blocks=torch_config.num_blocks,
        context_length=torch_config.context_length,
        tie_weights=torch_config.tie_weights,
        add_embedding_dropout=torch_config.add_embedding_dropout,
        add_post_blocks_norm=torch_config.add_post_blocks_norm,
        mlstm_block=config_jax,
        dtype=jnp.float32
    )
    model_jax = xLSTMLMModel_jax(config_jax)
    params_jax = model_jax.init(jax.random.PRNGKey(0), input_tensor, train=False)["params"]
    print("Params JAX", jax.tree.map(lambda x: x.shape, params_jax))
    params_jax = jax.device_get(params_jax)
    params_jax = _convert_params_torch_to_jax(params_torch=dict(model_torch.named_parameters()), params_jax=params_jax)
    logits_jax = model_jax.apply({"params": params_jax}, input_tensor, train=False)
    assert logits_jax.shape == (2, 128, 100)
    assert logits_jax.dtype == jnp.float32
    assert np.allclose(logits_torch.numpy(), logits_jax, atol=1e-6)

def _convert_params_torch_to_jax(params_torch: dict[str, Any], params_jax: PyTree | None = None):
    params_jax_new = dict()
    for key, p_torch in params_torch.items():
        param = p_torch.data.numpy()
        print("Key in", key)
        if key == 'token_embedding.weight':
            key = 'token_embedding.embedding'
        elif key.endswith('norm.weight'):
            key = key[:-len('.weight')] + ".scale"
        elif key.endswith('.weight'):
            key = key[:-len('.weight')] + ".kernel"
            if not any([key.endswith(f'.{proj}_proj.kernel') for proj in ['q', 'k', 'v']]):
                param = np.swapaxes(param, 0, -1)
        print("Key out", key)
        _add_nested_param_to_dict(params_jax_new, key, param)
    if params_jax is not None:
        _check_equal_pytree_struct(params_jax_new, params_jax)
    return params_jax_new

def _add_nested_param_to_dict(param_dict: dict[str, Any], key: str, param: Any) -> None:
    if "." in key:
        sub_key, key = key.split('.', 1)
        if sub_key not in param_dict:
            param_dict[sub_key] = dict()
        _add_nested_param_to_dict(param_dict[sub_key], key, param)
    else:
        if key in param_dict:
            raise RuntimeError(f"Found already parameter at key {key}.")
        param_dict[key] = param

def _check_equal_pytree_struct(tree1: PyTree, tree2: PyTree, full_key: str = ""):
    if isinstance(tree1, dict):
        assert isinstance(tree2, dict), f"[Key {full_key}] Found tree-1 to be a dict with keys {list(tree1.keys())}, but tree-2 is {type(tree2)}."
        tree_keys_1 = sorted(list(tree1.keys()))
        tree_keys_2 = sorted(list(tree2.keys()))
        assert tree_keys_1 == tree_keys_2, f"[Key {full_key}] Found unmatching keys in tree: {tree_keys_1} vs {tree_keys_2}."
        for key in tree_keys_1:
            _check_equal_pytree_struct(tree1[key], tree2[key], full_key=full_key + ("." if len(full_key) > 0 else "") + str(key))
    else:
        assert isinstance(tree2, type(tree1)), f"[Key {full_key}] Found tree-1 to be a {type(tree1)}, but tree-2 is a {type(tree2)}."
        assert tree1.shape == tree2.shape, f"[Key {full_key}] Found different shapes: {tree1.shape} vs {tree2.shape}."
        assert tree1.dtype == tree2.dtype, f"[Key {full_key}] Found different dtypes: {tree1.dtype} vs {tree2.dtype}."