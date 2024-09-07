import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import jax
import jax.numpy as jnp
import numpy as np
import torch
import pytest
from model.xlstm_lm_model import xLSTMLMModel as xLSTMLMModel_jax
from model.xlstm_lm_model import xLSTMLMModelConfig as xLSTMLMModelConfig_jax
from model.blocks.mlstm.block import mLSTMBlockConfig as mLSTMBlockConfig_jax
from model.blocks.mlstm.layer import mLSTMLayerConfig as mLSTMLayerConfig_jax
from model_pytorch.xlstm_lm_model import xLSTMLMModel as xLSTMLMModel_torch
from model_pytorch.xlstm_lm_model import xLSTMLMModelConfig as xLSTMLMModelConfig_torch
from model_pytorch.blocks.mlstm.block import mLSTMBlockConfig as mLSTMBlockConfig_torch
from model_pytorch.blocks.mlstm.layer import mLSTMLayerConfig as mLSTMLayerConfig_torch

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
    params_jax = model_jax.init(jax.random.PRNGKey(0), input_tensor, train=False)
    print("Params JAX", jax.tree.map(lambda x: x.shape, params_jax))
    logits_jax = model_jax.apply(params_jax, input_tensor, train=False)
    assert logits_jax.shape == (2, 128, 100)
    assert logits_jax.dtype == jnp.float32
    assert np.allclose(logits_torch.numpy(), logits_jax, atol=1e-6)
