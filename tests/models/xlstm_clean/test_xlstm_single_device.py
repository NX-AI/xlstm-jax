#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from typing import Any

import jax
import jax.numpy as jnp
import jax.test_util
import numpy as np
import pytest
from flax import linen as nn

from xlstm_jax.models.xlstm_clean.blocks.mlstm.backend.simple import (
    parallel_stabilized_simple,
    recurrent_step_stabilized_simple,
)
from xlstm_jax.models.xlstm_clean.blocks.mlstm.block import mLSTMBlock, mLSTMBlockConfig
from xlstm_jax.models.xlstm_clean.blocks.mlstm.cell import mLSTMCellConfig
from xlstm_jax.models.xlstm_clean.blocks.mlstm.layer import mLSTMLayer, mLSTMLayerConfig
from xlstm_jax.models.xlstm_clean.blocks.xlstm_block import xLSTMBlock, xLSTMBlockConfig
from xlstm_jax.models.xlstm_clean.components.conv import CausalConv1d, CausalConv1dConfig
from xlstm_jax.models.xlstm_clean.components.feedforward import FeedForwardConfig, create_feedforward
from xlstm_jax.models.xlstm_clean.components.init import bias_linspace_init
from xlstm_jax.models.xlstm_clean.components.linear_headwise import LinearHeadwiseExpand, LinearHeadwiseExpandConfig
from xlstm_jax.models.xlstm_clean.components.ln import LayerNorm, MultiHeadLayerNorm
from xlstm_jax.models.xlstm_clean.xlstm_block_stack import xLSTMBlockStack, xLSTMBlockStackConfig
from xlstm_jax.models.xlstm_clean.xlstm_lm_model import xLSTMLMModel, xLSTMLMModelConfig
from xlstm_jax.models.xlstm_pytorch.blocks.mlstm.backend.simple import (
    parallel_stabilized_simple as parallel_stabilized_simple_torch,
    recurrent_step_stabilized_simple as recurrent_step_stabilized_simple_torch,
)


def test_xLSTMLMModel():
    """Test a full xLSTM model."""
    config = xLSTMLMModelConfig(
        vocab_size=100,
        embedding_dim=16,
        num_blocks=2,
        context_length=128,
        tie_weights=False,
        add_embedding_dropout=True,
        add_post_blocks_norm=True,
        dtype="bfloat16",
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                proj_factor=2.0,
                conv1d_kernel_size=4,
                num_heads=4,
                dropout=0.2,
                embedding_dim=16,
                context_length=128,
                vmap_qk=True,
                dtype="bfloat16",
            )
        ),
    )
    rng = jax.random.PRNGKey(0)
    inp_rng, model_rng = jax.random.split(rng, 2)
    model = xLSTMLMModel(config=config)
    input_tensor = jax.random.randint(inp_rng, (2, 128), 0, 100)
    params = model.init(model_rng, input_tensor)
    logits, intermediates = model.apply(params, input_tensor, capture_intermediates=True)
    assert logits.shape == (2, 128, 100)
    assert logits.dtype == jnp.float32
    intermediates = _pytree_get_dtype(intermediates)
    assert all(
        all(v == jnp.bfloat16 for v in intermediates[key])
        or key in ("intermediates.lm_head.__call__", "intermediates.__call__")
        for key in intermediates
    )


def _pytree_get_dtype(tree: Any) -> dict[str, Any]:
    """Get the dtype of all elements in a pytree."""
    if isinstance(tree, dict):
        new_dict = {}
        for key in tree:
            sub_dict = _pytree_get_dtype(
                tree[key],
            )
            if isinstance(sub_dict, dict):
                for sub_key in sub_dict:
                    new_dict[key + "." + sub_key] = sub_dict[sub_key]
            else:
                new_dict[key] = sub_dict
        return new_dict
    if isinstance(tree, tuple):
        return tuple(t.dtype for t in tree)
    return tree.dtype


def test_xLSTMBlockStack():
    """Test a stack of xLSTM blocks."""
    config = xLSTMBlockStackConfig(
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                conv1d_kernel_size=4,
                qkv_proj_blocksize=4,
                num_heads=4,
                proj_factor=2.0,
                embedding_dim=16,
                bias=True,
                dropout=0.0,
                context_length=128,
                dtype="bfloat16",
            ),
            _num_blocks=8,
            _block_idx=0,
        ),
        context_length=128,
        num_blocks=8,
        embedding_dim=16,
        add_post_blocks_norm=True,
        bias=True,
        dropout=0.0,
        dtype="bfloat16",
        slstm_at=[],
    )
    rng = jax.random.PRNGKey(0)
    inp_rng, model_rng, dp_rng = jax.random.split(rng, 3)
    block = xLSTMBlockStack(config=config)
    input_tensor = jax.random.normal(inp_rng, (2, 128, 16), dtype=jnp.bfloat16)
    params = block.init(model_rng, input_tensor)
    output_tensor = block.apply(params, input_tensor, rngs={"dropout": dp_rng}, train=True)
    assert output_tensor.shape == input_tensor.shape, f"Expected shape {input_tensor.shape}, got {output_tensor.shape}"
    assert output_tensor.dtype == jnp.bfloat16


def test_ln():
    """Test LayerNorm and MultiHeadLayerNorm."""
    rng = jax.random.PRNGKey(42)
    inp_rng, model_rng = jax.random.split(rng)
    x = jax.random.normal(inp_rng, (2, 3, 4))
    model = LayerNorm(dtype=jnp.float32)
    params = model.init(model_rng, x)
    y = model.apply(params, x)
    assert params["params"]["scale"].shape == (4,)
    assert len(params["params"]) == 1
    assert y.shape == (2, 3, 4)
    assert jnp.allclose(y.std(axis=-1), 1, atol=1e-3), y.std(axis=-1)
    assert jnp.allclose(y.mean(axis=-1), 0, atol=1e-3), y.mean(axis=-1)

    x = jax.random.normal(inp_rng, (2, 8, 3, 384))
    model = MultiHeadLayerNorm(dtype=jnp.float32, axis=1, eps=1e-5)
    params = model.init(model_rng, x)
    y = model.apply(params, x)
    assert params["params"]["scale"].shape == (x.shape[1], x.shape[-1])
    assert len(params["params"]) == 1
    assert y.shape == x.shape
    assert jnp.allclose(y.std(axis=-1), 1, atol=1e-3), y.std(axis=-1)
    assert jnp.allclose(y.mean(axis=-1), 0, atol=1e-3), y.mean(axis=-1)

    # Compare to Group normalization
    group_norm = nn.GroupNorm(dtype=jnp.float32, num_groups=x.shape[1], epsilon=1e-5)
    x_gn = x.transpose(0, 2, 1, 3).reshape(x.shape[0] * x.shape[2], -1)
    params = group_norm.init(model_rng, x_gn)
    y_gn = group_norm.apply(params, x_gn)
    y_gn = y_gn.reshape(x.shape[0], x.shape[2], x.shape[1], x.shape[3]).transpose(0, 2, 1, 3)
    y = jax.device_get(y)
    y_gn = jax.device_get(y_gn)
    np.testing.assert_allclose(y_gn.std(axis=-1), 1, atol=1e-3)
    np.testing.assert_allclose(y_gn.mean(axis=-1), 0, atol=1e-3)
    np.testing.assert_allclose(y, y_gn)


def test_linear_headwise():
    """Test the LinearHeadwiseExpand layer."""
    config = LinearHeadwiseExpandConfig(in_features=4, num_heads=2, expand_factor_up=1)
    rng = jax.random.PRNGKey(0)
    inp_rng, model_rng = jax.random.split(rng)
    input_tensor = jax.random.normal(inp_rng, (2, 5, 4))
    model = LinearHeadwiseExpand(config)
    params = model.init(model_rng, input_tensor)
    output_tensor = model.apply(params, input_tensor)
    assert (
        output_tensor.shape == input_tensor.shape
    ), f"Expected output shape {input_tensor.shape}, but got {output_tensor.shape}"
    input_tensor = input_tensor.at[0, 0, 2].set(-1.0)
    output_tensor_new = model.apply(params, input_tensor)
    diff = (output_tensor_new - output_tensor) != 0
    assert not jnp.any(diff[1]), "Output tensor changed unexpectedly."
    assert not jnp.any(diff[0, 1:]), "Output tensor changed unexpectedly."
    assert not jnp.any(diff[0, 0, :2]), "Output tensor changed unexpectedly."
    assert jnp.all(diff[0, 0, 3:]), "Output tensor changed unexpectedly."


def test_bias_linear_init():
    """Test the bias_linspace_init function."""
    init_fn = bias_linspace_init(start=0.0, end=6.0)
    key = jax.random.PRNGKey(0)
    shape = (7,)
    dtype = jnp.float32
    init_vals = init_fn(key, shape, dtype)
    assert init_vals.shape == shape
    assert jnp.allclose(init_vals, jnp.arange(0, 7), atol=1e-3), init_vals
    assert init_vals.dtype == dtype
    init_vals_small = init_fn(key, (2,), dtype)
    assert init_vals_small.shape == (2,)
    assert jnp.allclose(init_vals_small, jnp.array([0.0, 6.0]), atol=1e-3), init_vals_small
    init_vals_long = init_fn(key, (100,), dtype)
    assert init_vals_long.shape == (100,)
    assert jnp.allclose(init_vals_long, jnp.linspace(0, 6, num=100), atol=1e-3), init_vals_long
    init_vals_bfloat16 = init_fn(key, shape, jnp.bfloat16)
    assert init_vals_bfloat16.shape == shape
    assert jnp.allclose(init_vals_bfloat16, jnp.arange(0, 7), atol=1e-3), init_vals_bfloat16
    assert init_vals_bfloat16.dtype == jnp.bfloat16


def test_feedforward():
    """Test the FeedForward layer."""
    config = FeedForwardConfig(
        proj_factor=1.3,
        act_fn="gelu",
        embedding_dim=16,
        dropout=0.0,
        bias=False,
        ff_type="ffn_gated",
        dtype="bfloat16",
    )
    rng = jax.random.PRNGKey(0)
    inp_rng, model_rng, drp_rng = jax.random.split(rng, 3)
    input_tensor = jax.random.normal(inp_rng, (2, 128, 16), dtype=jnp.bfloat16)
    model = create_feedforward(config)
    params = model.init(model_rng, input_tensor, train=False)
    output_tensor = model.apply(params, input_tensor, train=True, rngs={"dropout": drp_rng})
    assert output_tensor.shape == input_tensor.shape, f"Expected shape {input_tensor.shape}, got {output_tensor.shape}"
    assert output_tensor.dtype == jnp.bfloat16
    assert jnp.allclose(output_tensor.mean(), 0, atol=1e-1), output_tensor.mean()


def test_causal_conv1d():
    """Test the CausalConv1d layer."""
    config = CausalConv1dConfig(feature_dim=3, kernel_size=4, channel_mixing=False)
    rng = jax.random.PRNGKey(0)
    inp_rng, model_rng = jax.random.split(rng)
    input_tensor = jax.random.normal(inp_rng, (2, 5, 3))
    model = CausalConv1d(config)
    params = model.init(model_rng, input_tensor)
    output_tensor = model.apply(params, input_tensor)
    assert (
        output_tensor.shape == input_tensor.shape
    ), f"Expected output shape {input_tensor.shape}, but got {output_tensor.shape}"
    input_tensor = input_tensor.at[0, 2, 0].set(-1.0)
    output_tensor_new = model.apply(params, input_tensor)
    diff = (output_tensor_new - output_tensor) != 0
    assert diff.any(), "Expected output to change after changing input, but it remained the same"
    assert diff[0, 2:, 0].all(), f"Expected output to change after changing input, but it remained the same: {diff}"
    assert not diff[:, :2, :].any(), f"Expected output to remain unchanged after changing input, but it changed: {diff}"
    assert not diff[
        :, 2:, 1:
    ].any(), f"Expected output to remain unchanged after changing input, but it changed: {diff}"
    assert not diff[1].any(), f"Expected output to remain unchanged after changing input, but it changed: {diff}"


def test_xLSTMBlock():
    """Test the xLSTMBlock."""
    config = xLSTMBlockConfig(
        mlstm=mLSTMLayerConfig(
            conv1d_kernel_size=4,
            qkv_proj_blocksize=4,
            num_heads=4,
            proj_factor=2.0,
            embedding_dim=16,
            bias=True,
            dropout=0.2,
            context_length=128,
            dtype="bfloat16",
        ),
        _num_blocks=1,
        _block_idx=0,
        feedforward=FeedForwardConfig(
            proj_factor=4.0,
            embedding_dim=16,
            dropout=0.2,
            dtype="bfloat16",
        ),
    )
    rng = jax.random.PRNGKey(0)
    inp_rng, model_rng, dp_rng = jax.random.split(rng, 3)
    block = xLSTMBlock(config=config)
    input_tensor = jax.random.normal(inp_rng, (2, 128, 16), dtype=jnp.bfloat16)
    params = block.init(model_rng, input_tensor)
    output_tensor = block.apply(params, input_tensor, rngs={"dropout": dp_rng}, train=True)
    assert output_tensor.shape == input_tensor.shape, f"Expected shape {input_tensor.shape}, got {output_tensor.shape}"
    assert output_tensor.dtype == jnp.bfloat16, f"Expected dtype {jnp.bfloat16}, got {output_tensor.dtype}"


@pytest.mark.parametrize("vmap_qk", [True, False])
def test_mLSTMLayer(vmap_qk: bool):
    """Test the mLSTMLayer."""
    config = mLSTMLayerConfig(
        embedding_dim=8,
        context_length=16,
        num_heads=4,
        proj_factor=2.0,
        conv1d_kernel_size=4,
        qkv_proj_blocksize=4,
        vmap_qk=vmap_qk,
        mlstm_cell=mLSTMCellConfig(
            context_length=16,
            num_heads=4,
            embedding_dim=8,
        ),
    )
    rng = jax.random.PRNGKey(0)
    inp_rng, model_rng = jax.random.split(rng)
    input_tensor = jax.random.normal(inp_rng, (2, config.context_length, config.embedding_dim))
    model = mLSTMLayer(config)
    params = model.init(model_rng, input_tensor)
    output_tensor = model.apply(params, input_tensor)
    assert (
        output_tensor.shape == input_tensor.shape
    ), f"Expected output shape {input_tensor.shape}, but got {output_tensor.shape}"


def test_mLSTMBlock():
    """Test the mLSTMBlock."""
    config = mLSTMBlockConfig(
        mlstm=mLSTMLayerConfig(
            conv1d_kernel_size=4,
            qkv_proj_blocksize=4,
            num_heads=4,
            proj_factor=2.0,
            embedding_dim=16,
            bias=True,
            dropout=0.0,
            context_length=128,
            dtype="bfloat16",
        ),
        _num_blocks=1,
        _block_idx=0,
    )
    rng = jax.random.PRNGKey(0)
    inp_rng, model_rng, dp_rng = jax.random.split(rng, 3)
    block = mLSTMBlock(config=config)
    input_tensor = jax.random.normal(inp_rng, (2, 128, 16))
    params = block.init(model_rng, input_tensor)
    output_tensor = block.apply(params, input_tensor, rngs={"dropout": dp_rng}, train=True)
    assert output_tensor.shape == (2, 128, 16)


def test_mLSTMBackend():
    """Test the mLSTM backend functions."""
    rng = np.random.default_rng(42)
    B, NH, S, DH = 2, 3, 4, 5
    q = rng.standard_normal((B, NH, S, DH)).astype(jnp.float32)
    k = rng.standard_normal((B, NH, S, DH)).astype(jnp.float32)
    v = rng.standard_normal((B, NH, S, DH)).astype(jnp.float32)
    igate_preact = rng.standard_normal((B, NH, S, 1)).astype(jnp.float32)
    fgate_preact = rng.standard_normal((B, NH, S, 1)).astype(jnp.float32)

    h_tilde_state = parallel_stabilized_simple(q, k, v, igate_preact, fgate_preact)
    assert h_tilde_state.shape == (B, NH, S, DH)

    c_state = rng.standard_normal((B, NH, DH, DH)).astype(jnp.float32)
    n_state = rng.standard_normal((B, NH, DH, 1)).astype(jnp.float32)
    m_state = rng.standard_normal((B, NH, 1, 1)).astype(jnp.float32)

    h, (c_state_new, n_state_new, m_state_new) = recurrent_step_stabilized_simple(
        c_state,
        n_state,
        m_state,
        q[:, :, 0:1],
        k[:, :, 0:1],
        v[:, :, 0:1],
        igate_preact[:, :, 0:1],
        fgate_preact[:, :, 0:1],
    )
    assert h.shape == (B, NH, 1, DH)
    assert c_state_new.shape == (B, NH, DH, DH)
    assert n_state_new.shape == (B, NH, DH, 1)
    assert m_state_new.shape == (B, NH, 1, 1)

    if parallel_stabilized_simple_torch is not None and recurrent_step_stabilized_simple_torch is not None:
        import torch

        q_torch = torch.from_numpy(q)
        k_torch = torch.from_numpy(k)
        v_torch = torch.from_numpy(v)
        igate_preact_torch = torch.from_numpy(igate_preact)
        fgate_preact_torch = torch.from_numpy(fgate_preact)

        h_tilde_state_torch = parallel_stabilized_simple_torch(
            q_torch, k_torch, v_torch, igate_preact_torch, fgate_preact_torch
        )
        assert jnp.allclose(h_tilde_state, h_tilde_state_torch.numpy(), atol=1e-3)

        c_state_torch = torch.from_numpy(c_state)
        n_state_torch = torch.from_numpy(n_state)
        m_state_torch = torch.from_numpy(m_state)

        h_torch, (c_state_new_torch, n_state_new_torch, m_state_new_torch) = recurrent_step_stabilized_simple_torch(
            c_state_torch,
            n_state_torch,
            m_state_torch,
            q_torch[:, :, 0:1],
            k_torch[:, :, 0:1],
            v_torch[:, :, 0:1],
            igate_preact_torch[:, :, 0:1],
            fgate_preact_torch[:, :, 0:1],
        )
        assert jnp.allclose(h, h_torch.numpy(), atol=1e-3)
        assert jnp.allclose(c_state_new, c_state_new_torch.numpy(), atol=1e-3)
        assert jnp.allclose(n_state_new, n_state_new_torch.numpy(), atol=1e-3)
        assert jnp.allclose(m_state_new, m_state_new_torch.numpy(), atol=1e-3)
