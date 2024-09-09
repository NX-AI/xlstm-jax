import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from distributed.utils import simulate_CPU_devices
NUM_DEVICES = 8
simulate_CPU_devices(NUM_DEVICES)

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import optax
import flax
from flax import linen as nn
import numpy as np
import torch
import pytest
from typing import Any
from model_parallel.xlstm_lm_model import xLSTMLMModel
from model_parallel.xlstm_lm_model import xLSTMLMModelConfig
from model_parallel.blocks.mlstm.block import mLSTMBlockConfig
from model_parallel.blocks.mlstm.layer import mLSTMLayerConfig
from model_parallel.training import init_xlstm, get_train_step_fn
from model_parallel.utils import ParallelConfig
from distributed.tensor_parallel_transformer import split_array_over_mesh
from jax.sharding import PartitionSpec as P
from jax.sharding import NamedSharding
from distributed.single_gpu import Batch

PyTree = Any

# Define configuration
MODEL_CONFIGS = [
    xLSTMLMModelConfig(
        vocab_size=100,
        embedding_dim=16,
        num_blocks=1,
        context_length=128,
        tie_weights=False,
        add_embedding_dropout=True,
        add_post_blocks_norm=True,
        parallel=ParallelConfig(
            remat=(),
            fsdp_modules=(),
        ),
        dtype=jnp.float32,
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
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

def _create_mesh(config: xLSTMLMModelConfig):
    device_array = np.array(jax.devices()).reshape(-1, 1, 1)
    return Mesh(
        device_array, (config.parallel.data_axis_name, config.parallel.pipeline_axis_name, config.parallel.model_axis_name)
    )

@pytest.mark.parametrize(
    "config", MODEL_CONFIGS
)
@pytest.mark.parametrize(
    "gradient_accumulate_steps", [1, 4]
)
def test_simple_data_parallel(config: xLSTMLMModelConfig, gradient_accumulate_steps: int):
    mesh = _create_mesh(config)
    rng = jax.random.PRNGKey(42)
    model_rng, data_rng = jax.random.split(rng)
    input_array = jax.random.randint(
        data_rng, shape=(32, config.context_length), minval=0, maxval=config.vocab_size
    )
    optimizer = optax.adamw(learning_rate=1e-3)
    state = init_xlstm(config=config, mesh=mesh, rng=model_rng, input_array=input_array, optimizer=optimizer)
    assert state is not None
    assert all([p.sharding.spec == P() for p in jax.tree.leaves(state.params)]), f"Parameters should be replicated over axes, but found different sharding: {[p.sharding for p in jax.tree.leaves(state.params)]}"
    
    batch = Batch(
        inputs=jnp.pad(input_array[:, :-1], ((0, 0), (1, 0)), constant_values=0),
        labels=input_array,
    )
    train_step_fn, metrics = get_train_step_fn(
        state,
        batch=batch,
        mesh=mesh,
        config=config.parallel,
        gradient_accumulate_steps=1,
    )
    state, metrics = train_step_fn(
        state,
        metrics,
        batch,
    )
    assert all([p.sharding.spec == P() for p in jax.tree.leaves(state.params)]), f"Parameters should be replicated over axes, but found different sharding: {[p.sharding for p in jax.tree.leaves(state.params)]}"
    assert all([m.sharding.spec == P() for m in jax.tree.leaves(metrics)]), f"Metrics should be replicated over axes, but found different sharding: {[m.sharding for m in jax.tree.leaves(metrics)]}"
    
    metrics = jax.device_get(metrics)
    assert "loss" in metrics
    assert len(metrics["loss"]) == 2, f"Metrics must be a tuple."
    assert metrics["loss"][1] == input_array.size, f"Second metric element must be counting the batch elements, but does not fit to actual batch size: {input_array.size} vs {metrics['loss'][1]}"
    loss = metrics["loss"][0] / metrics["loss"][1]
    assert loss > 0, f"Loss must be greater zero, but is {loss}."

    mesh_single_device = Mesh(
        np.array(jax.devices())[0:1, None, None], (config.parallel.data_axis_name, config.parallel.pipeline_axis_name, config.parallel.model_axis_name)
    )
    state_single_device = init_xlstm(config=config, mesh=mesh_single_device, rng=model_rng, input_array=input_array, optimizer=optimizer)
    train_step_fn_single_device, metrics_single_device = get_train_step_fn(
        state_single_device,
        batch=batch,
        mesh=mesh_single_device,
        config=config.parallel,
        gradient_accumulate_steps=1,
    )
    state_single_device, metrics_single_device = train_step_fn_single_device(
        state_single_device,
        metrics_single_device,
        batch,
    )
    params_multi_device = jax.device_get(state.params)
    params_single_device = jax.device_get(state_single_device.params)
    _assert_pytree_equal(params_single_device, params_multi_device)
    metrics_single_device = jax.device_get(metrics_single_device)
    for key in metrics:
        np.testing.assert_allclose(
            np.array(metrics[key]), 
            np.array(metrics_single_device[key]),
            err_msg=f"[Metrics Key {key}] Value mismatch",
            atol=1e-2,  # Higher because values are >1000.
            rtol=1e-5,
        )

def _assert_pytree_equal(tree1: PyTree, tree2: PyTree, full_key: str = ""):
    if isinstance(tree1, dict):
        assert isinstance(tree2, dict), f"[Key {full_key}] Found tree-1 to be a dict with keys {list(tree1.keys())}, but tree-2 is {type(tree2)}."
        tree_keys_1 = sorted(list(tree1.keys()))
        tree_keys_2 = sorted(list(tree2.keys()))
        assert tree_keys_1 == tree_keys_2, f"[Key {full_key}] Found unmatching keys in tree: {tree_keys_1} vs {tree_keys_2}."
        for key in tree_keys_1:
            _assert_pytree_equal(tree1[key], tree2[key], full_key=full_key + ("." if len(full_key) > 0 else "") + str(key))
    else:
        assert isinstance(tree2, type(tree1)), f"[Key {full_key}] Found tree-1 to be a {type(tree1)}, but tree-2 is a {type(tree2)}."
        assert tree1.shape == tree2.shape, f"[Key {full_key}] Found different shapes: {tree1.shape} vs {tree2.shape}."
        assert tree1.dtype == tree2.dtype, f"[Key {full_key}] Found different dtypes: {tree1.dtype} vs {tree2.dtype}."
        np.testing.assert_allclose(tree1, tree2, err_msg=f"[Key {full_key}] Found different values.", rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "config", MODEL_CONFIGS
)
def test_fsdp(config: xLSTMLMModelConfig):
    mesh = _create_mesh(config)
    config.parallel.fsdp_modules = ("Embed", "mLSTMBlock", "LMHead")
    config.parallel.fsdp_min_weight_size = 2 ** 8  # Reduce for testing.
    config.remat = ("mLSTMBlock")
    rng = jax.random.PRNGKey(123)
    model_rng, data_rng = jax.random.split(rng)
    input_array = jax.random.randint(
        data_rng, shape=(32, config.context_length), minval=0, maxval=config.vocab_size
    )
    optimizer = optax.adamw(learning_rate=1e-3)
    state = init_xlstm(config=config, mesh=mesh, rng=model_rng, input_array=input_array, optimizer=optimizer)
    assert state is not None
    param_spec_tree = nn.get_partition_spec(state.params)
    for param in jax.tree.leaves(state.params):
        if param.size <= config.parallel.fsdp_min_weight_size:
            assert param.sharding.spec == P(), f"Parameter should have been too small for sharding, but found sharded nonetheless: {param.sharding.spec} with shape {param.shape}"
        else:
            assert param.sharding.spec != P(), f"Parameter should have been sharded, but appears replicated: {param.sharding.spec} with shape {param.shape}"
    
    batch = Batch(
        inputs=jnp.pad(input_array[:, :-1], ((0, 0), (1, 0)), constant_values=0),
        labels=input_array,
    )
    train_step_fn, metrics = get_train_step_fn(
        state,
        batch=batch,
        mesh=mesh,
        config=config.parallel,
        gradient_accumulate_steps=1,
    )
    state, metrics = train_step_fn(
        state,
        metrics,
        batch,
    )
    new_param_spec_tree = nn.get_partition_spec(state.params)
    assert new_param_spec_tree == param_spec_tree, f"Specs differ: {new_param_spec_tree} vs {param_spec_tree}"
    assert all([m.sharding.spec == P() for m in jax.tree.leaves(metrics)]), f"Metrics should be replicated over axes, but found different sharding: {[m.sharding for m in jax.tree.leaves(metrics)]}"
    
    metrics = jax.device_get(metrics)
    assert "loss" in metrics
    assert len(metrics["loss"]) == 2, f"Metrics must be a tuple."
    assert metrics["loss"][1] == input_array.size, f"Second metric element must be counting the batch elements, but does not fit to actual batch size: {input_array.size} vs {metrics['loss'][1]}"
    loss = metrics["loss"][0] / metrics["loss"][1]
    assert loss > 0, f"Loss must be greater zero, but is {loss}."