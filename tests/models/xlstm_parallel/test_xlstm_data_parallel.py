#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import linen as nn
from jax.sharding import Mesh, PartitionSpec as P

from xlstm_jax.common_types import PyTree
from xlstm_jax.dataset import Batch
from xlstm_jax.models.configs import ParallelConfig
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.block import mLSTMBlockConfig
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.cell import mLSTMCellConfig
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.layer import mLSTMLayerConfig
from xlstm_jax.models.xlstm_parallel.components.feedforward import FeedForwardConfig
from xlstm_jax.models.xlstm_parallel.training import get_train_step_fn, init_xlstm
from xlstm_jax.models.xlstm_parallel.xlstm_lm_model import xLSTMLMModelConfig
from xlstm_jax.trainer.metrics import _update_single_metric, get_metrics

# Define configuration
MODEL_CONFIGS = [
    xLSTMLMModelConfig(
        vocab_size=100,
        embedding_dim=16,
        logits_soft_cap=30.0,
        num_blocks=3,
        context_length=32,
        tie_weights=False,
        norm_type="layernorm",
        add_embedding_dropout=True,
        add_post_blocks_norm=True,
        parallel=ParallelConfig(
            remat=(),
            fsdp_modules=(),
        ),
        scan_blocks=False,
        dtype="float32",
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                proj_factor=2.0,
                conv1d_kernel_size=4,
                num_heads=4,
                dropout=0.2,
                embedding_dim=16,
                context_length=32,
                gate_input="qkv",
                mlstm_cell=mLSTMCellConfig(
                    gate_linear_headwise=True,
                    gate_soft_cap=30.0,
                ),
            )
        ),
    ),
    xLSTMLMModelConfig(
        vocab_size=100,
        embedding_dim=16,
        logits_soft_cap=None,
        num_blocks=3,
        context_length=32,
        tie_weights=False,
        norm_type="rmsnorm",
        add_embedding_dropout=True,
        add_post_blocks_norm=True,
        parallel=ParallelConfig(
            remat=(),
            fsdp_modules=(),
        ),
        scan_blocks=True,
        dtype="float32",
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                proj_factor=2.0,
                conv1d_kernel_size=4,
                num_heads=4,
                dropout=0.2,
                embedding_dim=16,
                context_length=32,
                gate_input="x_mlstm_conv_act",
                mlstm_cell=mLSTMCellConfig(
                    gate_linear_headwise=False,
                    gate_soft_cap=None,
                ),
            )
        ),
    ),
    xLSTMLMModelConfig(
        vocab_size=100,
        embedding_dim=16,
        logits_soft_cap=None,
        num_blocks=1,
        context_length=32,
        tie_weights=False,
        add_embedding_dropout=True,
        add_post_blocks_norm=True,
        parallel=ParallelConfig(
            remat=(),
            fsdp_modules=(),
        ),
        scan_blocks=True,
        dtype="float32",
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                layer_type="mlstm_v1",
                num_heads=4,
                embedding_dim=16,
                context_length=32,
                mlstm_cell=mLSTMCellConfig(
                    igate_bias_init_range=-3.0,
                ),
            ),
            feedforward=FeedForwardConfig(
                proj_factor=2.0,
                act_fn="gelu",
                embedding_dim=16,
                dropout=0.0,
                bias=False,
                ff_type="ffn",
                dtype="float32",
            ),
        ),
    ),
]


def _create_mesh(config: xLSTMLMModelConfig, fsdp_axis_size: int = 1) -> Mesh:
    """Create a mesh with the given FSDP configuration."""
    device_array = np.array(jax.devices()).reshape((-1, fsdp_axis_size, 1, 1))
    return Mesh(
        device_array,
        (
            config.parallel.data_axis_name,
            config.parallel.fsdp_axis_name,
            config.parallel.pipeline_axis_name,
            config.parallel.model_axis_name,
        ),
    )


@pytest.mark.parametrize("config", MODEL_CONFIGS)
@pytest.mark.parametrize("gradient_accumulate_steps", [1, 4])
def test_simple_data_parallel(config: xLSTMLMModelConfig, gradient_accumulate_steps: int):
    """Test that data parallel produces the same outputs as single device."""
    mesh = _create_mesh(config)
    rng = jax.random.PRNGKey(42)
    model_rng, data_rng = jax.random.split(rng)
    input_array = jax.random.randint(data_rng, shape=(32, config.context_length), minval=0, maxval=config.vocab_size)
    optimizer = optax.adamw(learning_rate=1e-3)
    state = init_xlstm(config=config, mesh=mesh, rng=model_rng, input_array=input_array, optimizer=optimizer)
    assert state is not None
    assert all(p.sharding.spec == P() for p in jax.tree.leaves(state.params)), (
        "Parameters should be replicated over axes, but found different sharding: "
        f"{[p.sharding for p in jax.tree.leaves(state.params)]}"
    )

    batch = Batch(
        inputs=jnp.pad(input_array[:, :-1], ((0, 0), (1, 0)), constant_values=0),
        targets=input_array,
    )
    train_step_fn, metrics = get_train_step_fn(
        state,
        batch=batch,
        mesh=mesh,
        config=config.parallel,
        gradient_accumulate_steps=gradient_accumulate_steps,
    )
    state, metrics = train_step_fn(
        state,
        metrics,
        batch,
    )
    assert all(p.sharding.spec == P() for p in jax.tree.leaves(state.params)), (
        "Parameters should be replicated over axes, but found different sharding: "
        f"{[p.sharding for p in jax.tree.leaves(state.params)]}"
    )
    assert all(m.sharding.spec == P() for m in jax.tree.leaves(metrics)), (
        "Metrics should be replicated over axes, but found different sharding: "
        f"{[m.sharding for m in jax.tree.leaves(metrics)]}"
    )

    metrics = jax.device_get(metrics)
    assert "loss" in metrics
    assert len(metrics["loss"]) == 2, "Metrics must be a tuple."
    assert metrics["loss"][1] == input_array.size, (
        "Second metric element must be counting the batch elements, but does not fit to actual batch size: "
        f"{input_array.size} vs {metrics['loss'][1]}"
    )
    loss = metrics["loss"][0] / metrics["loss"][1]
    assert loss > 0, f"Loss must be greater zero, but is {loss}."

    mesh_single_device = Mesh(
        np.array(jax.devices())[0:1, None, None, None],
        (
            config.parallel.data_axis_name,
            config.parallel.fsdp_axis_name,
            config.parallel.pipeline_axis_name,
            config.parallel.model_axis_name,
        ),
    )
    state_single_device = init_xlstm(
        config=config, mesh=mesh_single_device, rng=model_rng, input_array=input_array, optimizer=optimizer
    )
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
    _assert_pytree_equal(
        params_single_device,
        params_multi_device,
        # Currently high tolerance for GPU, to be checked why.
        rtol=1e-4 if jax.default_backend() == "cpu" else 1e-2,
        atol=1e-4 if jax.default_backend() == "cpu" else 1e-2,
    )
    metrics_single_device = jax.device_get(metrics_single_device)

    host_metrics, host_metrics_single_device = {}, {}
    for key, (value, count) in metrics.items():
        _update_single_metric(host_metrics, key=key, value=value, count=count)
    for key, (value, count) in metrics_single_device.items():
        _update_single_metric(host_metrics_single_device, key=key, value=value, count=count)

    _, host_metrics = get_metrics(host_metrics)
    _, host_metrics_single_device = get_metrics(host_metrics_single_device)
    for key in metrics:
        np.testing.assert_allclose(
            np.array(host_metrics[key]),
            np.array(host_metrics_single_device[key]),
            err_msg=f"[Metrics Key {key}] Value mismatch",
            atol=1e-2,  # Higher because values are >1000.
            rtol=1e-5 if jax.default_backend() == "cpu" else 1e-4,
        )


def _assert_pytree_equal(
    tree1: PyTree, tree2: PyTree, full_key: str = "", rtol: float = 1e-4, atol: float = 1e-4
) -> None:
    """Assert that two pytrees are equal."""
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
            _assert_pytree_equal(
                tree1[key],
                tree2[key],
                full_key=full_key + ("." if len(full_key) > 0 else "") + str(key),
                rtol=rtol,
                atol=atol,
            )
    else:
        assert isinstance(
            tree2, type(tree1)
        ), f"[Key {full_key}] Found tree-1 to be a {type(tree1)}, but tree-2 is a {type(tree2)}."
        if isinstance(tree1, nn.Partitioned):
            tree1 = tree1.value
            tree2 = tree2.value
        assert tree1.shape == tree2.shape, f"[Key {full_key}] Found different shapes: {tree1.shape} vs {tree2.shape}."
        assert tree1.dtype == tree2.dtype, f"[Key {full_key}] Found different dtypes: {tree1.dtype} vs {tree2.dtype}."
        np.testing.assert_allclose(
            tree1,
            tree2,
            err_msg=f"[Key {full_key}] Found different values.",
            rtol=rtol,
            atol=atol,
        )


@pytest.mark.parametrize("config", MODEL_CONFIGS)
def test_fsdp(config: xLSTMLMModelConfig):
    """Test that FSDP shards parameters correctly."""
    mesh = _create_mesh(config, fsdp_axis_size=pytest.num_devices)
    config = dataclasses.replace(
        config,
        parallel=dataclasses.replace(
            config.parallel,
            fsdp_modules=("Embed", "mLSTMBlock", "LMHead"),
            fsdp_min_weight_size=2**8,  # Reduce for testing.
            remat=("xLSTMResBlock", "FFNResBlock"),
        ),
    )
    rng = jax.random.PRNGKey(123)
    model_rng, data_rng = jax.random.split(rng)
    input_array = jax.random.randint(data_rng, shape=(32, config.context_length), minval=0, maxval=config.vocab_size)
    optimizer = optax.adamw(learning_rate=1e-3)
    state = init_xlstm(config=config, mesh=mesh, rng=model_rng, input_array=input_array, optimizer=optimizer)
    assert state is not None
    param_spec_tree = nn.get_partition_spec(state.params)
    for param in jax.tree.leaves(state.params):
        if pytest.num_devices == 1:
            # Nothing to be sharded over single device.
            continue
        if param.size <= config.parallel.fsdp_min_weight_size:
            assert param.sharding.spec == P(), (
                f"Parameter should have been too small for sharding, but found sharded nonetheless: "
                f"{param.sharding.spec} with shape {param.shape}"
            )
        if param.size > config.parallel.fsdp_min_weight_size * (config.num_blocks if config.scan_blocks else 1):
            # For stacked blocks, ignore parameters in this range.
            assert param.sharding.spec != P(), (
                "Parameter should have been sharded, but appears replicated: "
                f"{param.sharding.spec} with shape {param.shape}"
            )

    batch = Batch(
        inputs=jnp.pad(input_array[:, :-1], ((0, 0), (1, 0)), constant_values=0),
        targets=input_array,
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
    assert all(m.sharding.spec == P() for m in jax.tree.leaves(metrics)), (
        "Metrics should be replicated over axes, but found different sharding: "
        f"{[m.sharding for m in jax.tree.leaves(metrics)]}"
    )

    metrics = jax.device_get(metrics)
    assert "loss" in metrics
    assert len(metrics["loss"]) == 2, "Metrics must be a tuple."
    assert metrics["loss"][1] == input_array.size, (
        "Second metric element must be counting the batch elements, but does not fit to actual batch size: "
        f"{input_array.size} vs {metrics['loss'][1]}"
    )
    loss = metrics["loss"][0] / metrics["loss"][1]
    assert loss > 0, f"Loss must be greater zero, but is {loss}."
