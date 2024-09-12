import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
from distributed.utils import simulate_CPU_devices

NUM_DEVICES = 8
simulate_CPU_devices(NUM_DEVICES)

from typing import Any

from distributed.single_gpu import Batch
from model_parallel.blocks.mlstm.block import mLSTMBlockConfig
from model_parallel.blocks.mlstm.layer import mLSTMLayerConfig
from model_parallel.training import get_train_step_fn, init_xlstm
from model_parallel.utils import ParallelConfig
from model_parallel.xlstm_lm_model import xLSTMLMModelConfig

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax.sharding import Mesh, PartitionSpec as P

PyTree = Any

# Define configuration
MODEL_CONFIGS = [
    xLSTMLMModelConfig(
        vocab_size=100,
        embedding_dim=128,
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
                embedding_dim=128,
                context_length=128,
            )
        ),
    )
]


def _create_mesh(config: xLSTMLMModelConfig, model_axis_size: int = 1):
    device_array = np.array(jax.devices()).reshape(-1, 1, model_axis_size)
    return Mesh(
        device_array,
        (config.parallel.data_axis_name, config.parallel.pipeline_axis_name, config.parallel.model_axis_name),
    )


@pytest.mark.parametrize("config", MODEL_CONFIGS)
@pytest.mark.parametrize("gradient_accumulate_steps", [1])
@pytest.mark.parametrize("model_axis_size", [1, 2, 4])
def test_simple_data_parallel(config: xLSTMLMModelConfig, gradient_accumulate_steps: int, model_axis_size: int):
    mesh = _create_mesh(config, model_axis_size=model_axis_size)
    rng = jax.random.PRNGKey(42)
    model_rng, data_rng = jax.random.split(rng)
    input_array = jax.random.randint(data_rng, shape=(32, config.context_length), minval=0, maxval=config.vocab_size)
    optimizer = optax.adamw(learning_rate=1e-3)
    config.__post_init__()
    state = init_xlstm(config=config, mesh=mesh, rng=model_rng, input_array=input_array, optimizer=optimizer)
    assert state is not None
    # assert all([p.sharding.spec == P() for p in jax.tree.leaves(state.params)]), f"Parameters should be replicated over axes, but found different sharding: {[p.sharding for p in jax.tree.leaves(state.params)]}"

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
    # assert all([p.sharding.spec == P() for p in jax.tree.leaves(state.params)]), f"Parameters should be replicated over axes, but found different sharding: {[p.sharding for p in jax.tree.leaves(state.params)]}"
    assert all(
        [m.sharding.spec == P() for m in jax.tree.leaves(metrics)]
    ), f"Metrics should be replicated over axes, but found different sharding: {[m.sharding for m in jax.tree.leaves(metrics)]}"

    metrics = jax.device_get(metrics)
    assert "loss" in metrics
    assert len(metrics["loss"]) == 2, f"Metrics must be a tuple."
    assert (
        metrics["loss"][1] == input_array.size
    ), f"Second metric element must be counting the batch elements, but does not fit to actual batch size: {input_array.size} vs {metrics['loss'][1]}"
    loss = metrics["loss"][0] / metrics["loss"][1]
    assert loss > 0, f"Loss must be greater zero, but is {loss}."
