import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import linen as nn
from jax.sharding import Mesh, PartitionSpec as P

from xlstm_jax.dataset import Batch
from xlstm_jax.models.configs import ParallelConfig
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.block import mLSTMBlockConfig
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.layer import mLSTMCellConfig, mLSTMLayerConfig
from xlstm_jax.models.xlstm_parallel.components.feedforward import FeedForwardConfig
from xlstm_jax.models.xlstm_parallel.training import get_train_step_fn, init_xlstm
from xlstm_jax.models.xlstm_parallel.xlstm_lm_model import xLSTMLMModelConfig
from xlstm_jax.utils import flatten_dict

# Define configuration
MODEL_CONFIGS = [
    xLSTMLMModelConfig(
        vocab_size=100,
        embedding_dim=128,
        num_blocks=1,
        context_length=32,
        tie_weights=False,
        add_embedding_dropout=True,
        add_post_blocks_norm=True,
        parallel=ParallelConfig(
            remat=(),
            fsdp_modules=(),
            tp_async_dense=False,
        ),
        dtype="float32",
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                proj_factor=2.0,
                conv1d_kernel_size=4,
                num_heads=4,
                dropout=0.2,
                embedding_dim=128,
                context_length=32,
                mlstm_cell=mLSTMCellConfig(
                    gate_linear_headwise=True,
                ),
            )
        ),
    ),
    xLSTMLMModelConfig(
        vocab_size=100,
        embedding_dim=128,
        num_blocks=1,
        context_length=32,
        tie_weights=False,
        add_embedding_dropout=True,
        add_post_blocks_norm=True,
        parallel=ParallelConfig(
            remat=(),
            fsdp_modules=(),
            tp_async_dense=False,
        ),
        dtype="float32",
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                layer_type="mlstm_v1",
                num_heads=4,
                embedding_dim=128,
                context_length=32,
                mlstm_cell=mLSTMCellConfig(
                    igate_bias_init_range=-10.0,
                ),
            ),
            feedforward=FeedForwardConfig(
                proj_factor=2.0,
                act_fn="gelu",
                embedding_dim=128,
                dropout=0.0,
                bias=False,
                ff_type="ffn",
                dtype="float32",
            ),
        ),
    ),
    xLSTMLMModelConfig(
        vocab_size=100,
        embedding_dim=128,
        num_blocks=1,
        context_length=32,
        tie_weights=False,
        add_embedding_dropout=True,
        add_post_blocks_norm=True,
        parallel=ParallelConfig(
            remat=(),
            fsdp_modules=(),
            tp_async_dense=False,
        ),
        dtype="bfloat16",
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                layer_type="mlstm_v1",
                num_heads=4,
                embedding_dim=128,
                context_length=32,
                mlstm_cell=mLSTMCellConfig(
                    igate_bias_init_range=-10.0,
                ),
                dtype="bfloat16",
            ),
            feedforward=FeedForwardConfig(
                proj_factor=1.0,
                act_fn="gelu",
                embedding_dim=128,
                dropout=0.0,
                bias=False,
                ff_type="ffn_gated",
                dtype="bfloat16",
            ),
        ),
    ),
]
LARGE_MODEL_CONFIGS = [
    xLSTMLMModelConfig(
        vocab_size=1024,
        embedding_dim=512,
        num_blocks=2,
        context_length=4,
        tie_weights=False,
        add_embedding_dropout=True,
        add_post_blocks_norm=True,
        parallel=ParallelConfig(
            remat=(),
            fsdp_modules=(),
            tp_async_dense=True,
        ),
        scan_blocks=True,
        dtype="float32",
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                proj_factor=2.0,
                conv1d_kernel_size=4,
                num_heads=4,
                dropout=0.2,
                embedding_dim=512,
                context_length=4,
                mlstm_cell=mLSTMCellConfig(
                    gate_linear_headwise=True,
                ),
            )
        ),
    ),
    xLSTMLMModelConfig(
        vocab_size=768,
        embedding_dim=512,
        num_blocks=1,
        context_length=4,
        tie_weights=False,
        add_embedding_dropout=True,
        add_post_blocks_norm=True,
        parallel=ParallelConfig(
            remat=(),
            fsdp_modules=(),
            tp_async_dense=False,
        ),
        dtype="float32",
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                layer_type="mlstm_v1",
                num_heads=4,
                embedding_dim=512,
                context_length=4,
                mlstm_cell=mLSTMCellConfig(
                    igate_bias_init_range=-10.0,
                ),
            ),
            feedforward=FeedForwardConfig(
                proj_factor=2.0,
                act_fn="gelu",
                embedding_dim=512,
                dropout=0.0,
                bias=False,
                ff_type="ffn",
                dtype="float32",
            ),
        ),
    ),
    xLSTMLMModelConfig(
        vocab_size=512,
        embedding_dim=768,
        num_blocks=1,
        context_length=4,
        tie_weights=False,
        add_embedding_dropout=True,
        add_post_blocks_norm=True,
        parallel=ParallelConfig(
            remat=(),
            fsdp_modules=(),
            tp_async_dense=False,
        ),
        dtype="bfloat16",
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                layer_type="mlstm_v1",
                num_heads=4,
                embedding_dim=768,
                context_length=4,
                mlstm_cell=mLSTMCellConfig(
                    igate_bias_init_range=-10.0,
                ),
                dtype="bfloat16",
            ),
            feedforward=FeedForwardConfig(
                proj_factor=4.0 / 3.0,
                act_fn="gelu",
                embedding_dim=768,
                dropout=0.0,
                bias=False,
                ff_type="ffn_gated",
                dtype="bfloat16",
            ),
        ),
    ),
]


def _create_mesh(config: xLSTMLMModelConfig, model_axis_size: int = 1):
    """Create a mesh with the given model axis size."""
    device_array = np.array(jax.devices()).reshape((-1, 1, 1, model_axis_size))
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
@pytest.mark.parametrize("gradient_accumulate_steps", [1])
@pytest.mark.parametrize("model_axis_size", [4])
def test_simple_tensor_parallel(config: xLSTMLMModelConfig, gradient_accumulate_steps: int, model_axis_size: int):
    """Test a simple forward pass with tensor parallelism."""
    if model_axis_size > pytest.num_devices:
        pytest.skip("Not enough devices for model axis size.")
    mesh = _create_mesh(config, model_axis_size=model_axis_size)
    rng = jax.random.PRNGKey(42)
    model_rng, data_rng = jax.random.split(rng)
    input_array = jax.random.randint(data_rng, shape=(32, config.context_length), minval=0, maxval=config.vocab_size)
    optimizer = optax.adamw(learning_rate=1e-3)
    config.__post_init__()
    state = init_xlstm(config=config, mesh=mesh, rng=model_rng, input_array=input_array, optimizer=optimizer)
    assert state is not None

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


@pytest.mark.parametrize("config", LARGE_MODEL_CONFIGS)
@pytest.mark.parametrize("model_axis_size", [2, 4])
def test_tensor_parallel_initialization(config: xLSTMLMModelConfig, model_axis_size: int):
    """Test that tensor parallel initialization matches data parallel initialization."""
    if model_axis_size > pytest.num_devices:
        pytest.skip("Not enough devices for model axis size.")
    config.__post_init__()
    base_mesh = _create_mesh(config, model_axis_size=1)
    rng = jax.random.PRNGKey(42)
    model_rng, data_rng = jax.random.split(rng)
    input_array = jax.random.randint(data_rng, shape=(32, config.context_length), minval=0, maxval=config.vocab_size)
    optimizer = optax.adamw(learning_rate=1e-3)
    base_state = init_xlstm(config=config, mesh=base_mesh, rng=model_rng, input_array=input_array, optimizer=optimizer)
    base_params = jax.device_get(base_state.params)
    base_params = flatten_dict(base_params)
    for key in base_params:
        if isinstance(base_params[key], nn.Partitioned):
            base_params[key] = base_params[key].value

    tp_mesh = _create_mesh(config, model_axis_size=model_axis_size)
    tp_state = init_xlstm(config=config, mesh=tp_mesh, rng=model_rng, input_array=input_array, optimizer=optimizer)
    tp_params = jax.device_get(tp_state.params)
    tp_params = flatten_dict(tp_params)
    tp_params = _map_tp_params_to_dp(tp_params, model_axis_size=model_axis_size)

    for key in base_params:
        assert key in tp_params, f"Key {key} not found in tensor parallel params."
        p_dp, p_tp = base_params[key], tp_params[key]
        assert np.prod(p_dp.shape) == np.prod(p_tp.shape), f"Shape mismatch for key {key}: {p_dp.shape} vs {p_tp.shape}"

        if np.prod(p_dp.shape) < 16:
            atol_mean, atol_std, rtol_std = 1e-1, 1e-1, 1e-1
        elif np.prod(p_dp.shape) <= 8192:
            atol_mean, atol_std, rtol_std = 2e-2, 1e-2, 1e-1
        else:
            atol_mean, atol_std, rtol_std = 1e-2, 1e-3, 1e-2
        np.testing.assert_allclose(
            p_dp.mean(),
            p_tp.mean(),
            atol=atol_mean,
            err_msg=f"Mean deviates for key: {key} / shape: {p_dp.shape}. Val DP: {p_dp}, Val TP: {p_tp}.",
        )
        np.testing.assert_allclose(
            p_dp.std(),
            p_tp.std(),
            atol=atol_std,
            rtol=rtol_std,
            err_msg=f"Std deviates for key: {key} / shape: {p_dp.shape}",
        )

    num_parameters = sum(np.prod(p.shape) for p in jax.tree.leaves(base_params))
    num_parameters_tp = sum(np.prod(p.shape) for p in jax.tree.leaves(tp_params))
    assert (
        num_parameters == num_parameters_tp
    ), f"Total number of parameters mismatch: {num_parameters} vs {num_parameters_tp}"


def _map_tp_params_to_dp(params: dict[str, jax.Array], model_axis_size: int) -> dict[str, jax.Array]:
    """Maps tensor-parallel parameters to data-parallel-only parameters."""
    for key in params:
        if isinstance(params[key], nn.Partitioned):
            params[key] = params[key].value
    keys = list(params.keys())
    for key in keys:
        new_key = key
        val = params.pop(key)
        if ".module.sharded." in new_key:
            new_key = new_key.replace(".module.sharded.", ".Dense_0.")
        if new_key.endswith(".shard_0.sharded.bias"):
            new_key = new_key.replace(".shard_0.sharded.bias", ".shard_0.bias")
        params[new_key] = val
    keys = list(params.keys())
    for key in keys:
        if key.endswith(".shard_0.sharded.kernel"):
            all_keys = [
                key.replace(".shard_0.sharded.kernel", f".shard_{i}.sharded.kernel") for i in range(model_axis_size)
            ]
            new_key = key.replace(".shard_0.sharded.kernel", ".shard_0.kernel")
            params[new_key] = np.concatenate([params.pop(k) for k in all_keys], axis=-1)
    return params
