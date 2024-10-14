from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import linen as nn
from jax.sharding import Mesh, PartitionSpec as P

from xlstm_jax.models.configs import ParallelConfig
from xlstm_jax.models.xlstm_clean.xlstm_lm_model import xLSTMLMModel as xLSTMLMModelClean
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.block import mLSTMBlockConfig
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.cell import mLSTMCellConfig
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.layer import mLSTMLayerConfig
from xlstm_jax.models.xlstm_parallel.components.feedforward import FeedForwardConfig
from xlstm_jax.models.xlstm_parallel.components.init import create_common_init_fn
from xlstm_jax.models.xlstm_parallel.training import init_xlstm
from xlstm_jax.models.xlstm_parallel.xlstm_lm_model import xLSTMLMModelConfig
from xlstm_jax.trainer.base.param_utils import flatten_dict

PyTree = Any

# Define configuration
PYTORCH_JAX_MODEL_CONFIGS = [
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
        scan_blocks=False,
        dtype=jnp.float32,
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                proj_factor=2.0,
                conv1d_kernel_size=4,
                num_heads=4,
                dropout=0.2,
                embedding_dim=512,
                context_length=4,
            )
        ),
    ),
    xLSTMLMModelConfig(
        vocab_size=768,
        embedding_dim=1536,
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
        scan_blocks=False,
        dtype=jnp.float32,
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                proj_factor=3.0,
                conv1d_kernel_size=6,
                num_heads=8,
                dropout=0.0,
                embedding_dim=1536,
                context_length=4,
            )
        ),
    ),
]
INIT_MODEL_CONFIG_FNS = [
    lambda output_init_fn, init_distribution: xLSTMLMModelConfig(
        vocab_size=512,
        embedding_dim=256,
        num_blocks=3,
        context_length=4,
        norm_type="layernorm",
        scan_blocks=False,
        parallel=ParallelConfig(),
        dtype=jnp.bfloat16,
        init_distribution_embed=init_distribution,
        init_distribution_out=init_distribution,
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                proj_factor=2.0,
                num_heads=4,
                embedding_dim=256,
                context_length=4,
                init_distribution=init_distribution,
                output_init_fn=output_init_fn,
                mlstm_cell=mLSTMCellConfig(
                    gate_linear_headwise=True,
                ),
                dtype=jnp.bfloat16,
            )
        ),
    ),
    lambda output_init_fn, init_distribution: xLSTMLMModelConfig(
        vocab_size=512,
        embedding_dim=384,
        logits_soft_cap=None,
        num_blocks=2,
        context_length=4,
        norm_type="rmsnorm",
        scan_blocks=True,
        parallel=ParallelConfig(),
        dtype=jnp.bfloat16,
        init_distribution_embed=init_distribution,
        init_distribution_out=init_distribution,
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                layer_type="mlstm_v1",
                num_heads=4,
                embedding_dim=256,
                context_length=4,
                init_distribution=init_distribution,
                output_init_fn=output_init_fn,
                mlstm_cell=mLSTMCellConfig(
                    igate_bias_init_range=-10.0,
                ),
                dtype=jnp.bfloat16,
            ),
            feedforward=FeedForwardConfig(
                proj_factor=2.0,
                act_fn="gelu",
                embedding_dim=256,
                dropout=0.0,
                bias=False,
                ff_type="ffn",
                init_distribution=init_distribution,
                output_init_fn=output_init_fn,
                dtype=jnp.bfloat16,
            ),
        ),
    ),
]


def _create_mesh(config: xLSTMLMModelConfig, fsdp_axis_size: int = 1) -> Mesh:
    """Create a mesh with the given FSDP configuration."""
    device_array = np.array(jax.devices()).reshape(-1, fsdp_axis_size, 1, 1)
    return Mesh(
        device_array,
        (
            config.parallel.data_axis_name,
            config.parallel.fsdp_axis_name,
            config.parallel.pipeline_axis_name,
            config.parallel.model_axis_name,
        ),
    )


@pytest.mark.parametrize("config", PYTORCH_JAX_MODEL_CONFIGS)
def test_data_parallel_initialization(config: xLSTMLMModelConfig):
    """Test that DP initialization gives same distributions as xlstm-clean single device."""
    mesh = _create_mesh(config)
    rng = jax.random.PRNGKey(42)
    model_rng, data_rng = jax.random.split(rng)
    input_array = jax.random.randint(data_rng, shape=(32, config.context_length), minval=0, maxval=config.vocab_size)
    optimizer = optax.adamw(learning_rate=1e-3)
    state = init_xlstm(config=config, mesh=mesh, rng=model_rng, input_array=input_array, optimizer=optimizer)
    assert state is not None
    assert all([p.sharding.spec == P() for p in jax.tree.leaves(state.params)]), (
        "Parameters should be replicated over axes, but found different sharding: "
        f"{[p.sharding for p in jax.tree.leaves(state.params)]}"
    )
    params = jax.device_get(state.params)
    params = flatten_dict(params)

    single_device_model = xLSTMLMModelClean(config)
    single_device_params = single_device_model.init(model_rng, input_array, train=False)["params"]
    single_device_params = jax.device_get(single_device_params)
    single_device_params = flatten_dict(single_device_params)
    params = _map_multi_device_to_single(params)

    for key in params:
        assert key in single_device_params, f"Key {key} not found in single device params."
        p_md, p_sd = params[key], single_device_params[key]
        assert np.prod(p_md.shape) == np.prod(p_sd.shape), f"Shape mismatch for key {key}: {p_md.shape} vs {p_sd.shape}"

        if np.prod(p_md.shape) < 16:
            atol_mean, atol_std = 1e-1, 1e-1
        else:
            atol_mean, atol_std = 1e-2, 1e-2
        np.testing.assert_allclose(
            p_md.mean(), p_sd.mean(), atol=atol_mean, err_msg=f"Mean deviates for key: {key} / shape: {p_md.shape}"
        )
        np.testing.assert_allclose(
            p_md.std(),
            p_sd.std(),
            atol=atol_std,
            rtol=0.1,
            err_msg=f"Std deviates for key: {key} / shape: {p_md.shape}",
        )


def _map_multi_device_to_single(params: dict[str, jax.Array]) -> dict[str, jax.Array]:
    """Maps multi-device parameters to single-device xlstm-clean parameters."""
    for key in params:
        if isinstance(params[key], nn.Partitioned):
            params[key] = params[key].value
    params["xlstm_block_stack.post_blocks_norm.scale"] = params.pop("lm_head.out_norm.scale")
    params["lm_head.kernel"] = params.pop("lm_head.out_dense.kernel")
    keys = list(params.keys())
    for key in keys:
        new_key = key
        val = params.pop(key)
        if ".sharded." in new_key:
            new_key = new_key.replace(".sharded.", ".")
        if ".shard_0." in new_key:
            new_key = new_key.replace(".shard_0.", ".")
        if ".inner_layer." in new_key:
            new_key = new_key.replace(".inner_layer.", ".")
        if ".Dense_0." in new_key:
            new_key = new_key.replace(".Dense_0.", ".")
        params[new_key] = val
    keys = list(params.keys())
    for key in keys:
        if key.endswith("proj_up_mlstm.kernel"):
            other_key = key.replace("proj_up_mlstm.kernel", "proj_up_z.kernel")
            new_key = key.replace("proj_up_mlstm.kernel", "proj_up.kernel")
            params[new_key] = np.concatenate([params.pop(key), params.pop(other_key)], axis=-1)
    return params


@pytest.mark.parametrize("config_fn", INIT_MODEL_CONFIG_FNS)
@pytest.mark.parametrize(
    "init_distribution, output_init_fn", [("normal", "wang"), ("uniform", "small"), ("truncated_normal", "wang2")]
)
def test_init_dist_and_fn(config_fn: callable, init_distribution: str, output_init_fn: str):
    """Test that different initialization distributions and functions are correctly applied."""
    config = config_fn(output_init_fn, init_distribution)
    mesh = _create_mesh(config)
    rng = jax.random.PRNGKey(42)
    model_rng, data_rng = jax.random.split(rng)
    input_array = jax.random.randint(data_rng, shape=(8, config.context_length), minval=0, maxval=config.vocab_size)
    optimizer = optax.adamw(learning_rate=1e-3)
    state = init_xlstm(config=config, mesh=mesh, rng=model_rng, input_array=input_array, optimizer=optimizer)
    assert state is not None
    assert all([p.sharding.spec == P() for p in jax.tree.leaves(state.params)]), (
        "Parameters should be replicated over axes, but found different sharding: "
        f"{[p.sharding for p in jax.tree.leaves(state.params)]}"
    )
    params = jax.device_get(state.params)
    params = flatten_dict(params)

    for key in params:
        if not (key.endswith("kernel") or key.endswith("embedding")):
            # Skip non-kernel parameters.
            continue
        p = params[key]
        if isinstance(p, nn.Partitioned):
            p = p.value
        if np.prod(p.shape) <= 4096:
            # Skip small parameters.
            continue
        p_max = p.max()
        p_min = p.min()
        p_std = p.std()
        p_mean = p.mean()

        # Verify zero mean.
        np.testing.assert_allclose(
            p_mean, 0.0, atol=1e-2, err_msg=f"Mean deviates from zero for key: {key} / shape: {p.shape}"
        )

        # Verify sample distribution.
        p_std_tol = p_std * 1.25  # Allow for some deviation from the expected standard deviation.
        exp_uniform_max = np.sqrt(3.0) * p_std_tol
        if init_distribution == "uniform":
            assert p_max <= exp_uniform_max, f"[Uniform] Max value too high for key: {key} / shape: {p.shape}"
            assert -exp_uniform_max <= p_min, f"[Uniform] Min value too low for key: {key} / shape: {p.shape}"
        elif init_distribution == "truncated_normal":
            assert p_max <= 2.0 * p_std_tol, f"[Truncated Normal] Max value too high for key: {key} / shape: {p.shape}"
            assert -2.0 * p_std_tol <= p_min, f"[Truncated Normal] Min value too low for key: {key} / shape: {p.shape}"
        elif init_distribution == "normal":
            assert 2.0 * p_std <= p_max, f"[Normal] Max value too low for key: {key} / shape: {p.shape}"
            assert p_min <= -2.0 * p_std, f"[Normal] Min value too high for key: {key} / shape: {p.shape}"

        # Verify output initialization function.
        if key.endswith("proj_down.kernel") or key.endswith("proj_down.Dense_0.kernel"):
            new_param = create_common_init_fn(
                output_init_fn, config.embedding_dim, config.num_blocks, init_distribution
            )(jax.random.PRNGKey(0), p.shape, p.dtype)
            new_std = new_param.std()
            np.testing.assert_allclose(
                p_std,
                new_std,
                rtol=1e-2,
                err_msg=f"[{output_init_fn} / {init_distribution}] Std deviates for key: {key} / shape: {p.shape}",
            )
