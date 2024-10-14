from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

from xlstm_jax.distributed.mesh_utils import initialize_mesh
from xlstm_jax.models.configs import ParallelConfig
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.block import mLSTMBlock, mLSTMBlockConfig
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.cell import mLSTMCellConfig
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.layer import mLSTMLayerConfig
from xlstm_jax.trainer.base.param_utils import flatten_dict

PyTree = Any


@pytest.mark.parametrize("add_post_norm", [True, False])
@pytest.mark.parametrize("add_qk_norm", [True, False])
@pytest.mark.parametrize("norm_type", ["layernorm", "rmsnorm"])
@pytest.mark.parametrize("layer_type", ["mlstm", "mlstm_v1"])
def test_norm_variants(add_post_norm: bool, add_qk_norm: bool, norm_type: str, layer_type: str):
    parallel = ParallelConfig(
        data_axis_size=-1,
    )
    mesh = initialize_mesh(parallel)
    block_config = mLSTMBlockConfig(
        mlstm=mLSTMLayerConfig(
            embedding_dim=16,
            num_heads=2,
            context_length=32,
            gate_input="qkv",
            layer_type=layer_type,
            norm_type=norm_type,
            dtype=jnp.float32,
            mlstm_cell=mLSTMCellConfig(
                gate_linear_headwise=True,
                gate_soft_cap=30.0,
                add_qk_norm=add_qk_norm,
            ),
        ),
        add_post_norm=add_post_norm,
        parallel=parallel,
        _num_blocks=12,
        _block_idx=0,
    )
    block = mLSTMBlock(block_config)
    x = jax.random.normal(jax.random.PRNGKey(0), (mesh.size, 32, 16), jnp.float32)
    x = x - x.mean(axis=-1, keepdims=True)

    def execute_fn(rng: jnp.ndarray, x: jnp.ndarray) -> PyTree:
        return block.init_with_output(rng, x)

    init_model_fn = jax.jit(
        shard_map(
            execute_fn,
            mesh,
            in_specs=(P(), P()),
            out_specs=P(),
            check_rep=False,
        )
    )
    y, params = init_model_fn(jax.random.PRNGKey(0), x)
    assert y.shape == x.shape, "Output shape should match input shape."
    y = jax.device_get(y.astype(jnp.float32))
    params = jax.device_get(params)
    params = flatten_dict(params)

    if add_qk_norm:
        assert any(key.endswith("q_norm.scale") for key in params), "Q norm scale should be present."
        assert any(key.endswith("k_norm.scale") for key in params), "K norm scale should be present."

    if add_post_norm:
        assert any(key.endswith("post_norm.scale") for key in params), "Post norm scale should be present."
        if norm_type == "layernorm":
            np.testing.assert_allclose(
                y.mean(axis=-1), 0.0, atol=1e-5, err_msg="LayerNorm applied to post norm should have zero mean."
            )
        else:
            assert np.abs(y.mean(axis=-1)).max() > 1e-3, "RMSNorm applied to post norm should not have zero mean."

    y_shifted, _ = init_model_fn(jax.random.PRNGKey(0), x + 1)
    y_shifted = jax.device_get(y_shifted.astype(jnp.float32))
    y_shifted = y_shifted - 1
    if norm_type == "layernorm":
        np.testing.assert_allclose(
            y, y_shifted, atol=1e-5, rtol=1e-5, err_msg="LayerNorm should be invariant to shifts."
        )
    else:
        assert np.abs(y - y_shifted).max() > 0.1, "RMSNorm should not be invariant to shifts."
