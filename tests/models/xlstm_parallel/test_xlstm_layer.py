from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

from xlstm_jax.common_types import PyTree
from xlstm_jax.distributed.mesh_utils import initialize_mesh
from xlstm_jax.models.configs import ParallelConfig
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.cell import mLSTMCellConfig
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.layer import mLSTMLayer, mLSTMLayerConfig
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.layer_v1 import mLSTMLayerV1


@pytest.mark.parametrize("layer_type", ["mlstm", "mlstm_v1"])
def test_layer_variants(layer_type: str):
    """
    Tests whether the outputs of mLSTMLayer and mLSTMLayerV1 differ for different norm types and gate soft caps.

    This is a simple test to verify that options are being used. Note that the outputs are not checked for correctness,
    as this is difficult to do without a ground-truth reference implementation.
    """
    parallel = ParallelConfig(
        data_axis_size=-1,
    )
    mesh = initialize_mesh(parallel)
    outputs = {}
    for norm_type, gate_soft_cap in product(["layernorm", "rmsnorm"], [None, 2.0, 15.0]):
        layer_config = mLSTMLayerConfig(
            embedding_dim=16,
            num_heads=2,
            context_length=32,
            gate_input="qkv",
            layer_type=layer_type,
            norm_type=norm_type,
            dtype="float32",
            mlstm_cell=mLSTMCellConfig(
                gate_linear_headwise=True,
                gate_soft_cap=gate_soft_cap,
                add_qk_norm=False,
                norm_type=norm_type,
                norm_type_v1=norm_type,
            ),
            parallel=parallel,
        )
        layer_class = mLSTMLayer if layer_type == "mlstm" else mLSTMLayerV1
        layer = layer_class(layer_config)
        x = jax.random.normal(jax.random.PRNGKey(0), (mesh.size, 32, 16), jnp.float32)
        x = x - x.mean(axis=-1, keepdims=True)

        def execute_fn(rng: jnp.ndarray, x: jnp.ndarray) -> PyTree:
            return layer.init_with_output(rng, x)

        init_model_fn = jax.jit(
            shard_map(
                execute_fn,
                mesh,
                in_specs=(P(), P()),
                out_specs=P(),
                check_rep=False,
            )
        )
        y, _ = init_model_fn(jax.random.PRNGKey(0), x)
        assert y.shape == x.shape, "Output shape should match input shape."
        y = jax.device_get(y.astype(jnp.float32))
        outputs[(norm_type, gate_soft_cap)] = y

    keys = list(outputs.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            assert not np.allclose(outputs[keys[i]], outputs[keys[j]]), (
                "Outputs for different norm types and gate soft caps should not match, but got a match between "
                f"{keys[i]} and {keys[j]}."
            )
