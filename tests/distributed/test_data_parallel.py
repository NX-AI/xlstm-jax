from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P

from xlstm_jax.distributed import fold_rng_over_axis
from xlstm_jax.distributed.data_parallel import gather_array_with_mean_grads


def _create_mesh(
    fsdp_axis_size: int = 1,
    data_axis_name: str = "dp",
    fsdp_axis_name: str = "fsdp",
    pipeline_axis_name: str = "pp",
    model_axis_name: str = "tp",
) -> Mesh:
    device_array = np.array(jax.devices()).reshape(-1, fsdp_axis_size, 1, 1)
    return Mesh(
        device_array,
        (
            data_axis_name,
            fsdp_axis_name,
            pipeline_axis_name,
            model_axis_name,
        ),
    )


def test_gather_array_with_mean_grads_scatter_dtype():
    """
    Test gather_dtype and grad_scatter_dtype in gather_array_with_mean_grads.

    Check that gathering arrays in float32 and then casting them to bfloat16 gives the same result as gathering them
    directly in bfloat16 and only casting the gradients up. Additionally, check that gradients are different when
    scattering them in different dtypes.
    """
    mesh = _create_mesh(fsdp_axis_size=pytest.num_devices, fsdp_axis_name="fsdp")
    batch_size = 256
    hidden_dim_per_device = 128
    output_dim = 64

    def _test_fn(rng: jax.Array, gather_dtype: str, grad_scatter_dtype: str):
        rng = fold_rng_over_axis(rng, "fsdp")
        inp_rng, target_rng, weight_rng = jax.random.split(rng, 3)
        inp = jax.random.normal(inp_rng, (batch_size, hidden_dim_per_device * pytest.num_devices), dtype=jnp.float32)
        targets = jax.random.normal(target_rng, (batch_size, output_dim), dtype=jnp.float32)
        weight = jax.random.normal(weight_rng, (hidden_dim_per_device, output_dim), dtype=jnp.float32)

        def _loss_fn(params):
            params = gather_array_with_mean_grads(
                params, axis=0, axis_name="fsdp", gather_dtype=gather_dtype, grad_scatter_dtype=grad_scatter_dtype
            )
            params = params.astype(jnp.bfloat16)
            out = inp @ params
            loss = ((out - targets) ** 2).mean()
            return loss

        grads = jax.grad(_loss_fn)(weight)
        return grads

    rng = jax.random.PRNGKey(0)
    all_grads = {}
    for gather_dtype in ["float32", "bfloat16"]:
        for grad_scatter_dtype in ["float32", "bfloat16", "float16"]:
            test_fn = shard_map(
                partial(_test_fn, gather_dtype=gather_dtype, grad_scatter_dtype=grad_scatter_dtype),
                mesh,
                in_specs=P(),
                out_specs=P("fsdp"),
            )
            grads = test_fn(rng)
            grads = jax.device_get(grads)
            all_grads[(gather_dtype, grad_scatter_dtype)] = grads

    np.testing.assert_allclose(
        all_grads[("float32", "float32")],
        all_grads[("bfloat16", "float32")],
        err_msg="Gathering the array in float32 and then casting it to bfloat16 should give the same result as "
        "gathering it directly in bfloat16 and only casting the gradients up.",
    )
    np.testing.assert_allclose(
        all_grads[("float32", "bfloat16")],
        all_grads[("bfloat16", "bfloat16")],
        err_msg="Gathering the array in float32 and then casting it to bfloat16 should give the same result as "
        "gathering it directly in bfloat16 and only casting the gradients up.",
    )

    # For single device, there is no difference as nothing is scattered.
    if pytest.num_devices > 1:
        assert not np.all(all_grads[("float32", "float32")] == all_grads[("float32", "bfloat16")]), (
            "Scattering the gradients in different dtypes should give different results, but resulted in same "
            "for float32 and bfloat16."
        )
        assert not np.all(all_grads[("float32", "float32")] == all_grads[("float32", "float16")]), (
            "Scattering the gradients in different dtypes should give different results, but resulted in same "
            "for float32 and float16."
        )
