import jax
import jax.numpy as jnp
import pytest

try:
    from xlstm_jax.kernels.mlstm_chunkwise.max_triton_fwbw_v3.triton_fwbw import mlstm_chunkwise_max_triton
    from xlstm_jax.kernels.mlstm_chunkwise.max_triton_fwbw_v3noslice.triton_fwbw import (
        mlstm_chunkwise_max_triton as mlstm_chunkwise_max_triton_noslice,
    )
except ImportError:
    mlstm_chunkwise_max_triton = None
    mlstm_chunkwise_max_triton_noslice = None


@pytest.mark.skipif(not pytest.triton_available, reason="Triton is not available.")
@pytest.mark.parametrize("mlstm_kernel", [mlstm_chunkwise_max_triton, mlstm_chunkwise_max_triton_noslice])
def test_mlstm_chunkwise_state_passing(
    default_qkvif: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    mlstm_state_passing_test: callable,
    mlstm_kernel: callable,
):
    """Compare single forward vs chunked one with states passed between steps."""
    # Repeat the inputs to have longer sequence length.
    default_qkvif = jax.tree.map(lambda x: jnp.repeat(x, 2, axis=2), default_qkvif)
    mlstm_state_passing_test(mlstm_kernel, *default_qkvif, num_chunks=4, rtol=1e-5, atol=1e-5)
