"""Runs tests to see whether setting of XLA flags in previous Hydra tests makes mesh initialization break."""

import pytest

from xlstm_jax.distributed.mesh_utils import initialize_mesh
from xlstm_jax.models.configs import ParallelConfig


@pytest.mark.skipif(pytest.num_devices < 2, reason="Test requires at least 2 devices.")
def test_xla_flag_mesh_init():
    """Just initialize a parallel mesh to see if mesh initialization works."""
    parallel_config = ParallelConfig(fsdp_axis_size=1, pipeline_axis_size=1, model_axis_size=2, data_axis_size=-1)
    _ = initialize_mesh(parallel_config=parallel_config)
