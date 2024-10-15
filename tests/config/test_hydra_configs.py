import glob
import os

import pytest
from hydra import compose, initialize
from hydra.errors import ConfigCompositionException
from omegaconf import DictConfig

from xlstm_jax.define_hydra_schemas import register_configs


@pytest.fixture(scope="module")
def hydra_setup():
    register_configs()
    with initialize(version_base=None, config_path="../../configs"):
        yield


def discover_config_files(config_dir: str):
    config_files = glob.glob(os.path.join(config_dir, "**", "*.yaml"), recursive=True)
    config_names = [
        os.path.relpath(file, config_dir).replace(os.sep, "/").replace(".yaml", "")
        for file in config_files
        if "experiment" not in os.path.relpath(file, config_dir).split(os.sep)
    ]
    return config_names


@pytest.mark.parametrize("config_name", discover_config_files("configs"))
def test_sub_configs(hydra_setup, config_name):
    # Checkpointing
    cfg = compose(config_name=config_name, return_hydra_config=True)
    assert isinstance(cfg, DictConfig)


def test_invalid_override_type(hydra_setup):
    # Test that an error is raised when the override type is invalid. This is only performed for one
    # example for now to check whether the type checking works.
    with pytest.raises(ConfigCompositionException) as e:
        compose(config_name="parallel/synthetic", overrides=["parallel.fsdp_axis_size=1.5"])

    assert e.value.args[0] == "Error merging override parallel.fsdp_axis_size=1.5"
