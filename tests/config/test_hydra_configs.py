import glob
import os
from pathlib import Path

import pytest
from hydra import compose, initialize
from hydra.errors import ConfigCompositionException
from omegaconf import DictConfig

from xlstm_jax.define_hydra_schemas import register_configs


@pytest.fixture(scope="module")
def hydra_setup():
    """Initializes Hydra and registers the schemas."""
    register_configs()
    with initialize(version_base=None, config_path="../../configs"):
        yield


def discover_config_files(config_dir: str):
    """Discovers all config files in the given directory and subdirectories. Excludes experiment files."""
    config_files = glob.glob(os.path.join(config_dir, "**", "*.yaml"), recursive=True)
    config_names = [
        os.path.relpath(file, config_dir).replace(os.sep, "/").replace(".yaml", "")
        for file in config_files
        if "experiment" not in os.path.relpath(file, config_dir).split(os.sep)
    ]
    # We require functionality that is not provided by the current release 1.2.0 of slurm_launcher but is tested here.
    # Therefore, we remove the slurm_launcher config from the tests.
    if "hydra/launcher/slurm_launcher" in config_names:
        config_names.remove("hydra/launcher/slurm_launcher")
    return config_names


def discover_experiment_files(experiment_dir: str):
    """Discovers all experiment files in the experiment directory."""
    experiment_files = [
        os.path.relpath(f.with_suffix("").as_posix(), experiment_dir) for f in Path(experiment_dir).glob("**/*.yaml")
    ]
    return experiment_files


@pytest.mark.parametrize("config_name", discover_config_files("configs"))
def test_sub_configs(hydra_setup, config_name):  # pylint: disable=unused-argument
    """Tests if all config files can be loaded without errors."""
    cfg = compose(config_name=config_name, return_hydra_config=True)
    assert isinstance(cfg, DictConfig)


@pytest.mark.parametrize("experiment_name", discover_experiment_files("configs/experiment"))
def test_experiments(hydra_setup, experiment_name):
    """Tests if all experiment files can be loaded without errors."""
    cfg = compose(config_name="config", overrides=[f"+experiment={experiment_name}"])
    assert isinstance(cfg, DictConfig)


def test_invalid_override_type(hydra_setup):  # pylint: disable=unused-argument
    """Test that an error is raised when the override type is invalid. This is only performed for one
    example for now to check whether the type checking works."""
    with pytest.raises(ConfigCompositionException) as e:
        compose(config_name="parallel/synthetic", overrides=["parallel.fsdp_axis_size=1.5"])

    assert e.value.args[0] == "Error merging override parallel.fsdp_axis_size=1.5"


def test_wrong_key_in_config(hydra_setup):
    # Test that an error is raised when a wrong key is used in the config file.
    with pytest.raises(ConfigCompositionException) as e:
        compose(config_name="parallel/synthetic", overrides=["parallel.wrong_key=1"])

    assert (
        e.value.args[0]
        == "Could not override 'parallel.wrong_key'.\nTo append to your config use +parallel.wrong_key=1"
    )


def test_wrong_comma_in_override(hydra_setup):
    # Test that an error is raised when a comma is used in the overrides when
    # a single value is expected.
    with pytest.raises(ConfigCompositionException) as e:
        compose(config_name="parallel/synthetic", overrides=["parallel.fsdp_axis_size=1,2"])

    assert e.value.args[0] == (
        "Ambiguous value for argument 'parallel.fsdp_axis_size=1,2'\n1. To use it as a list, "
        "use key=[value1,value2]\n2. To use it as string, quote the value: key='value1,value2'\n3."
        " To sweep over it, add --multirun to your command line"
    )
