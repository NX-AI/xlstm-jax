import os
import shutil
import sys
import traceback

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from xlstm_jax.define_hydra_schemas import register_configs
from xlstm_jax.distributed import set_XLA_flags
from xlstm_jax.distributed.xla_utils import simulate_CPU_devices
from xlstm_jax.main_train import main_train

set_XLA_flags()  # Must be executed before any JAX operation.

# Register Hydra configs
register_configs()

# Path to the configs directory.
CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "configs"))


def copy_experiment_file_to_out_folder(cfg: DictConfig) -> None:
    """
    Copy the experiment config file to the logging directory.

    Args:
        cfg: The full configuration.
    """
    if not HydraConfig.get().run.dir:
        return

    overrides = HydraConfig.get().overrides.task
    # See if experiment file is specified in the overrides.
    for override in overrides:
        if override.startswith("+experiment="):
            experiment_name = override.split("=")[1]

    os.makedirs(os.path.join(cfg.logger.log_path, "experiment_file"), exist_ok=True)
    shutil.copy(
        os.path.join(CONFIG_PATH, "experiment", f"{experiment_name}.yaml"),
        os.path.join(cfg.logger.log_path, "experiment_file", f"{experiment_name}.yaml"),
    )


@hydra.main(config_path=CONFIG_PATH, config_name="config", version_base="1.3")
def init_hydra(cfg: DictConfig):
    os.environ["JAX_PLATFORMS"] = cfg.device
    if os.environ["JAX_PLATFORMS"] == "cpu" and cfg.device_count > 1:
        simulate_CPU_devices(cfg.device_count)

    # Copy the experiment config file to the logging directory.
    copy_experiment_file_to_out_folder(cfg)

    # The following try-except block is used to circumvent a bug in Hydra. Solution from:
    # https://github.com/facebookresearch/hydra/issues/2664#issuecomment-1857695600
    try:
        main_train(cfg)
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        raise
    finally:
        sys.stdout.flush()
        sys.stderr.flush()


if __name__ == "__main__":
    init_hydra()
