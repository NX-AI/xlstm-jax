import os
import sys
import traceback

import hydra
from omegaconf import DictConfig

from xlstm_jax.define_hydra_schemas import register_configs
from xlstm_jax.distributed import set_XLA_flags
from xlstm_jax.distributed.xla_utils import simulate_CPU_devices
from xlstm_jax.main_train import main_train

set_XLA_flags()  # Must be executed before any JAX operation.

# Register Hydra configs
register_configs()


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def init_hydra(cfg: DictConfig):
    os.environ["JAX_PLATFORMS"] = cfg.device
    if os.environ["JAX_PLATFORMS"] == "cpu" and cfg.device_count > 1:
        simulate_CPU_devices(cfg.device_count)

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
