import logging

import hydra
import jax
from omegaconf import DictConfig, OmegaConf

from xlstm_jax.define_hydra_schemas import register_configs
from xlstm_jax.distributed.xla_utils import set_XLA_flags

set_XLA_flags()  # Must be executed before any JAX operation.

LOGGER = logging.getLogger(__name__)

# Register Hydra configs
register_configs()


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def check_config(cfg: DictConfig):
    if jax.process_index() == 0:
        LOGGER.info(f"Config:\n{OmegaConf.to_yaml(cfg, resolve=True)}")


if __name__ == "__main__":
    check_config()
