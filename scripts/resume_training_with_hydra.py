"""Loads checkpoint and data loaders from a logging directory and continues training model."""

import hydra
from omegaconf import DictConfig

from xlstm_jax.define_hydra_schemas import register_configs
from xlstm_jax.distributed import set_XLA_flags
from xlstm_jax.resume_training import resume_training

set_XLA_flags()  # Must be executed before any JAX operation.

# Register Hydra configs
register_configs()


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def init_hydra_for_resume(cfg: DictConfig):
    """Resumes training from a checkpoint that is specified in the configuration.

    Args:
        cfg: The configuration of the experiment that is to be resumed.

    Returns:
        The final metrics of the resumed training.
    """

    final_metrics = resume_training(cfg)
    return final_metrics


if __name__ == "__main__":
    init_hydra_for_resume()
