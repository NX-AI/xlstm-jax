from typing import Any

from omegaconf import DictConfig

from xlstm_jax.main_train import main_train
from xlstm_jax.utils.error_logging_utils import with_error_handling


@with_error_handling(flush_output=True)
def start_training(cfg: DictConfig) -> dict[str, Any]:
    """Starts training from scratch.
       Decorated with error catching decorator.

    Args:
        cfg: The full configuration.

    Returns:
        The final metrics of the resumed training.
    """
    final_metrics = main_train(cfg=cfg)

    return final_metrics
