from typing import Any

from omegaconf import DictConfig

from xlstm_jax.main_train import main_train
from xlstm_jax.utils.error_logging_utils import with_error_handling


@with_error_handling(flush_output=True)
def resume_training(cfg: DictConfig) -> dict[str, Any]:
    """Resumes training from a checkpoint.

    Args:
        cfg: The full configuration.

    Returns:
        The final metrics of the resumed training.
    """
    # When continuing training, the parameter "resume_from_folder" must be set in the configuration.
    if cfg.resume_from_folder is None:
        raise ValueError("The parameter 'resume_from_folder' must be set in the configuration when resuming training.")

    final_metrics = main_train(
        cfg=cfg,
        checkpoint_step=cfg.checkpoint_step,
        load_dataloaders=cfg.get("load_dataloaders", True),
        load_optimizer=cfg.get("load_optimizer", True),
    )

    return final_metrics
