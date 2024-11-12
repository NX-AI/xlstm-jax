import os
from pathlib import Path
from typing import Any

import jax
from omegaconf import DictConfig, OmegaConf

from xlstm_jax.distributed.mesh_utils import initialize_mesh
from xlstm_jax.train_init_fns import (
    LOGGER,
    init_data_iterator,
    init_model_config,
    init_parallel,
    init_trainer,
    log_info,
)
from xlstm_jax.utils.error_logging_utils import with_error_handling


@with_error_handling(flush_output=False, logger=LOGGER)
def main_train(cfg: DictConfig, checkpoint_step: int | None = None) -> dict[str, Any]:
    """
    The main training function. This function initializes the mesh, data iterators,
      model config, and trainer and then starts training. Can be optionally started
      from a checkpoint, in which case the training state is loaded from the checkpoint
      with the supplied step index.

    In order to see error logs in our custom logger, we use the with_error_handling
      decorator.

    Args:
        cfg: The full configuration.
        checkpoint_step (optional): Step index of checkpoint to be loaded.
         Defaults to None, in which case training starts from scratch.

    Returns:
        The final metrics of the training.
    """
    # Create mesh. Needs to be done before any JAX operation due to distribute initialize.
    parallel = init_parallel(cfg=cfg)

    # Initialize device mesh
    mesh = initialize_mesh(parallel_config=parallel)
    log_info("Mesh initialized.")

    log_info(f"Devices: {jax.devices()}")

    # Compute global batch size.
    global_batch_size = cfg.batch_size_per_device * len(jax.devices())
    cfg.global_batch_size = global_batch_size

    # Create data iterator.
    log_info("Creating data iterator.")
    data_iterator, eval_data_iterator = init_data_iterator(cfg=cfg, mesh=mesh)

    # Instatiate model config.
    model_config = init_model_config(cfg=cfg, parallel=parallel)

    # Instantiate trainer.
    trainer = init_trainer(cfg=cfg, data_iterator=data_iterator, model_config=model_config, mesh=mesh)

    # If a checkpoint step is provided, load the trainer state from this checkpoint.
    if checkpoint_step is not None:
        if checkpoint_step == -1:
            log_info("Resuming training from latest checkpoint.")
        else:
            log_info(f"Resuming training from checkpoint step {checkpoint_step}.")

        loaded_step_idx = trainer.load_pretrained_model(
            checkpoint_path=Path(cfg.resume_from_folder),
            step_idx=checkpoint_step,
            train_loader=data_iterator,
            val_loader=eval_data_iterator,
        )

    # Save resolved config to output directory if not continuing from checkpoint.
    if jax.process_index() == 0:
        if checkpoint_step is None:
            resolved_config_suffix = "from_scratch"
        else:
            resolved_config_suffix = f"resumed_from_checkpoint_{loaded_step_idx}"

        output_dir = cfg.logger.log_path
        with open(os.path.join(output_dir, f"resolved_config-{resolved_config_suffix}.yaml"), "w") as f:
            OmegaConf.save(cfg, f, resolve=True)

    # Start training
    log_info("Training model.")
    train_kwargs = {}
    if cfg.get("num_train_steps", None):
        train_kwargs["num_train_steps"] = cfg.num_train_steps
    elif cfg.get("num_epochs", None):
        train_kwargs["num_epochs"] = cfg.num_epochs
    final_metrics = trainer.train_model(
        train_loader=data_iterator,
        val_loader=eval_data_iterator,
        **train_kwargs,
    )
    log_info(f"Final metrics: {final_metrics}")

    return final_metrics
