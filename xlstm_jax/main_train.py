import os

import jax
from omegaconf import DictConfig, OmegaConf

from xlstm_jax.distributed.mesh_utils import initialize_mesh
from xlstm_jax.train_init_fns import init_data_iterator, init_model_config, init_parallel, init_trainer, log_info


def main_train(cfg: DictConfig):
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

    # Save resolved config to output directory
    if jax.process_index() == 0:
        output_dir = cfg.logger.log_path
        with open(os.path.join(output_dir, "resolved_config.yaml"), "w") as f:
            OmegaConf.save(cfg, f, resolve=True)

    # Start training
    log_info("Training model.")
    final_metrics = trainer.train_model(
        train_loader=data_iterator,
        val_loader=eval_data_iterator,
        num_epochs=cfg.num_epochs,
    )
    log_info(f"Final metrics: {final_metrics}")

    return final_metrics
