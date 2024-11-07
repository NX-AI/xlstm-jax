"""Test continuation of training run with Hydra."""

from pathlib import Path

import pytest
from hydra import compose, initialize
from omegaconf.omegaconf import open_dict

from xlstm_jax.define_hydra_schemas import register_configs
from xlstm_jax.main_train import main_train
from xlstm_jax.resume_training import resume_training


def test_hydra_trainer_continuation(tmp_path: Path):
    """Sets up the config via Hydra and calls regular main_train function."""

    register_configs()

    context_length = 32
    batch_size_per_device = 1
    global_batch_size = batch_size_per_device * pytest.num_devices
    with initialize(version_base="1.3", config_path="../../configs"):
        cfg = compose(
            config_name="config",
            overrides=[
                "+experiment=tiny_experiment_for_unit_testing",
                f"log_path={tmp_path}",
                f"data.global_batch_size={global_batch_size}",
                f"data.max_target_length={context_length}",
            ],
            return_hydra_config=True,
        )

        # Call main train function. We need to set the output_dir, log_path and cmd_logging_name
        # manually because Hydra does not fill the hydra configuration when composing the config
        # compared to using the @hydra.main decorator. Is only relevant for the unit test.
        cfg.logger.cmd_logging_name = "unit_test"
        cfg.hydra.runtime.output_dir = tmp_path
        cfg.logger.log_path = tmp_path
        _ = main_train(cfg=cfg)

        # Resume training and run for another 10 steps (to a total of 20 steps).
        cfg.num_train_steps = 20
        # Set folder to the output directory of the first run.
        with open_dict(cfg):
            cfg.resume_from_folder = tmp_path
            cfg.checkpoint_step = -1
        final_metrics_resumed = resume_training(cfg=cfg)

    # Start training from scratch and run for 20 steps.
    with initialize(version_base="1.3", config_path="../../configs"):
        cfg = compose(
            config_name="config",
            overrides=[
                "+experiment=tiny_experiment_for_unit_testing",
                f"log_path={tmp_path}",
                f"data.global_batch_size={global_batch_size}",
                f"data.max_target_length={context_length}",
                "num_train_steps=20",
            ],
            return_hydra_config=True,
        )

        cfg.logger.cmd_logging_name = "unit_test"
        cfg.hydra.runtime.output_dir = tmp_path
        cfg.logger.log_path = tmp_path
        final_metrics_20_steps = main_train(cfg=cfg)

    assert final_metrics_20_steps["val_epoch_1"]["accuracy"] == final_metrics_resumed["val_epoch_1"]["accuracy"]
    assert final_metrics_20_steps["val_epoch_1"]["loss"] == final_metrics_resumed["val_epoch_1"]["loss"]
    assert final_metrics_20_steps["val_epoch_1"]["perplexity"] == final_metrics_resumed["val_epoch_1"]["perplexity"]
