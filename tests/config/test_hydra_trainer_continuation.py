"""Test continuation of training run with Hydra."""

import argparse
import os
from pathlib import Path

from scripts.get_cli_command_to_resume_training import get_cli_command

import pytest
from hydra import compose, initialize
from omegaconf import OmegaConf

from xlstm_jax.define_hydra_schemas import register_configs
from xlstm_jax.resume_training import resume_training
from xlstm_jax.start_training import start_training


def test_get_cli_command_to_resume_training(tmp_path: Path):
    """Tests syntax of the function that returns the CLI command to resume
    training."""

    register_configs()

    context_length = 32
    batch_size_per_device = 1
    global_batch_size = batch_size_per_device * pytest.num_devices

    overrides = [
        "+experiment=tiny_experiment_for_unit_testing",
        f"log_path={tmp_path}",
        f"data.global_batch_size={global_batch_size}",
        f"data.max_target_length={context_length}",
    ]

    # Save overrides in tmp_folder at same location where hydra would save them.
    os.makedirs(tmp_path / ".hydra", exist_ok=True)
    with open(tmp_path / ".hydra" / "overrides.yaml", "w") as f:
        OmegaConf.save(overrides, f)

    # Test get_cli_command_to_resume_training
    args = argparse.Namespace(
        resume_from_folder=tmp_path, checkpoint_step=-1, new_overrides="data.global_batch_size=2", use_slurm=False
    )

    # Make sure that the command is correct.
    command = get_cli_command(args)
    assert (
        command
        == f"PYTHONPATH=. python scripts/resume_training_with_hydra.py  +experiment=tiny_experiment_for_unit_testing log_path={tmp_path} data.global_batch_size={global_batch_size} data.max_target_length={context_length} +resume_from_folder={tmp_path} +checkpoint_step=-1 data.global_batch_size=2"  # noqa: E501
    )

    # Test get_cli_command_to_resume_training
    args = argparse.Namespace(
        resume_from_folder=tmp_path, checkpoint_step=95000, new_overrides="lr=3e-5", use_slurm=True
    )
    command = get_cli_command(args)
    assert (
        command
        == f"PYTHONPATH=. python scripts/resume_training_with_hydra.py --multirun hydra/launcher=slurm_launcher +experiment=tiny_experiment_for_unit_testing log_path={tmp_path} data.global_batch_size={global_batch_size} data.max_target_length={context_length} +resume_from_folder={tmp_path} +checkpoint_step=95000 lr=3e-5"  # noqa: E501
    )


def test_training_continuation_with_cli_command(tmp_path: Path):
    """Sets up the config via Hydra and calls regular main_train function. Then,
    get the CLI command to resume training and run the training with the provided
    overrides."""

    # Start short training from scratch.
    register_configs()

    context_length = 32
    batch_size_per_device = 1
    global_batch_size = batch_size_per_device * pytest.num_devices

    overrides = [
        "+experiment=tiny_experiment_for_unit_testing",
        f"log_path={tmp_path}",
        f"data.global_batch_size={global_batch_size}",
        f"data.max_target_length={context_length}",
        "logger.cmd_logging_name=unit_test",
        "num_train_steps=10",
    ]

    # Save overrides in tmp_folder at same location where hydra would save them.
    os.makedirs(tmp_path / ".hydra", exist_ok=True)
    with open(tmp_path / ".hydra" / "overrides.yaml", "w") as f:
        OmegaConf.save(overrides, f)

    with initialize(version_base="1.3", config_path="../../configs"):
        cfg = compose(
            config_name="config",
            overrides=overrides,
            return_hydra_config=True,
        )

        # Call main train function. We need to set the output_dir, log_path and cmd_logging_name
        # manually because Hydra does not fill the hydra configuration when composing the config
        # compared to using the @hydra.main decorator. Is only relevant for the unit test.
        cfg.hydra.runtime.output_dir = tmp_path
        cfg.logger.log_path = tmp_path
        final_metrics_from_scratch = start_training(cfg=cfg)

    # Resume training for the first time.
    # =================================================================================================
    # Get the CLI command to resume training.
    args = argparse.Namespace(
        resume_from_folder=tmp_path, checkpoint_step=-1, new_overrides="num_train_steps=20 lr=3e-5", use_slurm=False
    )
    command = get_cli_command(args)

    # Separate the command string into a list of arguments, ignoring the first elements, which
    # are no overrides.
    new_overrides = command.split(" ")[4:]

    # Save new overrides in tmp_folder at same location where hydra would save them.
    os.makedirs(tmp_path / "continued" / ".hydra", exist_ok=True)
    with open(tmp_path / "continued" / ".hydra" / "overrides.yaml", "w") as f:
        OmegaConf.save(overrides, f)

    # Resume training with the new overrides. `resume_from_folder` and `checkpoint_step` are
    # now in the new_overrides.
    with initialize(version_base="1.3", config_path="../../configs"):
        cfg = compose(
            config_name="config",
            overrides=new_overrides,
            return_hydra_config=True,
        )

        # Call main train function. We need to set the output_dir, log_path and cmd_logging_name
        # manually because Hydra does not fill the hydra configuration when composing the config
        # compared to using the @hydra.main decorator. Is only relevant for the unit test.
        cfg.hydra.runtime.output_dir = tmp_path / "continued"
        cfg.logger.log_path = tmp_path / "continued"
        final_metrics_resumed = resume_training(cfg=cfg)

    # Resume training for a second time.
    # =================================================================================================
    # Get the CLI command to resume training.
    args = argparse.Namespace(
        resume_from_folder=tmp_path / "continued",
        checkpoint_step=-1,
        new_overrides="num_train_steps=30 lr=3e-5",
        use_slurm=False,
    )
    command = get_cli_command(args)

    # Separate the command string into a list of arguments, ignoring the first elements, which
    # are no overrides.
    new_overrides = command.split(" ")[4:]

    # Resume training with the new overrides. `resume_from_folder` and `checkpoint_step` are
    # now in the new_overrides.
    with initialize(version_base="1.3", config_path="../../configs"):
        cfg = compose(
            config_name="config",
            overrides=new_overrides,
            return_hydra_config=True,
        )

        # Call main train function. We need to set the output_dir, log_path and cmd_logging_name
        # manually because Hydra does not fill the hydra configuration when composing the config
        # compared to using the @hydra.main decorator. Is only relevant for the unit test.
        cfg.hydra.runtime.output_dir = tmp_path / "continued_2nd_time"
        cfg.logger.log_path = tmp_path / "continued_2nd_time"
        final_metrics_resumed_twice = resume_training(cfg=cfg)

    # Assert that loss and perplexity are lower after resuming training.
    for key in ["loss", "perplexity"]:
        assert (
            final_metrics_resumed_twice["val_epoch_3"][key]
            < final_metrics_resumed["val_epoch_2"][key]
            < final_metrics_from_scratch["val_epoch_1"][key]
        ), f"{key} did not give expected reduction in validation loss."

    # Start training from scratch and run for 20 steps.
    overrides.remove("num_train_steps=10")
    overrides.append("num_train_steps=20")
    with initialize(version_base="1.3", config_path="../../configs"):
        cfg = compose(
            config_name="config",
            overrides=overrides,
            return_hydra_config=True,
        )

        cfg.hydra.runtime.output_dir = tmp_path
        cfg.logger.log_path = tmp_path
        final_metrics_20_steps = start_training(cfg=cfg)

    # Assert that loss and perplexity are the same after running for 20 steps as after resuming training.
    for key in ["accuracy", "loss", "perplexity"]:
        assert final_metrics_20_steps["val_epoch_2"][key] == final_metrics_resumed["val_epoch_2"][key]
