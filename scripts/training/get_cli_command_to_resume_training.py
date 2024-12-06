import argparse
import os
from typing import Any

import yaml


def get_cli_command(args: Any) -> str:
    """Creates a CLI command to resume training from a checkpoint with Hydra.

    Args:
        Parsed command line arguments. They contain the fields:

            - 'resume_from_folder': The run dir of the experiment that is to be resumed
            - 'checkpoint_step': The step counter of the checkpoint that you want to use to resume training.
                Defaults to -1, which means that the latest checkpoint will be loaded.
            - 'use_slurm': Whether to use SLURM for the continuation runs. Only required if the original run was
                executed with SLURM via the CLI override ``--multirun hydra/launcher=slurm_launcher`` and not via
                experiment file. Otherwise, slurm is used anyway.
            - 'new_overrides': String that contains new overrides for this experiment. Example for more training
                steps, a different learning rate and a different logging frequency:
                ``num_train_steps=20000 lr=0.0001 logger.log_every_n_steps=10``

    Returns:
        The CLI command to resume training from a checkpoint.
    """
    # Get the overrides of the run that is to be resumed from its run_dir.
    with open(os.path.join(args.resume_from_folder, ".hydra/overrides.yaml")) as f:
        old_overrides_list = yaml.safe_load(f)

    # If the old overrides contain `checkpoint_step` or `resume_from_folder`,
    # remove them from the list since they originate from the old resume command.
    filtered_overrides = [
        override
        for override in old_overrides_list
        if ("checkpoint_step" not in override and "resume_from_folder" not in override)
    ]

    # Join the remaining overrides to a string that can be used in Hydra CLI command.
    old_overrides = " ".join(filtered_overrides)

    # If SLURM should be used, add the corresponding string to the command.
    if args.use_slurm:
        slurm_str = "--multirun hydra/launcher=slurm_launcher"
    else:
        slurm_str = ""

    # Create the CLI command.
    command_str = (
        f"PYTHONPATH=. python scripts/resume_training_with_hydra.py {slurm_str} {old_overrides} "
        f"+resume_from_folder={args.resume_from_folder} +checkpoint_step={args.checkpoint_step} "
        f"{args.new_overrides}"
    )

    return command_str


if __name__ == "__main__":
    # Processing parameters.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume_from_folder",
        type=str,
        help="The run dir of the experiment that is to be resumed.",
    )
    parser.add_argument(
        "--checkpoint_step",
        type=int,
        default=-1,
        help="The step counter of the checkpoint that you want to use to resume training. "
        "Defaults to -1, which means that the latest/best checkpoint will be loaded.",
    )
    parser.add_argument(
        "--use_slurm",
        action="store_true",
        help="Whether to use SLURM for the continuation runs.",
    )
    parser.add_argument(
        "--new_overrides",
        type=str,
        default="",
        help="String that containes new overrides for this experiment.",
    )
    args = parser.parse_args()

    command_str = get_cli_command(args=args)

    print(command_str)
