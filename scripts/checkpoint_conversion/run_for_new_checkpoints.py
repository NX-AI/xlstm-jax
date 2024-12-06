import argparse
import logging
import os
import re
import subprocess
import sys
import time
from operator import itemgetter
from pathlib import Path

LOGGER = logging.getLogger(__name__)

LOG_FORMAT = "[%(asctime)s][%(name)s:%(lineno)d][%(levelname)s] - %(message)s"
logging.basicConfig(
    level="INFO",
    format=LOG_FORMAT.format(rank=""),
    force=True,
)


def get_subfolders(directory) -> set[str]:
    """Return a set of subfolder names in the given directory.

    Args:
        directory: Directory which should be searched for subfolders.

    Returns:
        Subfolder names of the directory.
    """
    try:
        return {entry.name for entry in os.scandir(directory) if entry.is_dir()}
    except FileNotFoundError:
        LOGGER.error(f"Error: The directory '{directory}' does not exist.")
        sys.exit(1)
    except PermissionError:
        LOGGER.error(f"Error: Permission denied for directory '{directory}'.")
        sys.exit(1)


def execute_command(command: str, folder_path: str):
    """Execute the given command with the folder path as an argument

    Errors are caught inspecting the return code. Stdout and stderr are shown and not captured.
    Exceptions during run are shown but do not cause a program halt or shutdown.

    Args:
        command: Command to be executed, should contain {} to insert the new folder path.
        folder_path: The new folder path the command should be executed on.

    Returns:
        None
    """
    try:
        # Insert the folder path to the command or append it.

        full_command = command.replace("{}", str(folder_path)).split()
        LOGGER.info(f"Executing command: {' '.join(full_command)}")
        result = subprocess.run(full_command, text=True, check=False)
        if result.returncode == 0:
            LOGGER.info("Success")
        else:
            LOGGER.error("Error")
    except Exception as e:
        LOGGER.error(f"Exception occurred while executing command: {e}")


def map_folder_name_sorting(fn: str):
    """
    Maps a folder name to a string that can be used in sorted().

    Should reflect numeric ordering. Here, numbers are padded by zeros up to 20 digits
    >>> map_folder_name_sorting("ckpt_100")
    'ckpt_00000000000000000100'

    """
    changed = True
    while changed:
        match = re.search(r"\d+", fn)
        digit_length = match.end() - match.start()
        if digit_length != 20:
            fn = fn[: match.start()] + (20 - digit_length) * "0" + fn[match.start() :]
            changed = True
        else:
            changed = False
    return fn


def sort_new_checkpoint_folders(folder_names: list[str]) -> list[str]:
    """
    Sorts checkpoint folders, according to the checkpoint index.

    Args:
        folder_names: The list of folder names.

    Returns:
        The sorted list of folder names

    >>> sort_new_checkpoint_folders(["ckpt_12500", "ckpt_2500", "ckpt_5000"])
    ['ckpt_2500', 'ckpt_5000', 'ckpt_12500']

    """

    return list(
        map(
            itemgetter(0),
            sorted(map(lambda fn: (fn, map_folder_name_sorting(fn)), folder_names), key=itemgetter(1)),
        )
    )


def monitor_folder(
    target_directory: str,
    check_interval: int,
    command: str,
    run_initially: bool = False,
    execution_wait_time: int = 1,
    skip_initially_num: int = 0,
):
    """Monitor the target directory for new subfolders and execute command on them.

    Args:
        target_directory: The directory of which the subdirectories should be checked.
        check_interval: Waiting time in seconds in between checks.
        command: Command to be executed on new directories.
        run_initially: Whether to run the command on all folders on first invocation.
        execution_wait_time: Time in seconds to wait between executions.
        skip_initially_num: Skip number of subfolders initially.

    Returns:
        None
    """
    LOGGER.info(f"Starting to monitor '{target_directory}' every {check_interval} seconds.")
    if run_initially:
        known_folders = set()
    else:
        known_folders = get_subfolders(target_directory)
    LOGGER.info(f"Initial subfolders: {known_folders}")

    n_runs = len(known_folders)
    try:
        while True:
            current_folders = get_subfolders(target_directory)
            new_folders = current_folders - known_folders
            if new_folders:
                for folder in sort_new_checkpoint_folders(new_folders):
                    n_runs += 1
                    if n_runs <= skip_initially_num:
                        continue
                    folder_path = target_directory / folder
                    LOGGER.info(f"New folder detected: {folder_path}")
                    execute_command(command, folder_path)
                    time.sleep(execution_wait_time)
                # Update the known folders set
                known_folders = current_folders
            else:
                LOGGER.info("No new folders detected.")
            time.sleep(check_interval)
    except KeyboardInterrupt:
        LOGGER.info(f"Ran on total on {n_runs} folders (including skips).")


def main():
    """
    Main function of run_for_new_checkpoints.py script
    """
    parser = argparse.ArgumentParser(
        description="Monitor a directory for new subfolders and execute a command on them."
    )
    parser.add_argument("--directory", help="Path to the directory to monitor.")
    parser.add_argument(
        "--command",
        help="Command or script to execute on each new subfolder. Use '{}' as a placeholder for the folder path.",
    )
    parser.add_argument(
        "--interval", type=int, default=1800, help="Time interval between checks in seconds (default: 300 seconds)."
    )
    parser.add_argument("--run_initially", action="store_true")
    parser.add_argument("--skip_initially_num", type=int, default=0, help="How many folders to skip in an initial run")
    parser.add_argument("--execution_wait_time", type=int, default=10, help="Time in between two command executions")
    args = parser.parse_args()

    monitor_folder(
        Path(args.directory),
        args.interval,
        args.command,
        run_initially=args.run_initially,
        execution_wait_time=args.execution_wait_time,
        skip_initially_num=args.skip_initially_num,
    )


if __name__ == "__main__":
    main()
