import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def run_experiment(config_group: str, config_name: str, script_path: str, wait_time: int, pythonpath: str = "."):
    """
    Constructs and runs the Hydra command for a single experiment,
    then sends SIGINT after a delay.
    """
    hydra_override = f"+{config_group}={config_name}"
    # Ensure script_path is treated correctly relative to the script's execution dir
    # Assuming the script is run from the project root where PYTHONPATH=. makes sense
    command = ["python", script_path, hydra_override]

    print(f"\n--- Starting experiment for: {config_group}/{config_name} ---")
    # print(f"Running command: PYTHONPATH={pythonpath} {' '.join(command)}")
    print(f"Running command: PYTHONPATH={pythonpath} {' '.join(command)}")

    # Set PYTHONPATH environment variable for the subprocess
    # This ensures the python interpreter running the script can find modules
    # in the current directory (.) as specified in your original command.
    env = os.environ.copy()
    # Prepend '.' to existing PYTHONPATH or set it if it doesn't exist
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = pythonpath + os.pathsep + existing_pythonpath if existing_pythonpath else "."

    process = None
    try:
        # Use Popen to start the process without blocking
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,  # Capture standard output
            stderr=subprocess.PIPE,  # Capture standard error
            text=True,  # Decode output/error as text
            env=env,  # Pass the modified environment
        )

        print(f"Process started (PID: {process.pid}). Waiting {wait_time} seconds before sending SIGINT...")

        # Wait for the specified time
        # This allows the hydra script to initialize and potentially submit the job
        time.sleep(wait_time)

        # Check if the process is still running before sending signal
        if process.poll() is None:  # poll() returns None if process is running
            print(f"Sending SIGINT (Ctrl+C equivalent) to process {process.pid}...")
            process.send_signal(signal.SIGINT)
        else:
            print(f"Process {process.pid} already terminated before SIGINT could be sent.")

        # Wait for the process to terminate completely and capture output
        # communicate() reads all output/error until EOF and waits for process termination
        stdout, stderr = process.communicate()

        print(f"--- Output for {config_name} ---")
        if stdout:
            print("Stdout:\n", stdout.strip())
        if stderr:
            # You can optionally filter the known verbose startup messages
            # filtered_stderr_lines = [
            #     line for line in stderr.splitlines()
            #     if not ("Unable to register cuFFT factory" in line or
            #             "Unable to register cuDNN factory" in line or
            #             "Unable to register cuBLAS factory" in line or
            #             "Could not find TensorRT" in line)
            # ]
            # if filtered_stderr_lines:
            #      print("Stderr (filtered):\n", "\n".join(filtered_stderr_lines))
            # Or print the raw stderr if you prefer:
            print("Raw Stderr:\n", stderr.strip())

        # SIGINT often results in a negative return code
        print(f"--- Finished attempt for: {config_group}/{config_name} (Return code: {process.returncode}) ---")

    except FileNotFoundError:
        print(f"Error: The script '{script_path}' was not found.", file=sys.stderr)
        # If the main script is missing, no point continuing
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while running {config_name}: {e}", file=sys.stderr)
        if process and process.poll() is None:
            print("Terminating potentially running process...")
            process.terminate()  # Attempt graceful termination
            try:
                process.wait(timeout=5)  # Wait briefly
            except subprocess.TimeoutExpired:
                process.kill()  # Force kill if necessary
        # Decide if you want to stop the whole script or continue with the next config
        # For now, let's continue:
        # pass


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run multiple Hydra experiments sequentially from config files in a directory.",
            " ATTENTION: use only absolute paths.",
        )
    )
    parser.add_argument(
        "config_dir", type=str, help="Path to the directory containing Hydra configuration files (.yaml/.yml)."
    )
    parser.add_argument(
        "--script",
        type=str,
        # default="scripts/training/train_with_hydra.py",
        help="Path to the Python script to execute for training.",
    )
    parser.add_argument(
        "--wait",
        type=int,
        default=25,  # Default wait time in seconds before sending SIGINT
        help="Seconds to wait after starting the command before sending SIGINT (Ctrl+C).",
    )
    parser.add_argument(
        "--pythonpath",
        type=str,
        # default=".",
        help="PYTHONPATH to set for the subprocess. Default is current directory.",
    )
    parser.add_argument("--filename_suffices", type=str, help="File name suffices to filter config files, E.g '_0,_1'.")

    args = parser.parse_args()

    config_path = Path(args.config_dir)
    script_path = args.script
    wait_time = args.wait
    pythonpath = args.pythonpath

    filname_suffices = args.filename_suffices.split(",") if args.filename_suffices else []

    if not config_path.is_dir():
        print(f"Error: Configuration directory not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    if not Path(script_path).is_file():
        print(f"Error: Training script not found: {script_path}", file=sys.stderr)
        sys.exit(1)

    # The config group name is derived from the directory name, as in your example
    config_group = config_path.name

    config_files = sorted([f for f in config_path.iterdir() if f.is_file() and f.suffix in [".yaml", ".yml"]])

    if not config_files:
        print(f"No .yaml or .yml files found in {config_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(config_files)} configuration files in '{config_path}'.")

    filtered_config_files = []
    if filname_suffices:
        for suff in filname_suffices:
            filtered_config_files.extend([f for f in config_files if f.stem.endswith(suff)])

        print(f"Filtered {len(filtered_config_files)} configuration files with suffices: {filname_suffices}.")
        config_files = filtered_config_files

    print(f"Using {len(config_files)} configuration files for experiments.")
    print(f"Using config group name: '{config_group}'")
    print(f"Using script: '{script_path}'")
    print(f"Wait time before SIGINT: {wait_time} seconds")
    print(f"PYTHONPATH for subprocess: {pythonpath}")
    print("\n--- Starting experiments ---")

    for i, config_file in enumerate(config_files):
        print(f"\n--- Starting experiment {i + 1}/{len(config_files)} ---")
        config_name = config_file.stem  # Get filename without extension
        run_experiment(config_group, config_name, script_path, wait_time, pythonpath=pythonpath)

    print("\n--- FINISHED EXPERIMENT START ---")
    print(f"\nAll {len(config_files)} configuration files processed.")


if __name__ == "__main__":
    # example usage:
    # python scripts/training/run_train_with_hydra_sweep.py
    # /home/beck/wdir/cleaned_repos/xlstm-jax-internal/configs/experiment_sclaw_mlstmctx_iso
    # --script '/home/beck/wdir/cleaned_repos/xlstm-jax-internal/scripts/training/train_with_hydra.py'
    # --pythonpath '/home/beck/wdir/cleaned_repos/xlstm-jax-internal'
    main()
