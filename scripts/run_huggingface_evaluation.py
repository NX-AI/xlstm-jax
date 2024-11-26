import argparse
import logging
import os
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__file__)


def main():
    parser = argparse.ArgumentParser("Run the default evaluation for a checkpoint via LightEval")
    parser.add_argument("--checkpoint_dir", help="Original Orbax checkpoint dir including the step idx")
    parser.add_argument(
        "--default_converted_checkpoint_dir", type=str, default="/nfs-gpu/xlstm/converted_model_checkpoints"
    )
    parser.add_argument("--no_convert_to_hf", action="store_true")
    parser.add_argument("--single-gpu", action="store_true")
    parser.add_argument("--old_leaderboard", action="store_true")
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
    )

    parser.add_argument("--eval_output_dir", type=str, default=None)
    args = parser.parse_args()

    if args.eval_output_dir is None:
        if args.old_leaderboard:
            eval_output_dir = "/nfs-gpu/xlstm/logs/evals/"
        else:
            eval_output_dir = "nfs-gpu/xlstm/logs/evals_leaderboard/"
    else:
        eval_output_dir = args.eval_output_dir
    if args.tasks is None:
        if args.old_leaderboard:
            tasks = (
                "leaderboard|winogrande|5|0,leaderboard|hellaswag|5|0,"
                "leaderboard|arc:challenge|5|0,lighteval|arc:easy|5|0,leaderboard|mmlu|5|0,lighteval|openbookqa|5|0,"
                "lighteval|piqa|5|0"
            )
        else:
            tasks = "leaderboard"
    else:
        tasks = args.tasks

    checkpoint_dir = Path(os.path.abspath(args.checkpoint_dir))
    checkpoint_idx = checkpoint_dir.name.split("_")[-1]

    checkpoint_name = f"{checkpoint_dir.parent.parent.parent.name}_{checkpoint_dir.parent.parent.name}_{checkpoint_idx}"

    previous_jax_platforms = os.environ.get("JAX_PLATFORMS", None)

    output_path = Path(args.default_converted_checkpoint_dir) / checkpoint_name
    if output_path.exists():
        LOGGER.info("Skipping checkpoint conversion.")
    else:
        os.environ["JAX_PLATFORMS"] = "cpu"
        run_args = [
            "python3",
            "scripts/convert_mlstm_checkpoint_jax_to_torch_simple.py",
            "--checkpoint_dir",
            str(checkpoint_dir),
            "--output_path",
            str(output_path),
            "--checkpoint_type",
            "huggingface",
        ]
        LOGGER.info("Running:\n" + " ".join(run_args))
        subprocess.run(
            [
                "python3",
                "scripts/convert_mlstm_checkpoint_jax_to_torch_simple.py",
                "--checkpoint_dir",
                str(checkpoint_dir),
                "--output_path",
                str(output_path),
                "--checkpoint_type",
                "huggingface",
            ],
            check=True,
        )
        if previous_jax_platforms is not None:
            os.environ["JAX_PLATFORMS"] = previous_jax_platforms
        else:
            del os.environ["JAX_PLATFORMS"]

    output_path_hf = Path(str(output_path) + "_hf")
    if not args.no_convert_to_hf and not output_path_hf.exists():
        run_args = [
            "python3",
            "-c",
            f'import transformers ; transformers.AutoModelForCausalLM.from_pretrained("{str(output_path)}")'
            f'.save_pretrained("{str(output_path_hf)}")',
        ]

        LOGGER.info("Running:\n" + " ".join(run_args))
        subprocess.run(
            run_args,
            check=True,
        )

        output_path = output_path_hf
    else:
        LOGGER.info("Skipping HF checkpoint conversion.")

    if args.old_leaderboard:
        run_args = [
            "accelerate",
            "launch",
            "--multi_gpu" if not args.single_gpu else "",
            f"--num_processes={1 if args.single_gpu else 8}",
            "--dynamo_backend=inductor",
            "-m",
            "lighteval",
            "accelerate",
            "--model_args",
            f"pretrained={str(output_path)},tokenizer=EleutherAI/gpt-neox-20b",
            "--override_batch_size",
            "16",
            f"--output_dir={str(eval_output_dir)}",
            "--tasks",
            str(tasks),
        ]
    else:
        run_args = [
            "accelerate",
            "launch",
            "--multi_gpu",
            "--num_processes=8",
            "-m",
            "lm_eval",
            "--model_args",
            f"pretrained={str(output_path)},tokenizer=EleutherAI/gpt-neox-20b",
            f"--tasks={str(tasks)}",
            "--batch_size=16",
            f"--output_path={str(output_path)}",
        ]
    LOGGER.info("Running:\n" + " ".join(run_args))
    subprocess.run(
        run_args,
        check=True,
    )


if __name__ == "__main__":
    main()
