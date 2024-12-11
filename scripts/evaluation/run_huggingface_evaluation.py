#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import argparse
import logging
import os
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__file__)


def main():
    """
    Script for running the the HuggingFace Leaderboard tasks, using lighteval and LM Evaluation Harness lm_eval scripts.
    For the new leaderboard it is using lm_eval (assuming it is installed from the HF github repo according to
    https://huggingface.co/docs/leaderboards/open_llm_leaderboard/about#reproducibility

    For the old leaderboard, we use lighteval with the tasks according to the standard settings from:
    https://huggingface.co/spaces/open-llm-leaderboard-old/open_llm_leaderboard

    This script assumes a `transformers` library to be installed that contains `xLSTMForCausalLM`.

    Please use your own `eval_output_dir`, `checkpoint_dir` and `default_converted_checkpoint_dir` flags.
    """
    parser = argparse.ArgumentParser("Run the default evaluation for a checkpoint via LightEval or HuggingFace lm_eval")
    parser.add_argument("--checkpoint_dir", help="Original Orbax checkpoint dir including the step idx")
    parser.add_argument(
        "--default_converted_checkpoint_dir", type=str, default="/nfs-gpu/xlstm/converted_model_checkpoints"
    )
    parser.add_argument("--no_convert_to_hf", action="store_true")
    parser.add_argument("--single-gpu", action="store_true")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--old_leaderboard", action="store_true")
    parser.add_argument(
        "--mixed_precision", type=str, default="no", help="Use mixed precision with optional dtypes fp16,bf16,fp32"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
    )

    parser.add_argument("--eval_output_dir", type=str, default=None)
    args = parser.parse_args()

    if args.eval_output_dir is None:
        if args.old_leaderboard:
            eval_output_dir = "/nfs-gpu/xlstm/logs/evals_leaderboardv1/"
        else:
            eval_output_dir = "/nfs-gpu/xlstm/logs/evals_leaderboard/"
    else:
        eval_output_dir = args.eval_output_dir
    if args.tasks is None:
        if args.old_leaderboard:
            tasks = (
                "leaderboard|arc:challenge|25|0,leaderboard|hellaswag|10|0,"
                "leaderboard|truthfulqa:mc|0|0,leaderboard|mmlu|5|0,"
                "leaderboard|winogrande|5|0,leaderboard|gsm8k|5|0,"
                "lighteval|piqa|5|0,lighteval|openbookqa|5|0"
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
            "scripts/checkpoint_conversion/convert_mlstm_checkpoint_jax_to_torch_simple.py",
            "--checkpoint_dir",
            str(checkpoint_dir),
            "--output_path",
            str(output_path),
            "--checkpoint_type",
            "huggingface",
        ]
        LOGGER.info("Running:\n" + " ".join(run_args))
        subprocess.run(
            run_args,
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

    run_args_accelerate = [
        "accelerate",
        "launch",
        "--multi_gpu" if not args.single_gpu else "",
        f"--num_processes={1 if args.single_gpu else 8}",
        f"--mixed_precision={args.mixed_precision}",
    ]

    if args.old_leaderboard:
        run_args = run_args_accelerate + [
            "--dynamo-backend=inductor",
            "-m",
            "lighteval",
            "accelerate",
            "--model_args",
            f"pretrained={str(output_path)},tokenizer=EleutherAI/gpt-neox-20b",
            "--override_batch_size",
            f"{args.batch_size}",
            f"--output_dir={str(eval_output_dir)}",
            "--tasks",
            str(tasks),
        ]
    else:
        run_args = run_args_accelerate + [
            "-m",
            "lm_eval",
            "--model_args",
            f"pretrained={str(output_path)},tokenizer=EleutherAI/gpt-neox-20b",
            f"--tasks={str(tasks)}",
            f"--batch_size={args.batch_size}",
            f"--output_path={str(eval_output_dir)}",
        ]
    LOGGER.info("Running:\n" + " ".join(run_args))
    subprocess.run(
        run_args,
        check=True,
    )


if __name__ == "__main__":
    main()
