#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any
import pandas as pd
from tqdm.auto import tqdm
from xlstm_jax.utils.model_param_handling.handle_mlstm_simple import (
    create_mlstm_simple_config_from_jax_config,
    convert_mlstm_checkpoint_jax_to_torch_simple,
    load_model_params_and_config_from_checkpoint,
)

LOGGER = logging.getLogger(__name__)


class TqdmLoggingHandler(logging.Handler):
    """Logging handler that keeps tqdm progress bars readable."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            tqdm.write(message, file=sys.stdout)
            self.flush()
        except Exception:
            self.handleError(record)


def format_compact_count(value: float | int) -> str:
    """Format large numeric counts with compact M/B suffixes."""
    numeric_value = float(value)
    abs_value = abs(numeric_value)

    if abs_value >= 1_000_000_000:
        return f"{numeric_value / 1_000_000_000:.2f}".rstrip("0").rstrip(".") + "B"
    if abs_value >= 1_000_000:
        return f"{numeric_value / 1_000_000:.2f}".rstrip("0").rstrip(".") + "M"
    return f"{numeric_value:.0f}"


def process_checkpoint_conversion_for_checkpoint(
    checkpoint_data: dict[str, Any],
    input_directory: str,
    output_directory: str,
    max_shard_size: int,
    dryrun: bool = False,
) -> None:

    # Load JAX checkpoint path
    # The checkpoint paths is a json string containing a list of checkpoint paths.
    # We select the last checkpoint from the list with format "checkpoint_{step}" for conversion, as this run folder contains the latest checkpoint.
    jax_original_checkpoint_path = json.loads(checkpoint_data["model_checkpoint_paths"])[-1]

    # Format the jax checkpoint path to strip the prefix
    # Take the part after ...outputs_beck/[SELECT/]wandb
    if "outputs_beck/" not in jax_original_checkpoint_path:
        LOGGER.info(
            f"'outputs_beck/' not found in checkpoint path: {jax_original_checkpoint_path}, skipping checkpoint."
        )
        return

    jax_checkpoint_path = jax_original_checkpoint_path.split("outputs_beck/")[1]
    jax_checkpoint_path = jax_checkpoint_path.split("wandb")[0]

    # Move into /checkpoints subdirectory
    jax_checkpoint_path = Path(input_directory) / jax_checkpoint_path / "checkpoints"

    # Select the last checkpoint from the checkpoint path list with format "checkpoint_{step}"
    checkpoint_list = jax_checkpoint_path.glob("checkpoint_*")
    checkpoint_list = sorted(checkpoint_list, key=lambda x: int(x.name.split("_")[1]))
    jax_checkpoint_path = checkpoint_list[-1]

    LOGGER.info(
        f"Found {len(checkpoint_list)} JAX checkpoints, original_checkpoint_path: {jax_original_checkpoint_path}, selected: {jax_checkpoint_path}"
    )

    # Create output directory
    output_dir = Path(output_directory)

    # Output checkpoint path:
    checkpoint_folder_name = (
        "{model_type}--{experiment_set}--ctx-{ctx_length}--params-{num_params}--tokens-{num_tokens}--id-{id}"
    )

    checkpoint_folder = output_dir / checkpoint_folder_name.format(
        model_type=checkpoint_data["model_type"],
        experiment_set=checkpoint_data["experiment_set"],
        id=checkpoint_data["run_id"],
        ctx_length=checkpoint_data["context_length"],
        num_params=format_compact_count(checkpoint_data["num_params"]),
        num_tokens=format_compact_count(checkpoint_data["num_tokens_training"]),
    )

    LOGGER.info(f"Converting checkpoint from {jax_checkpoint_path} to {checkpoint_folder}")

    if not dryrun:
        checkpoint_folder.mkdir(parents=True, exist_ok=True)
        # Save metadata (checkpoint data)
        with open(checkpoint_folder / "metadata.json", "w") as f:
            json.dump(checkpoint_data, f)

        # Convert checkpoint
        try:
            convert_mlstm_checkpoint_jax_to_torch_simple(
                load_jax_model_checkpoint_path=jax_checkpoint_path,
                store_torch_model_checkpoint_path=checkpoint_folder,
                checkpoint_type="huggingface",
                max_shard_size=args.max_shard_size,
            )
        except RuntimeError as e:
            # Note: We have different ffn_dim computations in jax and torch due to the different rounding strategies for the projection up dimension. This can lead to parameter mismatch issues during conversion. To mitigate this, we catch the RuntimeError and retry the conversion with a different ffn_proj_factor that leads to a different rounding result for the projection up dimension.
            LOGGER.warning(f"Conversion failed for checkpoint {jax_checkpoint_path} with error: {e}")
            LOGGER.warning(f"Retrying conversion with different ffn_proj_factor to avoid parameter mismatch issues..")

            _, jax_model_config = load_model_params_and_config_from_checkpoint(jax_checkpoint_path)
            mlstm_simple_config = create_mlstm_simple_config_from_jax_config(jax_model_config)
            orig_ffn_proj_fac = mlstm_simple_config.ffn_proj_factor
            ffn_round_up_to_multiple_of = mlstm_simple_config.ffn_round_up_to_multiple_of
            new_ffn_proj_factor = orig_ffn_proj_fac + (
                ffn_round_up_to_multiple_of / (orig_ffn_proj_fac * mlstm_simple_config.embedding_dim)
            )
            LOGGER.info(
                f"Original ffn_proj_factor: {orig_ffn_proj_fac}, new ffn_proj_factor: {new_ffn_proj_factor}. Converting checkpoint with new ffn_proj_factor..."
            )
            convert_mlstm_checkpoint_jax_to_torch_simple(
                load_jax_model_checkpoint_path=jax_checkpoint_path,
                store_torch_model_checkpoint_path=checkpoint_folder,
                checkpoint_type="huggingface",
                torch_model_config_overrides={"ffn_proj_factor": new_ffn_proj_factor},
                max_shard_size=args.max_shard_size,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converts a JAX orbax checkpoints from xLSTM scaling laws to PyTorch safetensors checkpoint for mlstm_simple. \n"
        'Use together with JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICE="" to not run out of memory.'
    )
    # Schema for csv file:
    # experiment_set_ctx_length,name,run_tag,model_type,num_params,num_tokens_training,num_flops_training,val/.dclm_loss,token_param_ratio,width_depth_ratio,Preset Token Param Ratio,experiment_set,context_length,learning_rate,global_batch_size,num_train_steps,val/.dclm_perplexity,Preset Num Params,Model Size,embedding_dim,num_blocks,num_heads,proj_factor_ffn,ffn_multiple_of,ffn_dim,head_dim_qk,head_dim_v,IsoFLOP,train/.loss_mean,run_id,model_checkpoint_paths
    parser.add_argument(
        "--checkpoints_file", type=str, help="Path to the .csv file containing the checkpoints to convert"
    )
    parser.add_argument("--input_dir", type=str, help="Path to the root input directory containing the JAX checkpoints")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory for the converted checkpoints")
    parser.add_argument(
        "--max_shard_size",
        type=int,
        default=4294967296,
        help="Maximum shard size in bytes for the safetensors output. Defaults to 4GB.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, the script will only print the actions it would take without performing any conversions.",
    )

    args = parser.parse_args()
    stdout_handler = TqdmLoggingHandler()
    logging.basicConfig(
        handlers=[stdout_handler],
        level=logging.INFO,
        format="[%(asctime)s][%(name)s:%(lineno)d][%(levelname)s] - %(message)s",
        force=True,
    )

    checkpoint_df = pd.read_csv(args.checkpoints_file)
    checkpoint_records = checkpoint_df.to_dict("records")

    LOGGER.info(f"Loaded {len(checkpoint_records)} checkpoints from {args.checkpoints_file}")
    
    # jax.distributed.initialize()  # Initialize JAX distributed system to avoid issues with loading large checkpoints, even for CPU usage.

    successful = []
    failed = []
    for checkpoint_dict in tqdm(checkpoint_records, desc="Converting checkpoints", unit="ckpt"):
        LOGGER.info(
            f"\n\nProcessing checkpoint with run name: {checkpoint_dict.get('name', 'N/A')} and run_id: {checkpoint_dict.get('run_id', 'N/A')} and model_checkpoint_paths: {checkpoint_dict.get('model_checkpoint_paths', 'N/A')}\n\n"
        )

        try:
            process_checkpoint_conversion_for_checkpoint(
                checkpoint_dict, args.input_dir, args.output_dir, args.max_shard_size, args.dry_run
            )
            successful.append(checkpoint_dict)
        except Exception as e:
            LOGGER.error(
                f"Error processing checkpoint with run name: {checkpoint_dict.get('name', 'N/A')} and run_id: {checkpoint_dict.get('run_id', 'N/A')}. Error: {e}"
            )
            failed.append(checkpoint_dict)

    successful_paths = '\n'.join([ckpt.get('model_checkpoint_paths', 'N/A') for ckpt in successful])
    LOGGER.info(f"Successfully converted {len(successful)} checkpoints: {successful_paths}")
    if len(failed) > 0:
        failed_paths = '\n'.join([ckpt.get('model_checkpoint_paths', 'N/A') for ckpt in failed])
        LOGGER.warning(
            f"Failed to convert {len(failed)} checkpoints. Failed checkpoint paths: {failed_paths}"
        )

"""Example usage:

Converting tokenparam xlstm checkpoints: 

PYTHONPATH=. python scripts/checkpoint_conversion/convert_mlstm_sclaw_checkpoints_to_torch.py \
    --checkpoints_file "./scripts/checkpoint_conversion/tokenparam_mlstm.csv" \
    --input_dir "/nfs-gpu/users_work/beck/xlstm_sclaw_ckpts/scaling_law_checkpoints" \
    --output_dir "/nfs-gpu/users_work/beck/xlstm_sclaw_ckpts/converted/xlstm/tokenparam" \
    --max_shard_size 4294967296 \
    --dry-run
"""
