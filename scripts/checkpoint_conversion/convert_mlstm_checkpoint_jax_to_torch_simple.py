#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import argparse
import logging
import sys
from pathlib import Path

from xlstm_jax.utils.model_param_handling.handle_mlstm_simple import convert_mlstm_checkpoint_jax_to_torch_simple

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converts a JAX orbax checkpoint to PyTorch safetensors checkpoint for mlstm_simple. \n"
        'Use together with JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICE="" to not run out of memory.'
    )
    parser.add_argument("--checkpoint_dir", type=str, help="Directory of the orbax model checkpoint")
    parser.add_argument("--output_path", type=str, help="Output Folder for the safetensors checkpoint")
    parser.add_argument("--max_shard_size", type=int, default=0, help="Maximum shard size for the safetensors output")
    parser.add_argument(
        "--checkpoint_type", type=str, default="plain", help="Which checkpoint type to use (plain | huggingface)"
    )

    args = parser.parse_args()
    stdout_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(
        handlers=[stdout_handler],
        level=logging.DEBUG,
        format="[%(asctime)s][%(name)s:%(lineno)d][%(levelname)s] - %(message)s",
    )

    convert_mlstm_checkpoint_jax_to_torch_simple(
        load_jax_model_checkpoint_path=Path(args.checkpoint_dir),
        store_torch_model_checkpoint_path=Path(args.output_path),
        checkpoint_type=args.checkpoint_type,
        max_shard_size=args.max_shard_size,
    )
