#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import json
import logging
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

LOGGER = logging.getLogger(__name__)


def store_checkpoint_sharded(
    state_dict: dict[str, torch.Tensor],
    checkpoint_path: Path,
    max_shard_size: int = 1 << 30,
    metadata: dict[str, Any] = None,
):
    """Save model parameters in sharded fashion into multiple safetensors files.

    Args:
        state_dict (dict[str, torch.Tensor]): Model state dict.
        checkpoint_path (Path): Checkpoint Path for the model to be stored in.
        max_shard_size (int, optional): Maximal shard size in bytes. Defaults to 1<<30.
        metadata (dict[str, Any], optional): Additional metadata for the checkpoint. Defaults to {}.
    """
    if max_shard_size != 0:
        keys = list(state_dict)
        shard_size = 0
        idx = 0
        key_idx = 0
        total_size = 0
        weight_map = {}
        torch_state_dict_shard = {}
        next_tensor = state_dict[keys[key_idx]]
        next_size = int(next_tensor.dtype.itemsize * next_tensor.numel())
        while key_idx < len(list(keys)):
            while shard_size + next_size < max_shard_size and key_idx < len(list(keys)):
                torch_state_dict_shard[keys[key_idx]] = next_tensor
                weight_map[keys[key_idx]] = checkpoint_path.name.replace(".safetensors", f"_{idx}.safetensors")
                total_size += next_size
                shard_size += next_size
                key_idx += 1
                if key_idx < len(list(keys)):
                    next_tensor = state_dict[keys[key_idx]]
                    next_size = int(next_tensor.dtype.itemsize * next_tensor.numel())
            if shard_size == 0:
                LOGGER.error("Single tensor is larger than maximal file size.")
                break
            save_file(
                torch_state_dict_shard,
                str(checkpoint_path).replace(".safetensors", f"_{idx}.safetensors"),
                metadata=metadata,
            )
            torch_state_dict_shard = {}
            shard_size = 0
            idx += 1
        with open(str(checkpoint_path) + ".index.json", "w", encoding="utf-8") as fp:
            json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, fp)
    else:
        save_file(state_dict, checkpoint_path, metadata=metadata)
