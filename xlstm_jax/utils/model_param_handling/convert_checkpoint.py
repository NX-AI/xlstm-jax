#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from typing import Any

import jax
import torch

from ...trainer.base.param_utils import is_partitioned
from ..pytree_utils import flatten_pytree


def convert_orbax_checkpoint_to_torch_state_dict(
    orbax_pytree: dict[str, Any], split_blocks: bool = True, blocks_layer_name: str = "blocks"
) -> dict[str, torch.Tensor]:
    """Convert orbax pytree params to a (flat) torch state dict.

    Args:
        orbax_pytree (dict[str, Any]): The orbax pytree params.
        split_blocks (bool): Whether to split the parameters of the blocks into individual tensors. Defaults to True.
            Jax stores the weight tensors/arrays of the all blocks in
            a single tensor/array with the first dimension being the number of blocks.
            PyTorch expects the weights of each block to be a separate tensor/array.
            This is why we split the weights of each block into separate tensors/arrays.
        blocks_layer_name (str): The blocks layer name to split parameters by. Defaults to "blocks".

    Returns:
        dict[str, torch.Tensor]: The torch state dict.
    """

    # Convert JAX arrays to PyTorch tensors.
    params = jax.tree.map(lambda x: x.value if is_partitioned(x) else x, tree=orbax_pytree, is_leaf=is_partitioned)
    params_flat = flatten_pytree(params)

    torch_state_dict = {key: torch.tensor(jax.device_get(value)) for key, value in params_flat.items()}

    # jax stores the weight tensors/arrays of the all blocks in a single tensor/array with the first dimension
    # being the number of blocks. PyTorch expects the weights of each block to be a separate tensor/array.
    # This is why we split the weights of each block into separate tensors/arrays.
    if split_blocks:
        split_state_dict = {}
        for key, val in torch_state_dict.items():
            if "." + blocks_layer_name + "." not in key:
                split_state_dict[key] = val
            else:
                split_state_dict.update(
                    {
                        key.replace("." + blocks_layer_name + ".", "." + blocks_layer_name + f".{num}."): split_tensor
                        for num, split_tensor in enumerate(torch.unbind(val, dim=0))
                    }
                )
        torch_state_dict = split_state_dict

    return torch_state_dict
