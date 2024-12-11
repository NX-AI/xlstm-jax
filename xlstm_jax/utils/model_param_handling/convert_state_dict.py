#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import logging
import re
from typing import Any

import torch
from safetensors import safe_open

LOGGER = logging.getLogger(__name__)


def find_parameter_match_key(from_key: str, to_keys: list[str], match_dict: dict[str, str]) -> str:
    """Finds the matching parameter key for the target state dict.

    Args:
        from_key (str): The key of the source state dict.
        to_keys (list[str]): The keys of the target state dict.
        match_dict (dict[str, str]): The dict that maps the source state dict keys to the target state dict keys.
                                     Should contain unique substrings of the source state dict keys as keys and the
                                     corresponding target state dict keys substrings as values.

    Returns:
        The target state dict key that matches the source state dict key.
    """

    def _get_param_idx_hierarchy(key: str) -> list[int]:
        """Returns a list of indices that represent the hierarchy of the parameter in the model."""
        return [int(param_idx.replace(".", "")) for param_idx in re.findall(r"\.\d+\.", key)]

    from_key_idx_hierarchy = _get_param_idx_hierarchy(from_key)
    # search in match dict for the key-value match for the from_key
    from_to_match_dict_key_matches = []
    for k, v in match_dict.items():
        if k in from_key:
            from_to_match_dict_key_matches.append(k)
    if len(from_to_match_dict_key_matches) > 1:
        raise ValueError(
            f"Found multiple matches for {from_key} in match_dict.keys(): "
            f"{from_to_match_dict_key_matches}! Please fix the match dict."
        )
    elif len(from_to_match_dict_key_matches) == 0:
        raise ValueError(f"No match found for {from_key} in {list(match_dict.keys())}! Please fix the match dict.")
    from_to_match_dict_key = from_to_match_dict_key_matches[0]
    from_to_match_dict_value = match_dict[from_to_match_dict_key]
    # search in the to_keys for the matching state dict key
    to_state_dict_key_matches = []
    for k in to_keys:
        k_idx_hierarchy = _get_param_idx_hierarchy(k)
        if from_key_idx_hierarchy == k_idx_hierarchy and from_to_match_dict_value in k:
            to_state_dict_key_matches.append(k)
    # now we might end up with multiple matches, we need to find the correct one
    if len(to_state_dict_key_matches) == 1:
        # if there is only one match, we can return it
        return to_state_dict_key_matches[0]
    elif len(to_state_dict_key_matches) > 1:
        # the correct one has to match for the last part of the key, i.e. the part after the last dot
        last_part_from_key = from_key.split(".")[-1]
        for k in to_state_dict_key_matches:
            if last_part_from_key == k.split(".")[-1]:
                return k

        # if we did not find a match, we raise an error
        raise RuntimeError(
            f"Did not find a match for {from_key} in {to_state_dict_key_matches}! Please fix the match dict."
        )
    else:
        raise RuntimeError(f"No match found for {from_key} in {to_keys}! Please fix the match dict.")


def create_full_state_dict_key_mapping(from_state_dict: dict, to_state_dict: dict, match_dict: dict) -> dict:
    """Creates a full state dict key mapping from the source state dict to the target state dict.

    Args:
        from_state_dict (dict): The source state dict.
        to_state_dict (dict): The target state dict.
        match_dict (dict): The dict that maps the source state dict keys to the target state dict keys.

    Returns:
        dict: The full state dict key mapping from the source state dict to the target state dict.
    """
    from_keys = list(from_state_dict.keys())
    to_keys = list(to_state_dict.keys())
    full_state_dict_key_mapping = {}
    for from_key in from_keys:
        to_key = find_parameter_match_key(from_key, to_keys, match_dict)
        full_state_dict_key_mapping[from_key] = to_key
    return full_state_dict_key_mapping


def move_state_dict_params_(
    from_state_dict: dict[str, Any],
    to_state_dict: dict[str, Any],
    match_dict: dict[str, str],
) -> dict[str, Any]:
    """Move the params of a model from one state dict to another state dict.
    Modifies the to_state_dict in place.

    Args:
        from_state_dict (dict[str, Any]): The source state dict.
        to_state_dict (dict[str, Any]): The target state dict.
        match_dict (dict[str, str]): The dict that maps the source state dict keys to the target state dict keys.
                                     Should contain unique substrings of the source state dict keys as keys and the
                                     corresponding target state dict keys substrings as values.
    Returns:
        dict[str, Any]: The target (modified to_state_dict) state dict with the converted parameters.
    """

    full_state_dict_key_from_to_mapping = create_full_state_dict_key_mapping(from_state_dict, to_state_dict, match_dict)
    full_state_dict_key_to_from_mapping = {v: k for k, v in full_state_dict_key_from_to_mapping.items()}
    for to_key in to_state_dict.keys():
        from_key = full_state_dict_key_to_from_mapping[to_key]
        to_state_dict[to_key] = from_state_dict[from_key]
    return to_state_dict


def move_safetensors_state_dict_params_(
    from_state_dict_path: dict[str, Any],
    to_state_dict: dict[str, Any],
    match_dict: dict[str, str],
) -> dict[str, Any]:
    """Move the params of a model from one state dict to another state dict.
    It loads the from_state_dict from a file on-the-fly. This means only the to_state_dict is in memory.
    Modifies the to_state_dict in place.

    Args:
        from_state_dict (Path): The path to the source state dict.
        to_state_dict (dict[str, Any]): The target state dict.
        match_dict (dict[str, str]): The dict that maps the source state dict keys to the target state dict keys.
                                     Should contain unique substrings of the source state dict keys as keys and the
                                     corresponding target state dict keys substrings as values.
    Returns:
        dict[str, Any]: The target (modified to_state_dict) state dict with the converted parameters.
    """

    # load only the keys from the from_state_dict that are needed
    with safe_open(from_state_dict_path, framework="pt", device=0) as f:
        from_state_dict_dummy = {k: None for k in f.keys()}

    full_state_dict_key_from_to_mapping = create_full_state_dict_key_mapping(
        from_state_dict_dummy, to_state_dict, match_dict
    )

    # load the full from_state_dict into to_state_dict
    with safe_open(from_state_dict_path, framework="pt", device=0) as f:
        for from_key in f.keys():
            to_key = full_state_dict_key_from_to_mapping[from_key]
            to_state_dict[to_key] = f.get_tensor[from_key]

    return to_state_dict


def convert_state_dict_keys_(state_dict: dict[str, Any], full_key_mapping: dict[str, str]) -> dict[str, Any]:
    """Converts the keys of the state dict according to the key mapping.

    Args:
        state_dict (dict[str, Any]): The state dict to convert.
        full_key_mapping (dict[str, str]): The key mapping that maps the old keys to the new keys.

    Returns:
        dict[str, Any]: The converted state dict with the new keys.
    """
    new_state_dict = {}
    for from_key, to_key in full_key_mapping.items():
        new_state_dict[to_key] = state_dict[from_key]
    return new_state_dict


def apply_weight_transforms_(
    state_dict: dict[str, torch.Tensor], apply_transforms_to_keys: dict[str, list[str]]
) -> dict[str, torch.Tensor]:
    """Applies weight transforms to the weights of the state dict.

    There are currently these transforms supported:
        - "transpose": Transposes the weight tensor. Accepts only 2D tensors.
        - "squeeze-XXX": Squeezes the XXX dimension of the weight tensor.
           If XXX is not given, squeezes all dimensions of size 1.
        - "flatten": Flattens the weight tensor.

    If possible the transforms are applied in-place on tensors.
    Also the state_dict is modified in-place.

    Args:
        state_dict (dict[str, torch.Tensor]): The state dict with the weights.
        apply_transforms_to_keys (dict[str, list[str]]): The dict that maps the transform
            to the keys of the state dict.

    Returns:
        dict[str, torch.Tensor]: The state dict with the transformed weights
    """

    def _transpose_(tensor: torch.Tensor) -> torch.Tensor:
        assert tensor.dim() == 2, f"Can only transpose 2D tensors, got tensor with shape {tensor.shape}."
        return tensor.transpose_(0, 1)

    def _squeeze_(tensor: torch.Tensor, dim: int | None = None) -> torch.Tensor:
        if dim is None:
            return tensor.squeeze_()
        return tensor.squeeze_(dim)

    def _flatten(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.flatten()

    def _is_valid_transform(transform: str) -> bool:
        return transform in ["transpose", "flatten"] or transform.startswith("squeeze")

    def get_matching_full_keys_in_state_dict(key_partial: str):
        matching_keys = [key for key in state_dict.keys() if key_partial in key]
        assert len(matching_keys) > 0, f"No key found in state_dict that matches the partial key {key_partial}!"
        return matching_keys

    num_keys = len(list(state_dict.keys()))

    total_num_transform_applied = 0

    for transform, transform_apply_to_keys in apply_transforms_to_keys.items():
        if not _is_valid_transform(transform):
            raise ValueError(
                f"Invalid transform {transform}! Supported transforms are 'transpose', 'flatten' and 'squeeze-XXX'."
            )
        num_transform_applied = 0
        for partial_key in transform_apply_to_keys:
            matching_keys = get_matching_full_keys_in_state_dict(partial_key)
            num_transform_applied += len(matching_keys)
            for key in matching_keys:
                if transform == "transpose":
                    state_dict[key] = _transpose_(state_dict[key])
                elif transform == "flatten":
                    state_dict[key] = _flatten(state_dict[key])
                elif transform.startswith("squeeze"):
                    if "-" not in transform:
                        state_dict[key] = _squeeze_(state_dict[key])
                    else:
                        dim = int(transform.split("-")[1])
                        state_dict[key] = _squeeze_(state_dict[key], dim)

        LOGGER.info(f"Applied transform '{transform}' to {num_transform_applied} keys out of {num_keys}.")
        total_num_transform_applied += num_transform_applied

    LOGGER.info(f"Applied transforms to {total_num_transform_applied} keys out of {num_keys}.")
    return state_dict
