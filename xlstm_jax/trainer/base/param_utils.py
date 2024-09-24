import re
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.core import FrozenDict
from tabulate import tabulate as python_tabulate

from .train_state import TrainState

PyTree = Any


def flatten_dict(d: dict | FrozenDict | list | tuple, flatten_sequences: bool = False) -> dict:
    """Flattens a nested dictionary."""
    if not isinstance(d, (dict, FrozenDict)):
        assert flatten_sequences, "If flatten_sequences is False, only dicts and FrozenDicts are supported."
        return {f"{i}": v for i, v in enumerate(d)}
    flat_dict = {}
    for k, v in d.items():
        if isinstance(v, (dict, FrozenDict)) or (isinstance(v, (list, tuple)) and flatten_sequences):
            sub_dict = {f"{k}.{k2}": v2 for k2, v2 in flatten_dict(v, flatten_sequences=flatten_sequences).items()}
            # Verify that there are no overlapping keys.
            assert (
                len(set(sub_dict.keys()).intersection(set(flat_dict.keys()))) == 0
            ), f"Overlapping keys found in the nested dict: {set(sub_dict.keys()).intersection(set(flat_dict.keys()))}"
            flat_dict.update(sub_dict)
        else:
            flat_dict[k] = v
    return flat_dict


def get_num_params(params: PyTree) -> int:
    """Calculates the number of parameters in a PyTree."""
    param_shape = jax.tree.map(
        lambda x: x.value.size if isinstance(x, nn.Partitioned) else (x.size if hasattr(x, "size") else 0),
        params,
        is_leaf=lambda x: isinstance(x, nn.Partitioned),
    )
    return sum(jax.tree.leaves(param_shape))


def tabulate_params(
    state: TrainState | dict[str, Any],
    show_weight_decay: bool = False,
    weight_decay_exclude: Sequence[re.Pattern] | None = None,
    weight_decay_include: Sequence[re.Pattern] | None = None,
) -> str:
    """Prints a summary of the parameters represented as table.

    Args:
        state: The TrainState or the parameters as a dictionary.
        show_weight_decay: Whether to show the weight decay mask.
        weight_decay_exclude: List of regex patterns to exclude from weight decay. See optimizer config for more
            information.
        weight_decay_include: List of regex patterns to include in weight decay. See optimizer config for more
            information.

    Returns:
        str: The summary table as a string.
    """
    if isinstance(state, TrainState):
        params = state.params
    else:
        params = state
    flat_params = flatten_dict(params)
    param_shape = jax.tree.map(
        lambda x: x.value.shape if isinstance(x, nn.Partitioned) else x.shape,
        flat_params,
        is_leaf=lambda x: isinstance(x, nn.Partitioned),
    )
    param_count = jax.tree.map(
        lambda x: int(np.prod(x)),
        param_shape,
        is_leaf=lambda x: isinstance(x, tuple) and all([isinstance(i, int) for i in x]),
    )
    param_dtype = jax.tree.map(
        lambda x: str(x.value.dtype if isinstance(x, nn.Partitioned) else x.dtype),
        flat_params,
        is_leaf=lambda x: isinstance(x, nn.Partitioned),
    )
    param_sharding = jax.tree.map(
        lambda x: str(x.names if isinstance(x, nn.Partitioned) else "Replicated"),
        flat_params,
        is_leaf=lambda x: isinstance(x, nn.Partitioned),
    )
    summary = defaultdict(list)
    for key in sorted(list(flat_params.keys())):
        summary["Name"].append(key)
        summary["Shape"].append(param_shape[key])
        summary["Count"].append(param_count[key])
        summary["Dtype"].append(param_dtype[key])
        summary["Sharding"].append(param_sharding[key])
    if show_weight_decay:
        mask_fn = get_param_mask_fn(exclude=weight_decay_exclude, include=weight_decay_include)
        weight_decay_mask = mask_fn(params)
        weight_decay_mask = jax.tree.map(
            lambda x: x.value if isinstance(x, nn.Partitioned) else x,
            weight_decay_mask,
            is_leaf=lambda x: isinstance(x, nn.Partitioned),
        )
        weight_decay_mask = flatten_dict(weight_decay_mask)
        for key, mask in weight_decay_mask.items():
            summary["Weight Decay"].append(mask)
    table_str = python_tabulate(summary, headers="keys", intfmt="_", floatfmt=".3f")
    num_global_params = sum(param_count.values())
    return f"Total number of global parameters: {num_global_params:_}\n\n{table_str}"


def get_grad_norms(grads: Any, return_per_param: bool = False) -> dict[str, Any]:
    """
    Determine the gradient norms.

    Args:
        grads: The gradients as a PyTree.
        return_per_param: Whether to return the gradient norms per parameter or only the global norm.

    Returns:
        dict: A dictionary containing the gradient norms.
    """
    metrics = {}
    global_norm, param_norm = get_sharded_global_norm(grads)
    metrics["grad_norm"] = global_norm
    if return_per_param:
        param_norms = flatten_dict(param_norm)
        for key, norm in param_norms.items():
            metrics[f"grad_norm_{key}"] = norm
    return metrics


def get_param_norms(params: Any, return_per_param: bool = False) -> dict[str, Any]:
    """
    Determine the parameter norms.

    Args:
        params: The parameters as a PyTree.
        return_per_param: Whether to return the parameter norms per parameter or only the global norm.

    Returns:
        dict: A dictionary containing the parameter norms.
    """
    metrics = {}
    global_norm, param_norm = get_sharded_global_norm(params)
    metrics["param_norm"] = global_norm
    if return_per_param:
        param_norms = flatten_dict(param_norm)
        for key, norm in param_norms.items():
            metrics[f"param_norm_{key}"] = norm
    return metrics


def get_sharded_norm_logits(x: jax.Array | nn.Partitioned) -> jax.Array:
    """
    Calculate the norm of a sharded parameter or gradient.

    Args:
        x: The parameter or gradient.

    Returns:
        jax.Array: The norm logit, i.e. the squared norm.
    """
    if isinstance(x, nn.Partitioned):
        # For partitioned parameters, we first calculate the norm per device parameter.
        # Then, we sum the norm logit over every axes the parameter has been sharded over.
        # This gives the norm logit for the whole parameter.
        norm_logit = (x.value**2).sum()
        sharded_axes = [name for name in jax.tree.leaves(x.names) if name is not None]
        return jax.lax.psum(norm_logit, axis_name=sharded_axes)
    else:
        # For replicated parameters, we calculate the norm logit directly.
        return (x**2).sum()


def get_sharded_global_norm(x: PyTree) -> tuple[jax.Array, PyTree]:
    """
    Calculate the norm of a sharded PyTree.

    Args:
        x: The PyTree. Each leaf should be a jax.Array or nn.Partitioned.

    Returns:
        tuple[jax.Array, PyTree]: The global norm and the norm per leaf.
    """
    # Calculate the global norm over sharded parameters.
    # General norm: sqrt(sum(x**2))
    norm_logits_per_param = jax.tree.map(
        lambda x: get_sharded_norm_logits(x), x, is_leaf=lambda x: isinstance(x, nn.Partitioned)
    )
    norm_sq = jax.tree_util.tree_reduce(jnp.add, norm_logits_per_param)
    return jnp.sqrt(norm_sq), jax.tree.map(jnp.sqrt, norm_logits_per_param)


def _key_path_to_str(path: jax.tree_util.KeyPath) -> str:
    """Converts a path to a string.

    An adjusted version of jax.tree_util.keystr to be more intuitive
    and fitting to our flatten_dict method.

    Args:
        path (jax.tree_util.KeyPath): Path.

    Returns:
        str: Path as string.
    """
    cleaned_keys = []
    for key in path:
        if isinstance(key, jax.tree_util.DictKey):
            cleaned_keys.append(f"{key.key}")
        elif isinstance(key, jax.tree_util.SequenceKey):
            cleaned_keys.append(f"{key.idx}")
        elif isinstance(key, jax.tree_util.GetAttrKey):
            cleaned_keys.append(key.name)
        else:
            cleaned_keys.append(str(key))
    return ".".join(cleaned_keys)


def get_param_mask_fn(
    exclude: Sequence[str] | None, include: Sequence[str] | None = None
) -> Callable[[PyTree], PyTree]:
    """
    Returns a function that generates a mask, which can for instance be used for weight decay.

    Args:
        exclude (Sequence[str]): List of strings to exclude.
        include (Sequence[str]): List of strings to include. If None, all parameters except those in exclude are
            included.

    Returns:
        Callable[[PyTree], PyTree]: Function that generates a mask.
    """
    assert exclude is None or include is None, "Only one of exclude or include can be set."

    def is_param_included(path, _):
        param_name = _key_path_to_str(path)
        if exclude is not None:
            return not any(re.search(excl, param_name) for excl in exclude)
        elif include is not None:
            return any(re.search(incl, param_name) for incl in include)
        else:
            return True

    def mask_fn(params: PyTree):
        mask_tree = jax.tree_util.tree_map_with_path(
            is_param_included,
            params,
        )
        return mask_tree

    return mask_fn
