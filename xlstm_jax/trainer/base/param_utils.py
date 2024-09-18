from collections import defaultdict
from typing import Any

import jax
import numpy as np
from flax import linen as nn
from flax.core import FrozenDict
from tabulate import tabulate as python_tabulate

from .train_state import TrainState


def flatten_dict(d: dict) -> dict:
    """Flattens a nested dictionary."""
    flat_dict = {}
    for k, v in d.items():
        if isinstance(v, (dict, FrozenDict)):
            sub_dict = {f"{k}.{k2}": v2 for k2, v2 in flatten_dict(v).items()}
            # Verify that there are no overlapping keys.
            assert (
                len(set(sub_dict.keys()).intersection(set(flat_dict.keys()))) == 0
            ), f"Overlapping keys found in the nested dict: {set(sub_dict.keys()).intersection(set(flat_dict.keys()))}"
            flat_dict.update(sub_dict)
        else:
            flat_dict[k] = v
    return flat_dict


def tabulate_params(state: TrainState | dict[str, Any]) -> str:
    """Prints a summary of the parameters represented as table.

    Args:
        exmp_input: An input to the model with which the shapes are inferred.
    """
    if isinstance(state, TrainState):
        params = state.params
    else:
        params = state
    params = flatten_dict(params)
    param_shape = jax.tree.map(
        lambda x: x.value.shape if isinstance(x, nn.Partitioned) else x.shape,
        params,
        is_leaf=lambda x: isinstance(x, nn.Partitioned),
    )
    param_count = jax.tree.map(
        lambda x: int(np.prod(x)),
        param_shape,
        is_leaf=lambda x: isinstance(x, tuple) and all([isinstance(i, int) for i in x]),
    )
    param_dtype = jax.tree.map(
        lambda x: str(x.value.dtype if isinstance(x, nn.Partitioned) else x.dtype),
        params,
        is_leaf=lambda x: isinstance(x, nn.Partitioned),
    )
    param_sharding = jax.tree.map(
        lambda x: str(x.names if isinstance(x, nn.Partitioned) else "Replicated"),
        params,
        is_leaf=lambda x: isinstance(x, nn.Partitioned),
    )
    summary = defaultdict(list)
    for key in sorted(list(params.keys())):
        summary["Name"].append(key)
        summary["Shape"].append(param_shape[key])
        summary["Count"].append(param_count[key])
        summary["Dtype"].append(param_dtype[key])
        summary["Sharding"].append(param_sharding[key])
    return python_tabulate(summary, headers="keys", intfmt="_", floatfmt=".3f")
