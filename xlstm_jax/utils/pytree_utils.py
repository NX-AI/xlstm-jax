import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict

from xlstm_jax.common_types import PyTree

LOGGER = logging.getLogger(__name__)


class RecursionLimit:
    pass


def pytree_diff(tree1: PyTree, tree2: PyTree) -> PyTree:
    """
    Computes the difference between two PyTrees.

    Args:
        tree1: First PyTree.
        tree2: Second PyTree.

    Returns:
        A PyTree of the same structure, with only differing leaves.
        Returns None if no differences are found.

    >>> pytree_diff({"a": 1}, {"a": 2})
    {'a': (1, 2)}
    >>> pytree_diff({"a": 1}, {"a": 1})
    >>> pytree_diff([1, 2, 3], [1, 2])
    {'length_mismatch': (3, 2)}
    >>> pytree_diff(np.array([1, 2, 3]), np.array([1, 2]))
    {'shape_mismatch': ((3,), (2,))}
    """

    def diff_fn(a, b) -> Any:
        """
        Creates a diff of two elementary objects / leaves.

        Args:
            a: Any (not dict|list)
            b: Any (not dict|list)

        Returns:
            None if a == b else an informative diff object
        """
        # Check if both are arrays and calculate the difference
        if isinstance(a, (jnp.ndarray, np.ndarray)) or isinstance(b, (jnp.ndarray, np.ndarray)):
            if isinstance(a, (jnp.ndarray, np.ndarray)) and isinstance(b, (jnp.ndarray, np.ndarray)):
                if a.shape != b.shape:
                    return {"shape_mismatch": (a.shape, b.shape)}
            try:
                if a.dtype == bool:
                    diff = a ^ b
                else:
                    diff = a - b
            except ValueError:
                return {"array_difference": (a, b)}
            if isinstance(diff, jax.Array):
                return diff if not np.allclose(diff, jnp.zeros_like(diff)) else None
            return diff if not np.allclose(diff, np.zeros_like(diff)) else None

        # Check for scalar values and report if different
        if a != b:
            return a, b
        # If identical, ignore
        return None

    def recursive_diff(t1, t2, max_recursion=20):
        """
        Recursive diff function for two PyTrees.

        Args:
            t1: PyTree object 1
            t2: PyTree object 2
            max_recursion: Recursion limiter

        Returns:
            None if the PyTree objects are equal, else an informative (recursive) diff object
        """
        if max_recursion == 0:
            return RecursionLimit
        if isinstance(t1, (jnp.ndarray, np.ndarray)) or isinstance(t2, (jnp.ndarray, np.ndarray)):
            return diff_fn(t1, t2)
        # Case 1: Both are mappings (e.g., dictionaries)
        if isinstance(t1, Mapping) and isinstance(t2, Mapping):
            diff = {}
            all_keys = set(t1.keys()).union(set(t2.keys()))
            for key in all_keys:
                val1, val2 = t1.get(key), t2.get(key)
                if key not in t1:
                    diff[key] = {"only_in_tree2": val2}
                elif key not in t2:
                    diff[key] = {"only_in_tree1": val1}
                else:
                    sub_diff = recursive_diff(val1, val2, max_recursion=max_recursion - 1)
                    if sub_diff is not None:
                        diff[key] = sub_diff
            return diff if diff else None

        # Case 2: Both are sequences (e.g., lists, tuples) and of the same type
        if (
            isinstance(t1, Sequence)
            and isinstance(t2, Sequence)
            and isinstance(t2, type(t1))
            and isinstance(t1, type(t2))
            and not isinstance(t1, str)
        ):
            if len(t1) != len(t2):
                return {"length_mismatch": (len(t1), len(t2))}
            diff = [recursive_diff(x, y, max_recursion=max_recursion - 1) for x, y in zip(t1, t2)]
            diff = [d for d in diff if d is not None]
            return diff if diff else None

        # Case 3: Both are comparable types (e.g., scalars, arrays)
        return diff_fn(t1, t2)

    diff_tree = recursive_diff(tree1, tree2)
    return diff_tree if diff_tree else None


def pytree_key_path_to_str(path: jax.tree_util.KeyPath, separator: str = ".") -> str:
    """Converts a path to a string.

    An adjusted version of jax.tree_util.keystr to support different separators and easier to read output.

    Args:
        path: Path.
        separator: Separator for the keys.

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
    return separator.join(cleaned_keys)


def flatten_pytree(
    pytree: PyTree, separator: str = ".", is_leaf: Callable[[Any], bool] | None = None
) -> dict[str, Any]:
    """
    Flattens a PyTree into a dict.

    Supports PyTrees with nested dictionaries, lists, tuples, and more. The keys are created by concatenating the
    path to the leaf with the separator. For sequences, the index is used as key (see examples below).

    Args:
        pytree: PyTree to be flattened.
        separator: Separator for the keys.
        is_leaf: Function that determines if a node is a leaf. If None, uses default PyTree leaf detection.

    Returns:
        dict: Flattened PyTree. In case of duplicate keys, a ValueError is raised.

    >>> flatten_pytree({"a": 1, "b": {"c": 2}})
    {'a': 1, 'b.c': 2}
    >>> flatten_pytree({"a": 1, "b": (2, 3, 4)}, separator="/")
    {'a': 1, 'b/0': 2, 'b/1': 3, 'b/2': 4}
    >>> flatten_pytree(("a", "b", "c"))
    {'0': 'a', '1': 'b', '2': 'c'}
    """
    leaves_with_path = jax.tree_util.tree_leaves_with_path(pytree, is_leaf=is_leaf)
    flat_pytree = {}
    for path, leave in leaves_with_path:
        key = pytree_key_path_to_str(path, separator=separator)
        if key in flat_pytree:
            raise ValueError(f"Duplicate key found: {key}")
        flat_pytree[key] = leave
    return flat_pytree


def flatten_dict(d: dict | FrozenDict, separator: str = ".") -> dict[str, Any]:
    """
    Flattens a nested dictionary.

    In contrast to flatten_pytree, this function is specifically designed for dictionaries and does not flatten
    sequences by default. It is equivalent to setting the is_leaf function in flatten_pytree to:
    `flatten_pytree(d, is_leaf=lambda x: not isinstance(x, (dict, FrozenDict)))`.

    Args:
        d: Dictionary to be flattened.
        separator: Separator for the keys.

    Returns:
        dict: Flattened dictionary.

    >>> flatten_dict({"a": {"b": 1}, "c": (2, 3, 4)})
    {'a.b': 1, 'c': (2, 3, 4)}
    """
    assert isinstance(
        d, (dict, FrozenDict)
    ), f"Expected a dict or FrozenDict, got {type(d)}. For general PyTrees, use flatten_pytree."
    return flatten_pytree(d, separator=separator, is_leaf=lambda x: not isinstance(x, (dict, FrozenDict)))


def get_shape_dtype_pytree(
    x: PyTree,
) -> PyTree:
    """Converts a PyTree of jax.Array objects to a PyTree of ShapeDtypeStruct objects.

    Leaf nodes of the PyTree that are not jax.Array objects are left unchanged.

    Args:
        x: PyTree of jax.Array objects.

    Returns:
        PyTree of ShapeDtypeStruct objects.
    """
    return jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding) if isinstance(x, jax.Array) else x, x
    )


def delete_arrays_in_pytree(x: PyTree) -> None:
    """Deletes and frees all jax.Array objects in a PyTree from the device memory.

    Leaf nodes of the PyTree that are not jax.Array objects are left unchanged.

    Args:
        x: PyTree of jax.Array objects.
    """

    def _delete_array(x: Any):
        if isinstance(x, jax.Array):
            LOGGER.debug("Delete array of shape", x.shape)
            x.delete()
        else:
            LOGGER.debug("Not deleting object of type", type(x))
        return x

    jax.tree.map(_delete_array, x)
