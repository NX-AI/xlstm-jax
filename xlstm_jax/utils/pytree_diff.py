from collections.abc import Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

PyTree = Any


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
                diff = a - b
            except ValueError:
                return {"array_difference": (a, b)}
            if isinstance(diff, jax.Array):
                return diff if not np.allclose(diff, jnp.zeros_like(diff)) else None
            else:
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
        elif (
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
        else:
            return diff_fn(t1, t2)

    diff_tree = recursive_diff(tree1, tree2)
    return diff_tree if diff_tree else None
