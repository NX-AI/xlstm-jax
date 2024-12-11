xlstm_jax.utils.pytree_utils
============================

.. py:module:: xlstm_jax.utils.pytree_utils


Attributes
----------

.. autoapisummary::

   xlstm_jax.utils.pytree_utils.LOGGER


Classes
-------

.. autoapisummary::

   xlstm_jax.utils.pytree_utils.RecursionLimit


Functions
---------

.. autoapisummary::

   xlstm_jax.utils.pytree_utils.pytree_diff
   xlstm_jax.utils.pytree_utils.pytree_key_path_to_str
   xlstm_jax.utils.pytree_utils.flatten_pytree
   xlstm_jax.utils.pytree_utils.flatten_dict
   xlstm_jax.utils.pytree_utils.get_shape_dtype_pytree
   xlstm_jax.utils.pytree_utils.delete_arrays_in_pytree


Module Contents
---------------

.. py:data:: LOGGER

.. py:class:: RecursionLimit

.. py:function:: pytree_diff(tree1, tree2)

   Computes the difference between two PyTrees.

   :param tree1: First PyTree.
   :param tree2: Second PyTree.

   :returns: A PyTree of the same structure, with only differing leaves.
             Returns None if no differences are found.

   >>> pytree_diff({"a": 1}, {"a": 2})
   {'a': (1, 2)}
   >>> pytree_diff({"a": 1}, {"a": 1})
   >>> pytree_diff([1, 2, 3], [1, 2])
   {'length_mismatch': (3, 2)}
   >>> pytree_diff(np.array([1, 2, 3]), np.array([1, 2]))
   {'shape_mismatch': ((3,), (2,))}


.. py:function:: pytree_key_path_to_str(path, separator = '.')

   Converts a path to a string.

   An adjusted version of jax.tree_util.keystr to support different separators and easier to read output.

   :param path: Path.
   :param separator: Separator for the keys.

   :returns: Path as string.
   :rtype: str


.. py:function:: flatten_pytree(pytree, separator = '.', is_leaf = None)

   Flattens a PyTree into a dict.

   Supports PyTrees with nested dictionaries, lists, tuples, and more. The keys are created by concatenating the
   path to the leaf with the separator. For sequences, the index is used as key (see examples below).

   :param pytree: PyTree to be flattened.
   :param separator: Separator for the keys.
   :param is_leaf: Function that determines if a node is a leaf. If None, uses default PyTree leaf detection.

   :returns: Flattened PyTree. In case of duplicate keys, a ValueError is raised.
   :rtype: dict

   >>> flatten_pytree({"a": 1, "b": {"c": 2}})
   {'a': 1, 'b.c': 2}
   >>> flatten_pytree({"a": 1, "b": (2, 3, 4)}, separator="/")
   {'a': 1, 'b/0': 2, 'b/1': 3, 'b/2': 4}
   >>> flatten_pytree(("a", "b", "c"))
   {'0': 'a', '1': 'b', '2': 'c'}


.. py:function:: flatten_dict(d, separator = '.')

   Flattens a nested dictionary.

   In contrast to flatten_pytree, this function is specifically designed for dictionaries and does not flatten
   sequences by default. It is equivalent to setting the is_leaf function in flatten_pytree to:
   `flatten_pytree(d, is_leaf=lambda x: not isinstance(x, (dict, FrozenDict)))`.

   :param d: Dictionary to be flattened.
   :param separator: Separator for the keys.

   :returns: Flattened dictionary.
   :rtype: dict

   >>> flatten_dict({"a": {"b": 1}, "c": (2, 3, 4)})
   {'a.b': 1, 'c': (2, 3, 4)}


.. py:function:: get_shape_dtype_pytree(x)

   Converts a PyTree of jax.Array objects to a PyTree of ShapeDtypeStruct objects.

   Leaf nodes of the PyTree that are not jax.Array objects are left unchanged.

   :param x: PyTree of jax.Array objects.

   :returns: PyTree of ShapeDtypeStruct objects.


.. py:function:: delete_arrays_in_pytree(x)

   Deletes and frees all jax.Array objects in a PyTree from the device memory.

   Leaf nodes of the PyTree that are not jax.Array objects are left unchanged.

   :param x: PyTree of jax.Array objects.


