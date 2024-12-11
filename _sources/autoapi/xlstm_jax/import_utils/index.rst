xlstm_jax.import_utils
======================

.. py:module:: xlstm_jax.import_utils


Functions
---------

.. autoapisummary::

   xlstm_jax.import_utils.resolve_import
   xlstm_jax.import_utils.resolve_import_from_string
   xlstm_jax.import_utils.class_to_name


Module Contents
---------------

.. py:function:: resolve_import(import_path)

   Resolves an import from a string or returns the input.

   :param import_path: The import path or the object itself.
   :type import_path: str | Any

   :returns: The resolved object.
   :rtype: Any


.. py:function:: resolve_import_from_string(import_string)

   Resolves an import from a string.

   :param import_string: The import string.
   :type import_string: str

   :returns: The resolved object.
   :rtype: Any


.. py:function:: class_to_name(x)

   Converts a class to a string representation.

   Useful for logging/saving the class name.

   :param x: The input object.
   :type x: Any

   :returns: The string representation of the object.
   :rtype: str | Any


