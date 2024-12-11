xlstm_jax.kernels.stride_utils
==============================

.. py:module:: xlstm_jax.kernels.stride_utils


Functions
---------

.. autoapisummary::

   xlstm_jax.kernels.stride_utils.get_strides
   xlstm_jax.kernels.stride_utils.get_stride


Module Contents
---------------

.. py:function:: get_strides(array)

   Returns the strides of a JAX array.

   :param array: JAX array or shape-dtype struct.

   :returns: The strides of the array. Length is equal to the number of dimensions.


.. py:function:: get_stride(array, axis)

   Returns the stride of a JAX array at a given axis.

   To calculate all strides, use get_strides.

   :param array: JAX array or shape-dtype struct.
   :param axis: The axis at which to calculate the stride.

   :returns: The stride of the array at the given axis.


