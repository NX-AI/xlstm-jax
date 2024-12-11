xlstm_jax.models.xlstm_parallel.components.init
===============================================

.. py:module:: xlstm_jax.models.xlstm_parallel.components.init


Functions
---------

.. autoapisummary::

   xlstm_jax.models.xlstm_parallel.components.init.bias_linspace_init


Module Contents
---------------

.. py:function:: bias_linspace_init(start, end, axis_name = None)

   Linearly spaced bias init across dimensions.

   Only supports 1D array shapes. Array values are including start and end. If axis name is provided, the linspace
   is sharded over the axis.

   :param start: Start value for the linspace.
   :param end: End value for the linspace.
   :param axis_name: Optional axis name to shard over.

   :returns: Initializer function that creates a 1D array with linearly spaced values between start and end.


