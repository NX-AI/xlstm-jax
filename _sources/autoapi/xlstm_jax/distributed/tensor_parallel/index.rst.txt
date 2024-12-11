xlstm_jax.distributed.tensor_parallel
=====================================

.. py:module:: xlstm_jax.distributed.tensor_parallel


Classes
-------

.. autoapisummary::

   xlstm_jax.distributed.tensor_parallel.ModelParallelismWrapper
   xlstm_jax.distributed.tensor_parallel.TPDense
   xlstm_jax.distributed.tensor_parallel.TPAsyncDense


Functions
---------

.. autoapisummary::

   xlstm_jax.distributed.tensor_parallel.scale_init
   xlstm_jax.distributed.tensor_parallel.async_gather
   xlstm_jax.distributed.tensor_parallel.async_gather_bidirectional
   xlstm_jax.distributed.tensor_parallel.async_gather_split
   xlstm_jax.distributed.tensor_parallel.async_scatter
   xlstm_jax.distributed.tensor_parallel.async_scatter_split


Module Contents
---------------

.. py:class:: ModelParallelismWrapper

   Bases: :py:obj:`flax.linen.Module`


   Wrapper for adding model parallelism to a module.

   This wrapper adds sharding over the model axis to the parameters of the module and initializes the module with
   different parameters across the model axis.

   :param model_axis_name: Name of the model axis to shard over.
   :param module_fn: Function that returns the Flax module to wrap.
   :param mask_except_model_idx: If not None, only the `mask_except_model_idx`-th shard will be non-zero.
   :param split_rngs: If True, split the random number generators across the model axis.
   :param module_kwargs: Additional keyword arguments to pass to the module function.


   .. py:attribute:: model_axis_name
      :type:  str


   .. py:attribute:: module_fn
      :type:  collections.abc.Callable[Ellipsis, flax.linen.Module]


   .. py:attribute:: mask_except_model_idx
      :type:  int | None
      :value: None



   .. py:attribute:: split_rngs
      :type:  bool
      :value: True



   .. py:attribute:: module_kwargs
      :type:  flax.core.frozen_dict.FrozenDict[str, Any]


.. py:function:: scale_init(init_fn, scale_factor = 1.0)

   Scales the output of the given init function by the given factor.

   :param init_fn: The init function to scale.
   :param scale_factor: The factor to scale the output of the init function by.

   :returns: A new init function that scales the output of the given init function by the given factor.


.. py:class:: TPDense

   Bases: :py:obj:`flax.linen.Module`


   Dense layer with Tensor Parallelism support.

   This layer can be used to perform a dense layer with Tensor Parallelism support.

   .. attribute:: dense_fn

      Constructor function of the dense layer to use. Needs to support the keyword argument `kernel_init`.

   .. attribute:: model_axis_name

      The name of the model axis.

   .. attribute:: tp_mode

      The Tensor Parallelism mode to use. Can be "scatter", "gather", or "none".

   .. attribute:: skip_communication

      Whether to skip communication in the Tensor Parallelism strategy. Useful for layers with
      custom communication or where input has been already gathered beforehand.

   .. attribute:: kernel_init

      The initializer to use for the kernel of the dense layer.

   .. attribute:: kernel_init_adjustment

      The adjustment factor to use for the kernel initializer.

   .. attribute:: use_bias

      Whether to use a bias in the dense layer.

   .. attribute:: dense_name

      The name of the dense layer module.


   .. py:attribute:: dense_fn
      :type:  Any


   .. py:attribute:: model_axis_name
      :type:  str


   .. py:attribute:: tp_mode
      :type:  Literal['scatter', 'gather', 'none']
      :value: 'none'



   .. py:attribute:: skip_communication
      :type:  bool
      :value: False



   .. py:attribute:: kernel_init
      :type:  collections.abc.Callable


   .. py:attribute:: kernel_init_adjustment
      :type:  float
      :value: 1.0



   .. py:attribute:: use_bias
      :type:  bool
      :value: True



   .. py:attribute:: dense_name
      :type:  str
      :value: 'module'



.. py:function:: async_gather(x, axis_name, shift_up = True)

   All gather using ring permutation.

   :param x: The input to gather.
   :param axis_name: The axis name to gather along.
   :param shift_up: Whether to shift up (device 0 send to device 1) or down (device 1 send to device 0).

   :returns: List of gathered inputs.


.. py:function:: async_gather_bidirectional(x, axis_name, shift_up = True)

   All gather using ring permutation with bidirectional communication.

   :param x: The input to gather.
   :param axis_name: The axis name to gather along.
   :param shift_up: Whether to return the order of tensors that complies with the unidirectional version of shift up
                    (device 0 send to device 1) or down (device 1 send to device 0).

   :returns: List of gathered inputs.


.. py:function:: async_gather_split(x, axis_name)

   All gather using ring permutation with features split for bidirectional communication.

   :param x: The input to gather.
   :param axis_name: The axis name to gather along.

   :returns: List of gathered inputs. Length is 2 * axis size - 1.


.. py:function:: async_scatter(xs, axis_name, shift_up = True)

   Scatter sum using ring permutation.

   :param xs: The inputs to scatter sum. The length of the list should match the size of the axis.
   :param axis_name: The axis name to scatter sum along.
   :param shift_up: Whether to shift up (device 0 send to device 1) or down (device 1 send to device 0).

   :returns: The scatter summed output.


.. py:function:: async_scatter_split(xs, axis_name)

   Scatter sum using ring permutation with features split for bidirectional communication.

   :param xs: The inputs to scatter sum. The length of the list should match the size of the axis.
   :param axis_name: The axis name to scatter sum along.

   :returns: The scatter summed output.


.. py:class:: TPAsyncDense

   Bases: :py:obj:`flax.linen.Module`


   Tensor-Parallel Dense Layer with Asynchronous Communication.

   This layer can be used to perform a dense layer with Tensor Parallelism support, and overlaps communication with
   computation whenever possible.

   .. attribute:: dense_fn

      Constructor function of the dense layer to use. Needs to support the keyword argument `kernel_init`.

   .. attribute:: model_axis_name

      The name of the model axis.

   .. attribute:: tp_mode

      The Tensor Parallelism mode to use. Can be "scatter", "gather", or "none".

   .. attribute:: kernel_init

      The initializer to use for the kernel of the dense layer.

   .. attribute:: kernel_init_adjustment

      The adjustment factor to use for the kernel initializer.

   .. attribute:: use_bias

      Whether to use a bias in the dense layer.

   .. attribute:: dense_name

      The name of the dense layer module.

   .. attribute:: use_bidirectional_gather

      Whether to use bidirectional or unidirectional gather over the device ring for
      communication.

   .. attribute:: use_bidirectional_scatter

      Whether to use bidirectional or unidirectional scatter over the device ring for
      communication.


   .. py:attribute:: dense_fn
      :type:  Any


   .. py:attribute:: model_axis_name
      :type:  str


   .. py:attribute:: tp_mode
      :type:  Literal['scatter', 'gather', 'none']
      :value: 'none'



   .. py:attribute:: kernel_init
      :type:  collections.abc.Callable


   .. py:attribute:: kernel_init_adjustment
      :type:  float
      :value: 1.0



   .. py:attribute:: use_bias
      :type:  bool
      :value: True



   .. py:attribute:: dense_name
      :type:  str
      :value: 'module'



   .. py:attribute:: use_bidirectional_gather
      :type:  bool
      :value: True



   .. py:attribute:: use_bidirectional_scatter
      :type:  bool
      :value: False



