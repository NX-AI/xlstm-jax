xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.recurrent_triton
=====================================================================

.. py:module:: xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.recurrent_triton


Classes
-------

.. autoapisummary::

   xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.recurrent_triton.mLSTMBackendRecurrentTritonConfig
   xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.recurrent_triton.mLSTMBackendRecurrentTriton


Module Contents
---------------

.. py:class:: mLSTMBackendRecurrentTritonConfig

   Configuration class for the mLSTM recurrent backend using Triton kernels.


   .. py:attribute:: eps
      :type:  float
      :value: 1e-06


      Epsilon value used in the kernel.


   .. py:attribute:: state_dtype
      :type:  str | None
      :value: None


      Data type for the state tensors. If None, the data type is inferred from the input tensors.


   .. py:attribute:: use_scan
      :type:  bool
      :value: False


      Whether to use scan for the recurrent sequence.


   .. py:method:: assign_model_config_params(model_config)


.. py:class:: mLSTMBackendRecurrentTriton

   Bases: :py:obj:`xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.config.mLSTMBackend`


   mLSTM recurrent backend using Triton kernels.

   This backend uses Triton kernels for the mLSTM recurrent cell.

   :param config: Configuration object for the backend.
   :param config_class: Configuration class for the backend.


   .. py:attribute:: config_class


   .. py:property:: can_vmap_over_heads
      :type: bool


      Whether the backend can be vmaped over the heads dimension.

      Triton kernels already handle the head dimension, hence not to be vmaped over.

      :returns: False
      :rtype: bool


