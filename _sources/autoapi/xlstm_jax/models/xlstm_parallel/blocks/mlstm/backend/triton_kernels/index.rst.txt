xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.triton_kernels
===================================================================

.. py:module:: xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.triton_kernels


Attributes
----------

.. autoapisummary::

   xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.triton_kernels.BackendNameType


Classes
-------

.. autoapisummary::

   xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.triton_kernels.mLSTMBackendTritonConfig
   xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.triton_kernels.mLSTMBackendTriton


Module Contents
---------------

.. py:data:: BackendNameType

.. py:class:: mLSTMBackendTritonConfig

   .. py:attribute:: autocast_dtype
      :type:  str | None
      :value: None


      Dtype to use for the kernel computation. If None, uses the query dtype.


   .. py:attribute:: chunk_size
      :type:  int
      :value: 64


      Chunk size for the kernel computation.


   .. py:attribute:: reduce_slicing
      :type:  bool
      :value: True


      Whether to reduce slicing operations before the kernel computation.
      Speeds up computation during training, but may limit initial states and
      forwarding states during inference.


   .. py:attribute:: backend_name
      :type:  BackendNameType
      :value: 'max_triton_noslice'


      Backend name for the kernel type used


   .. py:attribute:: eps
      :type:  float
      :value: 1e-06


      Epsilon value used in the kernel


   .. py:attribute:: norm_val
      :type:  float
      :value: 1.0


      Normalizer upper bound value - max(norm_val e^-m, |n q|)


   .. py:attribute:: stabilize_correctly
      :type:  bool
      :value: True


      Whether to stabilize correctly, i.e. scale norm_val with the maximizer state - see above


   .. py:method:: assign_model_config_params(model_config)


.. py:class:: mLSTMBackendTriton

   Bases: :py:obj:`xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.config.mLSTMBackend`


   .. py:attribute:: config_class


   .. py:property:: can_vmap_over_heads
      :type: bool


      Whether the backend can be vmaped over the heads dimension.

      Triton kernels already handle the head dimension, hence not to be vmaped over.

      :returns: False
      :rtype: bool


