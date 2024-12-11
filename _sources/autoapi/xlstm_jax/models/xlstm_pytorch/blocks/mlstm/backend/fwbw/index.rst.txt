xlstm_jax.models.xlstm_pytorch.blocks.mlstm.backend.fwbw
========================================================

.. py:module:: xlstm_jax.models.xlstm_pytorch.blocks.mlstm.backend.fwbw


Classes
-------

.. autoapisummary::

   xlstm_jax.models.xlstm_pytorch.blocks.mlstm.backend.fwbw.mLSTMfwbwConfig
   xlstm_jax.models.xlstm_pytorch.blocks.mlstm.backend.fwbw.mLSTMfwbw


Functions
---------

.. autoapisummary::

   xlstm_jax.models.xlstm_pytorch.blocks.mlstm.backend.fwbw.rev_cumsum_off
   xlstm_jax.models.xlstm_pytorch.blocks.mlstm.backend.fwbw.rev_cumsum
   xlstm_jax.models.xlstm_pytorch.blocks.mlstm.backend.fwbw.causal_forget_matrix
   xlstm_jax.models.xlstm_pytorch.blocks.mlstm.backend.fwbw.mLSTMTorchFunction


Module Contents
---------------

.. py:function:: rev_cumsum_off(x)

.. py:function:: rev_cumsum(x)

.. py:function:: causal_forget_matrix(forget_gates)

.. py:class:: mLSTMfwbwConfig

   .. py:attribute:: chunk_size
      :type:  int | None
      :value: 512



   .. py:attribute:: return_state
      :type:  bool
      :value: False



   .. py:attribute:: use_initial_state
      :type:  bool
      :value: False



   .. py:attribute:: keep_G
      :type:  bool
      :value: False



   .. py:attribute:: keep_gates
      :type:  bool
      :value: True



   .. py:attribute:: keep_M
      :type:  bool
      :value: False



   .. py:attribute:: keep_c
      :type:  bool
      :value: False



   .. py:attribute:: stabilize_correctly
      :type:  bool
      :value: False



   .. py:attribute:: scale
      :value: None



   .. py:attribute:: device_type
      :type:  str
      :value: 'cuda'



   .. py:method:: assign_model_config_params(model_config)


.. py:function:: mLSTMTorchFunction(config)

   Returns an autograd function that computes the gradient itself.

   config: mLSTMfwbwConfig     Configuration for mLSTMTorchFunc


.. py:class:: mLSTMfwbw(config)

   Bases: :py:obj:`torch.nn.Module`


   .. py:attribute:: config_class


   .. py:attribute:: config


   .. py:attribute:: func


   .. py:method:: forward(*args)


