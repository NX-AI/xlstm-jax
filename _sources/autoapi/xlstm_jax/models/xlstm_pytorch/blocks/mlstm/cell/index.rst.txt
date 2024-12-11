xlstm_jax.models.xlstm_pytorch.blocks.mlstm.cell
================================================

.. py:module:: xlstm_jax.models.xlstm_pytorch.blocks.mlstm.cell


Classes
-------

.. autoapisummary::

   xlstm_jax.models.xlstm_pytorch.blocks.mlstm.cell.mLSTMCellConfig
   xlstm_jax.models.xlstm_pytorch.blocks.mlstm.cell.mLSTMCell


Module Contents
---------------

.. py:class:: mLSTMCellConfig

   .. py:attribute:: context_length
      :type:  int
      :value: -1



   .. py:attribute:: embedding_dim
      :type:  int
      :value: -1



   .. py:attribute:: num_heads
      :type:  int
      :value: -1



   .. py:attribute:: backend
      :type:  xlstm_jax.models.xlstm_pytorch.blocks.mlstm.backend.mLSTMBackendNameAndKwargs


.. py:class:: mLSTMCell(config)

   Bases: :py:obj:`torch.nn.Module`


   .. py:attribute:: config_class


   .. py:attribute:: config


   .. py:attribute:: backend_fn


   .. py:attribute:: backend_fn_step


   .. py:attribute:: igate


   .. py:attribute:: fgate


   .. py:attribute:: outnorm


   .. py:method:: forward(q, k, v)


   .. py:method:: step(q, k, v, mlstm_state = None)


   .. py:method:: reset_parameters()


