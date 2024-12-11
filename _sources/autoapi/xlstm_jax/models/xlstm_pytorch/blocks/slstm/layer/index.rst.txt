xlstm_jax.models.xlstm_pytorch.blocks.slstm.layer
=================================================

.. py:module:: xlstm_jax.models.xlstm_pytorch.blocks.slstm.layer


Classes
-------

.. autoapisummary::

   xlstm_jax.models.xlstm_pytorch.blocks.slstm.layer.sLSTMLayerConfig
   xlstm_jax.models.xlstm_pytorch.blocks.slstm.layer.sLSTMLayer


Module Contents
---------------

.. py:class:: sLSTMLayerConfig

   Bases: :py:obj:`xlstm_jax.models.xlstm_pytorch.blocks.slstm.cell.sLSTMCellConfig`


   .. py:attribute:: embedding_dim
      :type:  int
      :value: -1



   .. py:attribute:: num_heads
      :type:  int
      :value: 4



   .. py:attribute:: conv1d_kernel_size
      :type:  int
      :value: 4



   .. py:attribute:: group_norm_weight
      :type:  bool
      :value: True



   .. py:attribute:: dropout
      :type:  float
      :value: 0.0



.. py:class:: sLSTMLayer(config)

   Bases: :py:obj:`torch.nn.Module`


   .. py:attribute:: config_class


   .. py:attribute:: config


   .. py:attribute:: fgate


   .. py:attribute:: igate


   .. py:attribute:: zgate


   .. py:attribute:: ogate


   .. py:attribute:: slstm_cell


   .. py:attribute:: group_norm


   .. py:attribute:: dropout


   .. py:method:: reset_parameters()


   .. py:method:: forward(x, initial_state = None, return_last_state=False)


