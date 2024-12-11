xlstm_jax.models.xlstm_pytorch.blocks.mlstm.layer
=================================================

.. py:module:: xlstm_jax.models.xlstm_pytorch.blocks.mlstm.layer


Classes
-------

.. autoapisummary::

   xlstm_jax.models.xlstm_pytorch.blocks.mlstm.layer.mLSTMLayerConfig
   xlstm_jax.models.xlstm_pytorch.blocks.mlstm.layer.mLSTMLayer


Module Contents
---------------

.. py:class:: mLSTMLayerConfig

   Bases: :py:obj:`xlstm_jax.models.xlstm_pytorch.utils.UpProjConfigMixin`


   .. py:attribute:: conv1d_kernel_size
      :type:  int
      :value: 4



   .. py:attribute:: qkv_proj_blocksize
      :type:  int
      :value: 4



   .. py:attribute:: num_heads
      :type:  int
      :value: 4



   .. py:attribute:: proj_factor
      :type:  float
      :value: 2.0



   .. py:attribute:: embedding_dim
      :type:  int
      :value: -1



   .. py:attribute:: bias
      :type:  bool
      :value: False



   .. py:attribute:: dropout
      :type:  float
      :value: 0.0



   .. py:attribute:: context_length
      :type:  int
      :value: -1



   .. py:attribute:: _num_blocks
      :type:  int
      :value: 1



   .. py:attribute:: _inner_embedding_dim
      :type:  int
      :value: None



   .. py:attribute:: mlstm_cell
      :type:  xlstm_jax.models.xlstm_pytorch.blocks.mlstm.cell.mLSTMCellConfig


   .. py:attribute:: round_proj_up_dim_up
      :type:  bool
      :value: True



   .. py:attribute:: round_proj_up_to_multiple_of
      :type:  int
      :value: 64



   .. py:attribute:: _proj_up_dim
      :type:  int
      :value: None



   .. py:method:: _set_proj_up_dim(embedding_dim)


.. py:class:: mLSTMLayer(config)

   Bases: :py:obj:`torch.nn.Module`


   .. py:attribute:: config_class


   .. py:attribute:: config


   .. py:attribute:: proj_up


   .. py:attribute:: q_proj


   .. py:attribute:: k_proj


   .. py:attribute:: v_proj


   .. py:attribute:: conv1d


   .. py:attribute:: conv_act_fn


   .. py:attribute:: mlstm_cell


   .. py:attribute:: ogate_act_fn


   .. py:attribute:: learnable_skip


   .. py:attribute:: proj_down


   .. py:attribute:: dropout


   .. py:method:: forward(x)


   .. py:method:: step(x, mlstm_state = None, conv_state = None)


   .. py:method:: reset_parameters()


