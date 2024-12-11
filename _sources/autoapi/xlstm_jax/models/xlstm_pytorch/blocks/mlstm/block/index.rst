xlstm_jax.models.xlstm_pytorch.blocks.mlstm.block
=================================================

.. py:module:: xlstm_jax.models.xlstm_pytorch.blocks.mlstm.block


Classes
-------

.. autoapisummary::

   xlstm_jax.models.xlstm_pytorch.blocks.mlstm.block.mLSTMBlockConfig
   xlstm_jax.models.xlstm_pytorch.blocks.mlstm.block.mLSTMBlock


Module Contents
---------------

.. py:class:: mLSTMBlockConfig

   .. py:attribute:: mlstm
      :type:  xlstm_jax.models.xlstm_pytorch.blocks.mlstm.layer.mLSTMLayerConfig


   .. py:attribute:: _num_blocks
      :type:  int | None
      :value: None



   .. py:attribute:: _block_idx
      :type:  int | None
      :value: None



.. py:class:: mLSTMBlock(config)

   Bases: :py:obj:`xlstm_jax.models.xlstm_pytorch.blocks.xlstm_block.xLSTMBlock`


   An xLSTM block can be either an sLSTM Block or an mLSTM Block.

   It contains the pre-LayerNorms and the skip connections.


   .. py:attribute:: config_class


   .. py:attribute:: config


   .. py:attribute:: xlstm_norm


   .. py:method:: forward(x, **kwargs)


   .. py:method:: step(x, **kwargs)


   .. py:method:: reset_parameters()


