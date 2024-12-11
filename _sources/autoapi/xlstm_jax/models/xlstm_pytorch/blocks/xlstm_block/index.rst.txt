xlstm_jax.models.xlstm_pytorch.blocks.xlstm_block
=================================================

.. py:module:: xlstm_jax.models.xlstm_pytorch.blocks.xlstm_block


Classes
-------

.. autoapisummary::

   xlstm_jax.models.xlstm_pytorch.blocks.xlstm_block.xLSTMBlockConfig
   xlstm_jax.models.xlstm_pytorch.blocks.xlstm_block.xLSTMBlock


Module Contents
---------------

.. py:class:: xLSTMBlockConfig

   .. py:attribute:: mlstm
      :type:  xlstm_jax.models.xlstm_pytorch.blocks.mlstm.layer.mLSTMLayerConfig | None
      :value: None



   .. py:attribute:: slstm
      :type:  xlstm_jax.models.xlstm_pytorch.blocks.slstm.layer.sLSTMLayerConfig | None
      :value: None



   .. py:attribute:: feedforward
      :type:  xlstm_jax.models.xlstm_pytorch.components.feedforward.FeedForwardConfig | None
      :value: None



   .. py:attribute:: _num_blocks
      :type:  int | None
      :value: None



   .. py:attribute:: _block_idx
      :type:  int | None
      :value: None



.. py:class:: xLSTMBlock(config)

   Bases: :py:obj:`torch.nn.Module`


   An xLSTM block can be either an sLSTM Block or an mLSTM Block.

   It contains the pre-LayerNorms and the skip connections.


   .. py:attribute:: config_class


   .. py:attribute:: config


   .. py:attribute:: xlstm_norm


   .. py:method:: forward(x, **kwargs)


   .. py:method:: step(x, **kwargs)


   .. py:method:: reset_parameters()


