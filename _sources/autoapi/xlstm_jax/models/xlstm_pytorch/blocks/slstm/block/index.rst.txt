xlstm_jax.models.xlstm_pytorch.blocks.slstm.block
=================================================

.. py:module:: xlstm_jax.models.xlstm_pytorch.blocks.slstm.block


Classes
-------

.. autoapisummary::

   xlstm_jax.models.xlstm_pytorch.blocks.slstm.block.sLSTMBlockConfig
   xlstm_jax.models.xlstm_pytorch.blocks.slstm.block.sLSTMBlock


Module Contents
---------------

.. py:class:: sLSTMBlockConfig

   .. py:attribute:: slstm
      :type:  xlstm_jax.models.xlstm_pytorch.blocks.slstm.layer.sLSTMLayerConfig


   .. py:attribute:: feedforward
      :type:  xlstm_jax.models.xlstm_pytorch.components.feedforward.FeedForwardConfig | None


   .. py:attribute:: _num_blocks
      :type:  int | None
      :value: None



   .. py:attribute:: _block_idx
      :type:  int | None
      :value: None



.. py:class:: sLSTMBlock(config)

   Bases: :py:obj:`xlstm_jax.models.xlstm_pytorch.blocks.xlstm_block.xLSTMBlock`


   An xLSTM block can be either an sLSTM Block or an mLSTM Block.

   It contains the pre-LayerNorms and the skip connections.


   .. py:attribute:: config_class


   .. py:attribute:: config


   .. py:attribute:: xlstm_norm


   .. py:method:: forward(x, **kwargs)


   .. py:method:: step(x, **kwargs)


   .. py:method:: reset_parameters()


