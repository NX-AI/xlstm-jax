xlstm_jax.models.xlstm_clean.blocks.xlstm_block
===============================================

.. py:module:: xlstm_jax.models.xlstm_clean.blocks.xlstm_block


Classes
-------

.. autoapisummary::

   xlstm_jax.models.xlstm_clean.blocks.xlstm_block.xLSTMBlockConfig
   xlstm_jax.models.xlstm_clean.blocks.xlstm_block.xLSTMBlock


Module Contents
---------------

.. py:class:: xLSTMBlockConfig

   .. py:attribute:: mlstm
      :type:  xlstm_jax.models.xlstm_clean.blocks.mlstm.layer.mLSTMLayerConfig | None
      :value: None



   .. py:attribute:: slstm
      :type:  None
      :value: None



   .. py:attribute:: feedforward
      :type:  xlstm_jax.models.xlstm_clean.components.feedforward.FeedForwardConfig | None
      :value: None



   .. py:attribute:: dtype
      :type:  str
      :value: 'bfloat16'



   .. py:attribute:: _num_blocks
      :type:  int | None
      :value: None



   .. py:attribute:: _block_idx
      :type:  int | None
      :value: None



   .. py:property:: _dtype
      :type: jax.numpy.dtype


      Returns the real dtype instead of the str from configs.

      :returns: The jnp dtype corresponding to the string value.


.. py:class:: xLSTMBlock

   Bases: :py:obj:`flax.linen.Module`


   An xLSTM block can be either an sLSTM Block or an mLSTM Block.

   It contains the pre-LayerNorms and the skip connections.


   .. py:attribute:: config
      :type:  xLSTMBlockConfig


