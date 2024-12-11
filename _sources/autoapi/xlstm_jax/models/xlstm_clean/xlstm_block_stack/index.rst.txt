xlstm_jax.models.xlstm_clean.xlstm_block_stack
==============================================

.. py:module:: xlstm_jax.models.xlstm_clean.xlstm_block_stack


Classes
-------

.. autoapisummary::

   xlstm_jax.models.xlstm_clean.xlstm_block_stack.xLSTMBlockStackConfig
   xlstm_jax.models.xlstm_clean.xlstm_block_stack.xLSTMBlockStack
   xlstm_jax.models.xlstm_clean.xlstm_block_stack.BlockStack


Module Contents
---------------

.. py:class:: xLSTMBlockStackConfig

   .. py:attribute:: mlstm_block
      :type:  xlstm_jax.models.xlstm_clean.blocks.mlstm.block.mLSTMBlockConfig | None
      :value: None



   .. py:attribute:: slstm_block
      :type:  None
      :value: None



   .. py:attribute:: context_length
      :type:  int
      :value: -1



   .. py:attribute:: num_blocks
      :type:  int
      :value: 1



   .. py:attribute:: embedding_dim
      :type:  int
      :value: 128



   .. py:attribute:: add_post_blocks_norm
      :type:  bool
      :value: True



   .. py:attribute:: bias
      :type:  bool
      :value: False



   .. py:attribute:: dropout
      :type:  float
      :value: 0.0



   .. py:attribute:: dtype
      :type:  str
      :value: 'bfloat16'



   .. py:attribute:: slstm_at
      :type:  list[int] | Literal['all']
      :value: []



   .. py:attribute:: _block_map
      :type:  str
      :value: None



   .. py:property:: block_map
      :type: list[int]



   .. py:method:: _create_block_map()

      Creates the block map, that specifies which block is used at which position.



   .. py:property:: _dtype
      :type: jax.numpy.dtype


      Returns the real dtype instead of the str from configs.

      :returns: The jnp dtype corresponding to the string value.


.. py:class:: xLSTMBlockStack

   Bases: :py:obj:`flax.linen.Module`


   .. py:attribute:: config
      :type:  xLSTMBlockStackConfig


.. py:class:: BlockStack

   Bases: :py:obj:`flax.linen.Module`


   .. py:attribute:: config
      :type:  xLSTMBlockStackConfig


   .. py:method:: _create_blocks(config)


