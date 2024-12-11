xlstm_jax.models.xlstm_pytorch.xlstm_block_stack
================================================

.. py:module:: xlstm_jax.models.xlstm_pytorch.xlstm_block_stack


Classes
-------

.. autoapisummary::

   xlstm_jax.models.xlstm_pytorch.xlstm_block_stack.xLSTMBlockStackConfig
   xlstm_jax.models.xlstm_pytorch.xlstm_block_stack.xLSTMBlockStack


Module Contents
---------------

.. py:class:: xLSTMBlockStackConfig

   .. py:attribute:: mlstm_block
      :type:  xlstm_jax.models.xlstm_pytorch.blocks.mlstm.block.mLSTMBlockConfig | None
      :value: None



   .. py:attribute:: slstm_block
      :type:  xlstm_jax.models.xlstm_pytorch.blocks.slstm.block.sLSTMBlockConfig | None
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



.. py:class:: xLSTMBlockStack(config)

   Bases: :py:obj:`torch.nn.Module`


   .. py:attribute:: config_class


   .. py:attribute:: config


   .. py:attribute:: blocks


   .. py:method:: _create_blocks(config)


   .. py:method:: reset_parameters()


   .. py:method:: forward(x, **kwargs)


   .. py:method:: step(x, state = None)


