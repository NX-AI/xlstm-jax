xlstm_jax.models.xlstm_parallel.xlstm_block_stack
=================================================

.. py:module:: xlstm_jax.models.xlstm_parallel.xlstm_block_stack


Classes
-------

.. autoapisummary::

   xlstm_jax.models.xlstm_parallel.xlstm_block_stack.xLSTMBlockStackConfig
   xlstm_jax.models.xlstm_parallel.xlstm_block_stack.xLSTMBlockStack
   xlstm_jax.models.xlstm_parallel.xlstm_block_stack.BlockStack


Module Contents
---------------

.. py:class:: xLSTMBlockStackConfig

   Bases: :py:obj:`xlstm_jax.models.configs.SubModelConfig`


   Sub-model configuration.

   This class is currently a quick fix to allow for post-init style model configs, like the xlstm-clean we ported from
   the original xlstm codebase. Once the config system is more mature, we should remove this and all becomes a subclass
   of ModelConfig.


   .. py:attribute:: mlstm_block
      :type:  xlstm_jax.models.xlstm_parallel.blocks.mlstm.block.mLSTMBlockConfig | None
      :value: None



   .. py:attribute:: slstm_block
      :type:  Any | None
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



   .. py:attribute:: scan_blocks
      :type:  bool
      :value: False



   .. py:attribute:: dtype
      :type:  str
      :value: 'bfloat16'



   .. py:attribute:: parallel
      :type:  xlstm_jax.models.configs.ParallelConfig | None
      :value: None



   .. py:attribute:: init_distribution_embed
      :type:  xlstm_jax.models.shared.InitDistribution
      :value: 'normal'


      Distribution type from which to sample the embeddings.


   .. py:attribute:: init_distribution_out
      :type:  xlstm_jax.models.shared.InitDistribution
      :value: 'normal'


      Distribution type from which to sample the LM output head.


   .. py:attribute:: slstm_at
      :type:  list[int]
      :value: []



   .. py:attribute:: _block_map
      :type:  str | None
      :value: None



   .. py:property:: block_map
      :type: list[int]



   .. py:method:: _create_block_map()

      Creates the block map, that specifies which block is used at which position.



   .. py:property:: _dtype
      :type: jax.numpy.dtype


      Returns the real dtype instead of the str from configs.

      :returns: The jnp dtype corresponding to the string value.


   .. py:method:: to_dict()

      Converts the config to a dictionary.

      Helpful for saving to disk or logging.



.. py:class:: xLSTMBlockStack

   Bases: :py:obj:`flax.linen.Module`


   .. py:attribute:: config
      :type:  xLSTMBlockStackConfig


.. py:class:: BlockStack

   Bases: :py:obj:`flax.linen.Module`


   .. py:attribute:: config
      :type:  xLSTMBlockStackConfig


   .. py:method:: _create_blocks(config)


