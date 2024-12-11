xlstm_jax.models.xlstm_parallel.xlstm_lm_model
==============================================

.. py:module:: xlstm_jax.models.xlstm_parallel.xlstm_lm_model


Classes
-------

.. autoapisummary::

   xlstm_jax.models.xlstm_parallel.xlstm_lm_model.xLSTMLMModelConfig
   xlstm_jax.models.xlstm_parallel.xlstm_lm_model.xLSTMLMModel


Module Contents
---------------

.. py:class:: xLSTMLMModelConfig

   Bases: :py:obj:`xlstm_jax.models.xlstm_parallel.xlstm_block_stack.xLSTMBlockStackConfig`


   Sub-model configuration.

   This class is currently a quick fix to allow for post-init style model configs, like the xlstm-clean we ported from
   the original xlstm codebase. Once the config system is more mature, we should remove this and all becomes a subclass
   of ModelConfig.


   .. py:attribute:: vocab_size
      :type:  int
      :value: -1



   .. py:attribute:: tie_weights
      :type:  bool
      :value: False



   .. py:attribute:: weight_decay_on_embedding
      :type:  bool
      :value: False



   .. py:attribute:: add_embedding_dropout
      :type:  bool
      :value: False



   .. py:attribute:: norm_eps
      :type:  float
      :value: 1e-06


      Epsilon value for numerical stability in normalization layer.


   .. py:attribute:: norm_type
      :type:  Literal['layernorm', 'rmsnorm']
      :value: 'layernorm'


      Type of normalization layer to use.


   .. py:attribute:: logits_soft_cap
      :type:  float | None
      :value: None


      Soft cap for the LM output logits. If None, no cap is applied.


   .. py:attribute:: lm_head_dtype
      :type:  str
      :value: 'float32'


      Data type to perform the LM Head Dense layer in. The output will always be casted to float32 for numerical
      stability.


   .. py:attribute:: parallel
      :type:  xlstm_jax.models.configs.ParallelConfig | None
      :value: None



   .. py:property:: _lm_head_dtype
      :type: jax.numpy.dtype


      Return the real dtype instead of the str in config.

      :returns: Dtype corresponding to the respective str attribute.


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



.. py:class:: xLSTMLMModel

   Bases: :py:obj:`flax.linen.Module`


   .. py:attribute:: config
      :type:  xLSTMLMModelConfig


