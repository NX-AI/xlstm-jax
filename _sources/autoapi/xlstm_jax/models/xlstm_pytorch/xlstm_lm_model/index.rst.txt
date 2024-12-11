xlstm_jax.models.xlstm_pytorch.xlstm_lm_model
=============================================

.. py:module:: xlstm_jax.models.xlstm_pytorch.xlstm_lm_model


Classes
-------

.. autoapisummary::

   xlstm_jax.models.xlstm_pytorch.xlstm_lm_model.xLSTMLMModelConfig
   xlstm_jax.models.xlstm_pytorch.xlstm_lm_model.xLSTMLMModel


Module Contents
---------------

.. py:class:: xLSTMLMModelConfig

   Bases: :py:obj:`xlstm_jax.models.xlstm_pytorch.xlstm_block_stack.xLSTMBlockStackConfig`


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



.. py:class:: xLSTMLMModel(config, **kwargs)

   Bases: :py:obj:`xlstm_jax.models.xlstm_pytorch.utils.WeightDecayOptimGroupMixin`, :py:obj:`torch.nn.Module`


   Helper class that provides a standard way to create an ABC using
   inheritance.


   .. py:attribute:: config_class


   .. py:attribute:: config


   .. py:attribute:: xlstm_block_stack


   .. py:attribute:: token_embedding


   .. py:attribute:: emb_dropout


   .. py:attribute:: lm_head


   .. py:method:: reset_parameters()


   .. py:method:: forward(idx)


   .. py:method:: step(idx, state = None, **kwargs)


   .. py:method:: _create_weight_decay_optim_groups(**kwargs)

      Return a tuple of two sequences, one for parameters with weight decay and one for parameters without weight
      decay.

      Default separation:
      - weight decay: all parameters which have > 1 dimensions.
      - no weight decay: all parameters which have = 1 dimension, e.g. biases.



   .. py:method:: get_weight_decay_optim_groups()

      Return a tuple of two sequences, one for parameters with weight decay and one for parameters without weight
      decay.

      Performs checks to ensure that each parameter is only in one of the two sequences.



   .. py:method:: get_weight_decay_optim_group_param_names()

      Return a tuple of two sequences, one for parameter names with weight decay and one for parameter names without
      weight decay.

      Performs checks to ensure that each parameter is only in one of the two sequences.



   .. py:method:: _get_weight_decay_optim_groups_for_modules(modules, **kwargs)
      :staticmethod:



