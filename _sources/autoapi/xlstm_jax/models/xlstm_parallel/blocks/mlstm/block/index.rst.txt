xlstm_jax.models.xlstm_parallel.blocks.mlstm.block
==================================================

.. py:module:: xlstm_jax.models.xlstm_parallel.blocks.mlstm.block


Classes
-------

.. autoapisummary::

   xlstm_jax.models.xlstm_parallel.blocks.mlstm.block.mLSTMBlockConfig


Functions
---------

.. autoapisummary::

   xlstm_jax.models.xlstm_parallel.blocks.mlstm.block.get_partial_mLSTMBlock
   xlstm_jax.models.xlstm_parallel.blocks.mlstm.block.mLSTMBlock


Module Contents
---------------

.. py:class:: mLSTMBlockConfig

   Bases: :py:obj:`xlstm_jax.models.configs.SubModelConfig`


   Sub-model configuration.

   This class is currently a quick fix to allow for post-init style model configs, like the xlstm-clean we ported from
   the original xlstm codebase. Once the config system is more mature, we should remove this and all becomes a subclass
   of ModelConfig.


   .. py:attribute:: mlstm
      :type:  xlstm_jax.models.xlstm_parallel.blocks.mlstm.layer.mLSTMLayerConfig


   .. py:attribute:: feedforward
      :type:  xlstm_jax.models.xlstm_parallel.components.feedforward.FeedForwardConfig | None
      :value: None



   .. py:attribute:: add_post_norm
      :type:  bool
      :value: False


      If True, adds a normalization layer after the mLSTM layer and the feedforward layer.


   .. py:attribute:: parallel
      :type:  xlstm_jax.models.configs.ParallelConfig | None
      :value: None


      Parallel configuration for the model.


   .. py:attribute:: _num_blocks
      :type:  int | None
      :value: None



   .. py:attribute:: _block_idx
      :type:  int | None
      :value: None



   .. py:method:: to_dict()

      Converts the config to a dictionary.

      Helpful for saving to disk or logging.



.. py:function:: get_partial_mLSTMBlock(config, *args, **kwargs)

.. py:function:: mLSTMBlock(config, *args, **kwargs)

