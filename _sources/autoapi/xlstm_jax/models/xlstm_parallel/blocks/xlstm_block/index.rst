xlstm_jax.models.xlstm_parallel.blocks.xlstm_block
==================================================

.. py:module:: xlstm_jax.models.xlstm_parallel.blocks.xlstm_block


Classes
-------

.. autoapisummary::

   xlstm_jax.models.xlstm_parallel.blocks.xlstm_block.xLSTMBlockConfig
   xlstm_jax.models.xlstm_parallel.blocks.xlstm_block.ResidualBlock
   xlstm_jax.models.xlstm_parallel.blocks.xlstm_block.xLSTMBlock


Module Contents
---------------

.. py:class:: xLSTMBlockConfig

   .. py:attribute:: mlstm
      :type:  xlstm_jax.models.xlstm_parallel.blocks.mlstm.layer.mLSTMLayerConfig | None
      :value: None



   .. py:attribute:: slstm
      :type:  None
      :value: None



   .. py:attribute:: parallel
      :type:  xlstm_jax.models.configs.ParallelConfig | None
      :value: None



   .. py:attribute:: feedforward
      :type:  xlstm_jax.models.xlstm_parallel.components.feedforward.FeedForwardConfig | None
      :value: None



   .. py:attribute:: dtype
      :type:  str
      :value: 'bfloat16'



   .. py:attribute:: norm_eps
      :type:  float
      :value: 1e-06


      Epsilon value for numerical stability in layer norm.


   .. py:attribute:: norm_type
      :type:  Literal['layernorm', 'rmsnorm']
      :value: 'layernorm'


      Type of normalization layer to use.


   .. py:attribute:: add_post_norm
      :type:  bool
      :value: False


      If True, adds a normalization layer after the mLSTM/sLSTM layer and the feedforward layer.
      Note that this is not the post-norm on the residual connection, but is applied to the output of the layers
      before the residual connection, following e.g. Gemma-2.


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


.. py:class:: ResidualBlock

   Bases: :py:obj:`flax.linen.Module`


   A residual block that applies a list of modules in sequence and adds the input to the output.

   Modules are created within this block to wrap them as children module of this one.


   .. py:attribute:: module_fns
      :type:  list[collections.abc.Callable[Ellipsis, flax.linen.Module]]


.. py:class:: xLSTMBlock

   Bases: :py:obj:`flax.linen.Module`


   An xLSTM block can be either an sLSTM Block or an mLSTM Block.

   It contains the pre-LayerNorms and the skip connections.


   .. py:attribute:: config
      :type:  xLSTMBlockConfig


