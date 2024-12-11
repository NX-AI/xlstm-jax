xlstm_jax.models.xlstm_parallel.blocks.mlstm.layer
==================================================

.. py:module:: xlstm_jax.models.xlstm_parallel.blocks.mlstm.layer


Classes
-------

.. autoapisummary::

   xlstm_jax.models.xlstm_parallel.blocks.mlstm.layer.mLSTMLayerConfig
   xlstm_jax.models.xlstm_parallel.blocks.mlstm.layer.mLSTMLayer
   xlstm_jax.models.xlstm_parallel.blocks.mlstm.layer.mLSTMInnerLayer


Module Contents
---------------

.. py:class:: mLSTMLayerConfig

   Bases: :py:obj:`xlstm_jax.models.xlstm_parallel.utils.UpProjConfigMixin`


   Sub-model configuration.

   This class is currently a quick fix to allow for post-init style model configs, like the xlstm-clean we ported from
   the original xlstm codebase. Once the config system is more mature, we should remove this and all becomes a subclass
   of ModelConfig.


   .. py:attribute:: conv1d_kernel_size
      :type:  int
      :value: 4



   .. py:attribute:: qkv_proj_blocksize
      :type:  int
      :value: 4



   .. py:attribute:: num_heads
      :type:  int
      :value: 4



   .. py:attribute:: proj_factor
      :type:  float
      :value: 2.0



   .. py:attribute:: vmap_qk
      :type:  bool
      :value: False



   .. py:attribute:: init_distribution
      :type:  xlstm_jax.models.shared.InitDistribution
      :value: 'normal'


      Distribution type from which to sample the weights.


   .. py:attribute:: output_init_fn
      :type:  xlstm_jax.models.shared.InitFnName
      :value: 'wang'


      Initialization function for the output projection layer.


   .. py:attribute:: layer_type
      :type:  Literal['mlstm', 'mlstm_v1']
      :value: 'mlstm'



   .. py:attribute:: norm_type
      :type:  Literal['layernorm', 'rmsnorm']
      :value: 'layernorm'


      this is only used in the 'mlstm' layer_type.

      :type: Type of normalization layer to use. NOTE


   .. py:attribute:: qk_dim_factor
      :type:  float
      :value: 1.0


      Factor to scale the qk projection dimension by. By default, the qk projection dimension is the same as the
      inner embedding dimension, split into num_heads. This factor is applied to this default size.


   .. py:attribute:: v_dim_factor
      :type:  float
      :value: 1.0


      Factor to scale the v projection dimension by. By default, the v projection dimension is the same as the
      inner embedding dimension, split into num_heads. This factor is applied to this default size.


   .. py:attribute:: embedding_dim
      :type:  int
      :value: -1



   .. py:attribute:: bias
      :type:  bool
      :value: False



   .. py:attribute:: dropout
      :type:  float
      :value: 0.0



   .. py:attribute:: context_length
      :type:  int
      :value: -1



   .. py:attribute:: dtype
      :type:  str
      :value: 'bfloat16'



   .. py:attribute:: parallel
      :type:  xlstm_jax.models.configs.ParallelConfig | None
      :value: None



   .. py:attribute:: debug_cell
      :type:  bool
      :value: False



   .. py:attribute:: gate_input
      :type:  Literal['qkv', 'x_mlstm', 'x_mlstm_conv', 'x_mlstm_conv_act']
      :value: 'qkv'


      Which input to use for the mLSTM cell gates. Options are:
      - "qkv": use the query, key and value vectors concatenated as input. Default, as in paper version.
      - "x_mlstm": use the output of the mLSTM up projection layer. These are the same features that go into
          the V projection.
      - "x_mlstm_conv": use the output of the convolution on the mLSTM up projection features.
      - "x_mlstm_conv_act": use the output of the activation function on the convolution on the mLSTM up projection
          features. These are the same features that go into the QK projection.


   .. py:attribute:: _num_blocks
      :type:  int
      :value: 1



   .. py:attribute:: _inner_embedding_dim
      :type:  int
      :value: None



   .. py:attribute:: mlstm_cell
      :type:  xlstm_jax.models.xlstm_parallel.blocks.mlstm.cell.mLSTMCellConfig


   .. py:property:: _dtype
      :type: jax.numpy.dtype


      Returns the real dtype instead of the str from configs.

      :returns: The jnp dtype corresponding to the string value.


   .. py:attribute:: round_proj_up_dim_up
      :type:  bool
      :value: True



   .. py:attribute:: round_proj_up_to_multiple_of
      :type:  int
      :value: 64



   .. py:attribute:: _proj_up_dim
      :type:  int | None
      :value: None



   .. py:method:: _set_proj_up_dim(embedding_dim)


   .. py:method:: to_dict()

      Converts the config to a dictionary.

      Helpful for saving to disk or logging.



.. py:class:: mLSTMLayer

   Bases: :py:obj:`flax.linen.Module`


   The mLSTM layer with Mamba block style.


   .. py:attribute:: config
      :type:  mLSTMLayerConfig


.. py:class:: mLSTMInnerLayer

   Bases: :py:obj:`flax.linen.Module`


   The inner mLSTM layer with Mamba block style.

   Applies a convolutional layer followed by a mLSTM cell.


   .. py:attribute:: config
      :type:  mLSTMLayerConfig


