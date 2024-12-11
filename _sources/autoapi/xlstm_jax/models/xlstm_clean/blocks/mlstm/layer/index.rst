xlstm_jax.models.xlstm_clean.blocks.mlstm.layer
===============================================

.. py:module:: xlstm_jax.models.xlstm_clean.blocks.mlstm.layer


Classes
-------

.. autoapisummary::

   xlstm_jax.models.xlstm_clean.blocks.mlstm.layer.mLSTMLayerConfig
   xlstm_jax.models.xlstm_clean.blocks.mlstm.layer.mLSTMLayer


Module Contents
---------------

.. py:class:: mLSTMLayerConfig

   Bases: :py:obj:`xlstm_jax.models.xlstm_clean.utils.UpProjConfigMixin`


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



   .. py:attribute:: _num_blocks
      :type:  int
      :value: 1



   .. py:attribute:: _inner_embedding_dim
      :type:  int
      :value: None



   .. py:attribute:: mlstm_cell
      :type:  xlstm_jax.models.xlstm_clean.blocks.mlstm.cell.mLSTMCellConfig


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
      :type:  int
      :value: None



   .. py:method:: _set_proj_up_dim(embedding_dim)


.. py:class:: mLSTMLayer

   Bases: :py:obj:`flax.linen.Module`


   .. py:attribute:: config
      :type:  mLSTMLayerConfig


