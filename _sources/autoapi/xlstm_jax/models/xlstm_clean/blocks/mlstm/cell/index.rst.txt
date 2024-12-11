xlstm_jax.models.xlstm_clean.blocks.mlstm.cell
==============================================

.. py:module:: xlstm_jax.models.xlstm_clean.blocks.mlstm.cell


Classes
-------

.. autoapisummary::

   xlstm_jax.models.xlstm_clean.blocks.mlstm.cell.mLSTMCellConfig
   xlstm_jax.models.xlstm_clean.blocks.mlstm.cell.mLSTMCell


Module Contents
---------------

.. py:class:: mLSTMCellConfig

   .. py:attribute:: context_length
      :type:  int
      :value: -1



   .. py:attribute:: embedding_dim
      :type:  int
      :value: -1



   .. py:attribute:: num_heads
      :type:  int
      :value: -1



   .. py:attribute:: backend
      :type:  xlstm_jax.models.xlstm_clean.blocks.mlstm.backend.mLSTMBackendNameAndKwargs


   .. py:attribute:: dtype
      :type:  str
      :value: 'bfloat16'



   .. py:property:: _dtype
      :type: jax.numpy.dtype


      Returns the real dtype instead of the str from configs.

      :returns: The jnp dtype corresponding to the string value.


.. py:class:: mLSTMCell

   Bases: :py:obj:`flax.linen.Module`


   .. py:attribute:: config
      :type:  mLSTMCellConfig


