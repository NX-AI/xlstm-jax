xlstm_jax.models.xlstm_parallel.components.linear_headwise
==========================================================

.. py:module:: xlstm_jax.models.xlstm_parallel.components.linear_headwise


Classes
-------

.. autoapisummary::

   xlstm_jax.models.xlstm_parallel.components.linear_headwise.LinearHeadwiseExpandConfig
   xlstm_jax.models.xlstm_parallel.components.linear_headwise.LinearHeadwiseExpand


Module Contents
---------------

.. py:class:: LinearHeadwiseExpandConfig

   .. py:attribute:: in_features
      :type:  int
      :value: 0



   .. py:attribute:: num_heads
      :type:  int
      :value: -1



   .. py:attribute:: expand_factor_up
      :type:  float
      :value: 1



   .. py:attribute:: _out_features
      :type:  int
      :value: -1



   .. py:attribute:: bias
      :type:  bool
      :value: True



   .. py:attribute:: trainable_weight
      :type:  bool
      :value: True



   .. py:attribute:: trainable_bias
      :type:  bool
      :value: True



   .. py:attribute:: dtype
      :type:  str
      :value: 'float32'



   .. py:property:: _dtype
      :type: jax.numpy.dtype


      Returns the real dtype instead of the str from configs.

      :returns: The jnp dtype corresponding to the string value.


.. py:class:: LinearHeadwiseExpand

   Bases: :py:obj:`flax.linen.Module`


   This is a structured projection layer that projects the input to a higher dimension.

   It only allows integer up-projection factors, i.e. the output dimension is a multiple of the input dimension.


   .. py:attribute:: config
      :type:  LinearHeadwiseExpandConfig


   .. py:attribute:: kernel_init
      :type:  Any
      :value: None



   .. py:attribute:: bias_init
      :type:  callable


   .. py:method:: extra_repr()


