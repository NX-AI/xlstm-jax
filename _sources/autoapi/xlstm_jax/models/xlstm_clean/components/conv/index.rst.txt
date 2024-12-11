xlstm_jax.models.xlstm_clean.components.conv
============================================

.. py:module:: xlstm_jax.models.xlstm_clean.components.conv


Classes
-------

.. autoapisummary::

   xlstm_jax.models.xlstm_clean.components.conv.CausalConv1dConfig
   xlstm_jax.models.xlstm_clean.components.conv.CausalConv1d


Module Contents
---------------

.. py:class:: CausalConv1dConfig

   .. py:attribute:: feature_dim
      :type:  int
      :value: None



   .. py:attribute:: kernel_size
      :type:  int
      :value: 4



   .. py:attribute:: causal_conv_bias
      :type:  bool
      :value: True



   .. py:attribute:: channel_mixing
      :type:  bool
      :value: False



   .. py:attribute:: conv1d_kwargs
      :type:  dict


   .. py:attribute:: dtype
      :type:  str
      :value: 'bfloat16'



   .. py:property:: _dtype
      :type: jax.numpy.dtype


      Returns the real dtype instead of the str from configs.

      :returns: The jnp dtype corresponding to the string value.


.. py:class:: CausalConv1d

   Bases: :py:obj:`flax.linen.Module`


   .. py:attribute:: config
      :type:  CausalConv1dConfig

      Implements causal depthwise convolution of a time series tensor.
      Input:  Tensor of shape (B,T,F), i.e. (batch, time, feature)
      Output: Tensor of shape (B,T,F)

      :param feature_dim: number of features in the input tensor
      :param kernel_size: size of the kernel for the depthwise convolution
      :param causal_conv_bias: whether to use bias in the depthwise convolution
      :param channel_mixing: whether to use channel mixing (i.e. groups=1) or not (i.e. groups=feature_dim)
                             If True, it mixes the convolved features across channels.
                             If False, all the features are convolved independently.


