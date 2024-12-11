xlstm_jax.models.xlstm_pytorch.components.conv
==============================================

.. py:module:: xlstm_jax.models.xlstm_pytorch.components.conv


Classes
-------

.. autoapisummary::

   xlstm_jax.models.xlstm_pytorch.components.conv.CausalConv1dConfig
   xlstm_jax.models.xlstm_pytorch.components.conv.CausalConv1d


Functions
---------

.. autoapisummary::

   xlstm_jax.models.xlstm_pytorch.components.conv.conv1d_step


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


.. py:function:: conv1d_step(x, conv_state, conv1d_weight, conv1d_bias = None)

   B: batch size
   S: sequence length
   D: feature dimension
   KS: kernel size
   :param x: (B, S, D)
   :param conv_state: (B, KS, D)
   :param conv1d_weight: (KS, D)
   :param conv1d_bias:
                       (D)


.. py:class:: CausalConv1d(config)

   Bases: :py:obj:`torch.nn.Module`


   .. py:attribute:: config_class

      Causal depth-wise convolution of a time series tensor.

      Input:  Tensor of shape (B,T,F), i.e. (batch, time, feature)
      Output: Tensor of shape (B,T,F)

      :param feature_dim: Number of features in the input tensor.
      :param kernel_size: Size of the kernel for the depth-wise convolution.
      :param causal_conv_bias: Whether to use bias in the depth-wise convolution
      :param channel_mixing: Whether to use channel mixing (i.e. groups=1) or not (i.e. groups=feature_dim).
                             If `True`, it mixes the convolved features across channels.
                             If `False`, all the features are convolved independently.


   .. py:attribute:: config


   .. py:attribute:: groups


   .. py:method:: reset_parameters()


   .. py:method:: _create_weight_decay_optim_groups()


   .. py:method:: forward(x)


   .. py:method:: step(x, conv_state = None)


