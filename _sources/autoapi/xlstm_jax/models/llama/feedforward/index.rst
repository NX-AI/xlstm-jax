xlstm_jax.models.llama.feedforward
==================================

.. py:module:: xlstm_jax.models.llama.feedforward


Classes
-------

.. autoapisummary::

   xlstm_jax.models.llama.feedforward.FeedForwardConfig
   xlstm_jax.models.llama.feedforward.FeedForward


Module Contents
---------------

.. py:class:: FeedForwardConfig

   Bases: :py:obj:`xlstm_jax.models.configs.SubModelConfig`


   Configuration for the feedforward network.


   .. py:attribute:: multiple_of
      :type:  int
      :value: 64


      The hidden dimension of the feedforward network will be increased to a multiple of this value. This is useful for
      ensuring an efficient use of the hardware, e.g. for tensor cores.


   .. py:attribute:: ffn_dim_multiplier
      :type:  float
      :value: 1.0


      Multiplier for the hidden dimension of the feedforward network. By default, the hidden dimension is up to 8/3 of
      the input dimension. This multiplier is applied to this default size and can be used to increase or decrease the
      hidden dimension. This is in line with the original PyTorch Llama implementation.


   .. py:attribute:: use_bias
      :type:  bool
      :value: False


      Whether to use bias in the feedforward network.


   .. py:attribute:: dropout_rate
      :type:  float
      :value: 0.0


      Dropout rate for the feedforward network.


   .. py:attribute:: num_layers
      :type:  int
      :value: 12


      Number of layers in the whole Llama Transformer model. Used for initialization.


   .. py:attribute:: dtype
      :type:  str
      :value: 'float32'


      Data type of the activations in the network.


   .. py:attribute:: parallel
      :type:  xlstm_jax.models.configs.ParallelConfig

      Parallel configuration.


   .. py:property:: _dtype
      :type: jax.numpy.dtype


      Returns the real dtype instead of the str from configs.

      :returns: The jnp dtype corresponding to the string value.


   .. py:method:: to_dict()

      Converts the config to a dictionary.

      Helpful for saving to disk or logging.



.. py:class:: FeedForward

   Bases: :py:obj:`flax.linen.Module`


   .. py:attribute:: config
      :type:  FeedForwardConfig


