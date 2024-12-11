xlstm_jax.models.llama.attention
================================

.. py:module:: xlstm_jax.models.llama.attention


Attributes
----------

.. autoapisummary::

   xlstm_jax.models.llama.attention.AttentionBackend


Classes
-------

.. autoapisummary::

   xlstm_jax.models.llama.attention.SelfAttentionConfig
   xlstm_jax.models.llama.attention.SelfAttention


Functions
---------

.. autoapisummary::

   xlstm_jax.models.llama.attention.precompute_freqs
   xlstm_jax.models.llama.attention.apply_rotary_emb
   xlstm_jax.models.llama.attention.segment_mask
   xlstm_jax.models.llama.attention.multihead_attention


Module Contents
---------------

.. py:data:: AttentionBackend

.. py:function:: precompute_freqs(feat_dim, pos_idx = None, max_length = None, theta = 10000.0, dtype = jnp.float32)

   Compute the sine and cosine frequencies for the rotary embeddings.

   :param feat_dim: Feature dimension of the input.
   :param pos_idx: Positional indices of the tokens in the input sequence. If None, uses an arange up to max_length.
   :param max_length: Maximum length of the input sequence. Only used if pos is None.
   :param theta: Theta parameter for the wave length calculation.
   :param dtype: Data type of the returned frequencies.

   :returns: Tuple of the sine and cosine frequencies, shape (B, S, D//2). If pos_idx is None, shape is (1, S, D//2).


.. py:function:: apply_rotary_emb(xq, xk, freqs_sin = None, freqs_cos = None, theta = 10000.0)

   Apply the rotary embeddings to the queries and keys.

   :param xq: Array containing the query features of shape (B, S, NH, DHQK).
   :param xk: Array containing the key features of shape (B, S, NH, DHQK).
   :param freqs_sin: Sine frequencies for the rotary embeddings. If None, computes them based on the shape of xq.
   :param freqs_cos: Cosine frequencies for the rotary embeddings. If None, computes them based on the shape of xq.
   :param theta: Theta parameter for calculating the frequencies.

   :returns: Tuple of the query and key features with the rotary embeddings applied.


.. py:function:: segment_mask(segment_ids)

   Create a mask for the self attention module based on the segment IDs.

   :param segment_ids: Segment IDs for the input tensor. The attention weights between elements of different segments
                       is set to zero. Shape (B, S).

   :returns: Boolean tensor of shape (B, 1, S, S).


.. py:class:: SelfAttentionConfig

   Bases: :py:obj:`xlstm_jax.models.configs.SubModelConfig`


   Configuration for the self attention module.


   .. py:attribute:: head_dim
      :type:  int
      :value: 64


      Dimension of the attention heads. Number of heads is inferred from the head and embedding dimensions.


   .. py:attribute:: qk_norm
      :type:  bool
      :value: False


      Whether to apply RMSNorm to the query and key tensors.


   .. py:attribute:: use_bias
      :type:  bool
      :value: False


      Whether to use bias in the linear layers of the self attention module.


   .. py:attribute:: dropout_rate
      :type:  float
      :value: 0.0


      Dropout rate for the self attention module. Only applied during training.


   .. py:attribute:: num_layers
      :type:  int
      :value: 12


      Number of layers in the Llama model. Used for initialization.


   .. py:attribute:: dtype
      :type:  str
      :value: 'float32'


      Data type of the activations in the network.


   .. py:attribute:: attention_backend
      :type:  AttentionBackend
      :value: 'xla'


      Which backend to use for the attention module. If triton or cudnn, respective Flash Attention kernels are used.
      cudnn is only supported for GPU backends, pallas_triton for both CPU and GPU backends, and xla on all backends.


   .. py:attribute:: causal
      :type:  bool
      :value: True


      Whether to use causal attention masking for the self attention module.


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



.. py:class:: SelfAttention

   Bases: :py:obj:`flax.linen.Module`


   Self attention module with support for rotary embeddings.

   :param config: Configuration for the self attention module.


   .. py:attribute:: config
      :type:  SelfAttentionConfig


.. py:function:: multihead_attention(q, k, v, segment_ids = None, causal = True, qk_scale = None, backend = 'xla')

   Compute multi-head self attention.

   :param q: Query tensor of shape (B, S, NH, DHQK).
   :param k: Key tensor of shape (B, S, NH, DHQK).
   :param v: Value tensor of shape (B, S, NH, DHV).
   :param segment_ids: Segment IDs for the input tensor. The attention weights between elements of different segments
                       is set to zero. If None, all elements are treated as belonging to the same segment, i.e. no masking.
   :param causal: Whether to use causal attention masking for the self attention module.
   :param qk_scale: Scaling factor for the query-key logits. If None, defaults to 1/sqrt(DHQK). The scaling factor is
                    applied to the query tensor before the dot product.
   :param backend: Which backend to use for the attention module. If triton or cudnn, respective Flash Attention kernels
                   are used. cudnn is only supported for GPU backends, pallas_triton for both CPU and GPU backends, and xla on
                   all backends.

   :returns: Output tensor of shape (B, S, NH, DHV).


