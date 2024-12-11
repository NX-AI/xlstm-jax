xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.attention
==============================================================

.. py:module:: xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.attention


Classes
-------

.. autoapisummary::

   xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.attention.mLSTMBackendAttentionConfig
   xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.attention.mLSTMBackendAttention


Functions
---------

.. autoapisummary::

   xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.attention.precompute_freqs
   xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.attention.apply_rotary_emb
   xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.attention.attention


Module Contents
---------------

.. py:function:: precompute_freqs(feat_dim, max_length, theta = 10000.0)

   Compute the sine and cosine frequencies for the rotary embeddings.

   :param feat_dim: Feature dimension of the input.
   :param max_length: Maximum length of the input sequence.
   :param theta: Theta parameter for the wave length calculation.

   :returns: Tuple of the sine and cosine frequencies.


.. py:function:: apply_rotary_emb(xq, xk, freqs_sin = None, freqs_cos = None, theta = 10000.0)

   Apply the rotary embeddings to the queries and keys.

   :param xq: Array containing the query features of shape (B, NH, S, DHQK).
   :param xk: Array containing the key features of shape (B, NH, S, DHQK).
   :param freqs_sin: Sine frequencies for the rotary embeddings. If None, computes them based on the shape of xq.
   :param freqs_cos: Cosine frequencies for the rotary embeddings. If None, computes them based on the shape of xq.
   :param theta: Theta parameter for calculating the frequencies.

   :returns: Tuple of the query and key features with the rotary embeddings applied.


.. py:function:: attention(queries, keys, values, attention_mask = None, qkv_dtype = None, activation_function = 'softmax', qk_pre_activation_function = 'none', theta = 10000.0)

   This is an attention backend that mimics the attention mechanism of the transformer.

   Note that no forget and input gates are applied here.

   :param queries: Array containing the query features of shape (B, NH, S, DHQK).
   :param keys: Array containing the key features of shape (B, NH, S, DHQK).
   :param values: Array containing the value features of shape (B, NH, S, DHV).
   :param attention_mask: Array of shape (S,S) denoting the attention mask. By default, uses a causal mask which is a
                          lower triangular matrix. Dtype should be bool, where False denotes masked positions.
   :param qkv_dtype: Dtype of the queries, keys and values. If None, uses the dtype of queries.
   :param activation_function: Activation function to apply on the attention logits. Softmax is performed over the key
                               sequence as in default transformers. Sigmoid is applied with a bias of -log(S).
   :param qk_pre_activation_function: Activation function to apply on the queries and keys before computing the attention
                                      logits.
   :param theta: Theta parameter for the rotary embeddings.

   :returns: The output features of the attention of shape (B, NH, S, DHV).


.. py:class:: mLSTMBackendAttentionConfig

   .. py:attribute:: context_length
      :type:  int
      :value: -1



   .. py:attribute:: activation_function
      :type:  Literal['softmax', 'sigmoid', 'none']
      :value: 'softmax'


      Activation function to apply on the attention logits. Softmax is performed over the key sequence
      as in default transformers. Sigmoid is applied with a bias of -log(context_length).


   .. py:attribute:: qk_pre_activation_function
      :type:  Literal['swish', 'none']
      :value: 'none'


      Activation function to apply on the queries and keys before computing the attention logits.


   .. py:attribute:: theta
      :type:  float
      :value: 10000.0


      Theta parameter for the rotary embeddings.


   .. py:method:: assign_model_config_params(model_config)


.. py:class:: mLSTMBackendAttention

   Bases: :py:obj:`xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.config.mLSTMBackend`


   .. py:attribute:: config_class


   .. py:property:: can_vmap_over_heads
      :type: bool


      Whether the backend can be vmaped over the heads dimension.

      The backend is written independent of the heads dimension, and thus can be vmapped.

      :returns: True
      :rtype: bool


