xlstm_jax.models.llama.llama
============================

.. py:module:: xlstm_jax.models.llama.llama


Classes
-------

.. autoapisummary::

   xlstm_jax.models.llama.llama.LlamaConfig
   xlstm_jax.models.llama.llama.SelfAttentionBlock
   xlstm_jax.models.llama.llama.FFNBlock
   xlstm_jax.models.llama.llama.TransformerBlock
   xlstm_jax.models.llama.llama.TransformerBlockStack
   xlstm_jax.models.llama.llama.LlamaTransformer


Module Contents
---------------

.. py:class:: LlamaConfig

   Bases: :py:obj:`xlstm_jax.models.configs.SubModelConfig`


   Configuration for the LLAMA model.

   For simplicity, all the configuration options are kept in a single class. Sub-configs for the attention and
   feedforward blocks are generated dynamically from this class.


   .. py:attribute:: vocab_size
      :type:  int

      Vocabulary size.


   .. py:attribute:: embedding_dim
      :type:  int

      Embedding dimension.


   .. py:attribute:: num_blocks
      :type:  int

      Number of transformer blocks. One block consists of self-attention and feedforward block.


   .. py:attribute:: head_dim
      :type:  int
      :value: 128


      Dimension of the attention heads. The number of heads is inferred by `embed_dim // head_dim`.


   .. py:attribute:: qk_norm
      :type:  bool
      :value: False


      Whether to apply RMSNorm to Q and K.


   .. py:attribute:: causal
      :type:  bool
      :value: True


      Whether to use causal attention masking.


   .. py:attribute:: theta
      :type:  float
      :value: 10000.0


      Rotary position encoding frequency.


   .. py:attribute:: ffn_multiple_of
      :type:  int
      :value: 64


      Multiple of the feedforward hidden dimension to increase to.


   .. py:attribute:: ffn_dim_multiplier
      :type:  float
      :value: 1.0


      Multiplier to apply to the feedforward hidden dimension. By default, the hidden dimension is 8/3 of the
      embedding dimension. The multiplier is applied to this hidden dimension.


   .. py:attribute:: use_bias
      :type:  bool
      :value: False


      Whether to use bias in the linear layers.


   .. py:attribute:: scan_blocks
      :type:  bool
      :value: True


      Whether to scan the transformer blocks. Recommended for larger models to reduce compilation time.


   .. py:attribute:: attention_backend
      :type:  xlstm_jax.models.llama.attention.AttentionBackend
      :value: 'xla'


      Which backend to use for the attention module. If triton or cudnn, respective Flash Attention kernels are used.
      cudnn is only supported for GPU backends, pallas_triton for both CPU and GPU backends, and xla on all backends.


   .. py:attribute:: mask_across_document_boundaries
      :type:  bool
      :value: False


      Whether to mask attention across document boundaries. If True, the tokens in a document can only attend to
      tokens within the same document.


   .. py:attribute:: dtype
      :type:  str
      :value: 'float32'


      Data type to use for the activations.


   .. py:attribute:: dropout_rate
      :type:  float
      :value: 0.0


      Dropout rate to apply to the activations.


   .. py:attribute:: add_embedding_dropout
      :type:  bool
      :value: False


      Whether to apply dropout to the embeddings.


   .. py:attribute:: logits_soft_cap
      :type:  float | None
      :value: None


      Soft cap for the logits. If not None, the logits are soft-clipped to the range
      [-logits_soft_cap, logits_soft_cap].


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



.. py:class:: SelfAttentionBlock

   Bases: :py:obj:`flax.linen.Module`


   Attention block consisting of self-attention and residual connection.

   :param config: Configuration for the attention block.
   :param train: Whether to run in training mode or not. If True, applies dropout.


   .. py:attribute:: config
      :type:  LlamaConfig


   .. py:attribute:: train
      :type:  bool
      :value: False



.. py:class:: FFNBlock

   Bases: :py:obj:`flax.linen.Module`


   Feedforward block consisting of a feedforward layer and residual connection.

   :param config: Configuration for the feedforward block.
   :param train: Whether to run in training mode or not. If True, applies dropout.


   .. py:attribute:: config
      :type:  LlamaConfig


   .. py:attribute:: train
      :type:  bool
      :value: False



.. py:class:: TransformerBlock

   Bases: :py:obj:`flax.linen.Module`


   Transformer block consisting of self-attention and feedforward block.

   :param config: Configuration for the transformer block.
   :param train: Whether to run in training mode or not. If True, applies dropout.


   .. py:attribute:: config
      :type:  LlamaConfig


   .. py:attribute:: train
      :type:  bool
      :value: False



.. py:class:: TransformerBlockStack

   Bases: :py:obj:`flax.linen.Module`


   Stack of transformer blocks.

   :param config: Configuration for the transformer block stack.


   .. py:attribute:: config
      :type:  LlamaConfig


.. py:class:: LlamaTransformer

   Bases: :py:obj:`flax.linen.Module`


   LLAMA transformer model.

   :param config: Configuration for the LLAMA model.


   .. py:attribute:: config
      :type:  LlamaConfig


