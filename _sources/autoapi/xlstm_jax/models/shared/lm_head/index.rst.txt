xlstm_jax.models.shared.lm_head
===============================

.. py:module:: xlstm_jax.models.shared.lm_head


Classes
-------

.. autoapisummary::

   xlstm_jax.models.shared.lm_head.TPLMHead


Module Contents
---------------

.. py:class:: TPLMHead

   Bases: :py:obj:`flax.linen.Module`


   Language model head with Tensor Parallelism.

   :param parallel: Configuration for parallelism.
   :param vocab_size: Size of the vocabulary.
   :param kernel_init: Initializer for the output layer.
   :param norm_fn: Normalization function to apply before the output layer. If None, no normalization is applied.
   :param lm_head_dtype: Data type for the output layer.
   :param logits_soft_cap: Soft cap for the logits. If not None, the logits will be clipped to this value.


   .. py:attribute:: parallel
      :type:  xlstm_jax.models.configs.ParallelConfig


   .. py:attribute:: vocab_size
      :type:  int


   .. py:attribute:: kernel_init
      :type:  flax.linen.initializers.Initializer


   .. py:attribute:: norm_fn
      :type:  collections.abc.Callable[Ellipsis, flax.linen.Module] | None


   .. py:attribute:: lm_head_dtype
      :type:  jax.numpy.dtype


   .. py:attribute:: logits_soft_cap
      :type:  float | None
      :value: None



