xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend_utils
==========================================================

.. py:module:: xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend_utils


Functions
---------

.. autoapisummary::

   xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend_utils.run_backend


Module Contents
---------------

.. py:function:: run_backend(parent, cell_config, q, k, v, igate_preact, fgate_preact)

   Execute the mLSTM backend for the given input tensors.

   This function handles the caching of intermediate states, if enabled, and the vmap over the heads dimension.
   The caching follows the setup of the cache in the Attention module:
   https://github.com/google/flax/blob/main/flax/linen/attention.py#L570. During decoding, if the cache is not
   initialized, the cache is initialized with zeros and we *do not* update the cache, following the setup in the
   Attention module. If the cache is provided, we update the cache with the new states.

   :param parent: The parent module.
   :param cell_config: The mLSTM cell configuration.
   :param q: The query tensor, shape (batch_size, seq_len, num_heads, qk_dim).
   :param k: The key tensor, shape (batch_size, seq_len, num_heads, qk_dim).
   :param v: The value tensor, shape (batch_size, seq_len, num_heads, v_dim).
   :param igate_preact: The input gate preactivation, shape (batch_size, seq_len, num_heads, 1).
   :param fgate_preact: The forget gate preactivation, shape (batch_size, seq_len, num_heads, 1).

   :returns: The output tensor, shape (batch_size, seq_len, num_heads, v_dim).


