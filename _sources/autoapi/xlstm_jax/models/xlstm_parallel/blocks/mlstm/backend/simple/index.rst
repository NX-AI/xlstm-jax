xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.simple
===========================================================

.. py:module:: xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.simple


Classes
-------

.. autoapisummary::

   xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.simple.mLSTMBackendParallelConfig
   xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.simple.mLSTMBackendParallel


Functions
---------

.. autoapisummary::

   xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.simple.parallel_stabilized_simple


Module Contents
---------------

.. py:function:: parallel_stabilized_simple(queries, keys, values, igate_preact, fgate_preact, lower_triangular_matrix = None, stabilize_rowwise = True, eps = 1e-06, qkv_dtype = None, gate_dtype = None)

   This is the mLSTM cell in parallel form.

   This version is stabilized. We control the range of exp() arguments by
   ensuring that they are always smaller than 0.0 by subtracting the maximum.

   :param queries: (B, NH, S, DHQK)
   :type queries: jax.Array
   :param keys: (B, NH, S, DHQK)
   :type keys: jax.Array
   :param values: (B, NH, S, DHV)
   :type values: jax.Array
   :param igate_preact: (B, NH, S, 1)
   :type igate_preact: jax.Array
   :param fgate_preact: (B, NH, S, 1)
   :type fgate_preact: jax.Array
   :param lower_triangular_matrix: (S,S). Defaults to None.
   :type lower_triangular_matrix: jax.Array, optional
   :param stabilize_rowwise: Wether to stabilize the combination matrix C rowwise (take maximum per row).
                             Alternative: Subtract the maximum over all rows. Defaults to True.
   :type stabilize_rowwise: bool, optional
   :param eps: Small value to avoid division by zero. Defaults to 1e-6.
   :type eps: float, optional
   :param qkv_dtype: dtype of queries, keys and values. Defaults to None,
                     which infers the dtype from the inputs.
   :type qkv_dtype: jnp.dtype, optional
   :param gate_dtype: dtype of igate_preact and fgate_preact. Defaults to None,
                      which infers the dtype from the inputs.
   :type gate_dtype: jnp.dtype, optional

   :returns: (B, NH, S, DH), h_tilde_state
   :rtype: jax.Array


.. py:class:: mLSTMBackendParallelConfig

   .. py:attribute:: context_length
      :type:  int
      :value: -1



   .. py:method:: assign_model_config_params(model_config)


.. py:class:: mLSTMBackendParallel

   Bases: :py:obj:`xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.config.mLSTMBackend`


   .. py:attribute:: config_class


   .. py:property:: can_vmap_over_heads
      :type: bool


      Whether the backend can be vmaped over the heads dimension.

      The backend is written independent of the heads dimension, and thus can be vmapped.

      :returns: True
      :rtype: bool


