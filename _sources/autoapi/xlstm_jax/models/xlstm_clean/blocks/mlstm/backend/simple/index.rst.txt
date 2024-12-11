xlstm_jax.models.xlstm_clean.blocks.mlstm.backend.simple
========================================================

.. py:module:: xlstm_jax.models.xlstm_clean.blocks.mlstm.backend.simple


Classes
-------

.. autoapisummary::

   xlstm_jax.models.xlstm_clean.blocks.mlstm.backend.simple.mLSTMBackendJaxConfig
   xlstm_jax.models.xlstm_clean.blocks.mlstm.backend.simple.mLSTMBackendJax


Functions
---------

.. autoapisummary::

   xlstm_jax.models.xlstm_clean.blocks.mlstm.backend.simple.parallel_stabilized_simple
   xlstm_jax.models.xlstm_clean.blocks.mlstm.backend.simple.recurrent_step_stabilized_simple


Module Contents
---------------

.. py:function:: parallel_stabilized_simple(queries, keys, values, igate_preact, fgate_preact, lower_triangular_matrix = None, stabilize_rowwise = True, eps = 1e-06)

   This is the mLSTM cell in parallel form.

   This version is stabilized. We control the range of exp() arguments by ensuring that they are always smaller than
   0.0 by subtracting the maximum.

   :param queries: (B, NH, S, DH)
   :param keys: (B, NH, S, DH)
   :param values: (B, NH, S, DH)
   :param igate_preact: (B, NH, S, 1)
   :param fgate_preact: (B, NH, S, 1)
   :param lower_triangular_matrix: (S,S). Defaults to None.
   :param stabilize_rowwise: Whether to stabilize the combination matrix C row-wise (take maximum per row).
                             Alternative: Subtract the maximum over all rows. Defaults to True.
   :param eps: Epsilon value. Defaults to 1e-6.

   :returns: (B, NH, S, DH), h_tilde_state
   :rtype: jax.Array


.. py:class:: mLSTMBackendJaxConfig

   .. py:attribute:: context_length
      :type:  int
      :value: -1



   .. py:method:: assign_model_config_params(model_config)


.. py:class:: mLSTMBackendJax

   Bases: :py:obj:`xlstm_jax.models.xlstm_clean.blocks.mlstm.backend.config.mLSTMBackend`


   .. py:attribute:: config_class


.. py:function:: recurrent_step_stabilized_simple(c_state, n_state, m_state, q, k, v, igate_preact, fgate_preact, eps = 1e-06)

   This is a single step of the mLSTM operation in recurrent form.

   :param c_state: (B, NH, DH, DH)
   :param n_state: (B, NH, DH, 1)
   :param m_state: (B, NH, 1, 1)
   :param q: (B, NH, 1, DH)
   :param k: (B, NH, 1, DH)
   :param v: (B, NH, 1, DH)
   :param igate_preact: (B, NH, 1, 1)
   :type igate_preact: jax.Array
   :param fgate_preact: (B, NH, 1, 1)
   :type fgate_preact: jax.Array
   :param eps: Epsilon value. Defaults to 1e-6.

   :returns:

                 (hidden_state [B, NH, DH],
                  (c_state_new [B, NH, DH, DH], n_state_new [B, NH, DH, 1], m_state_new [B, NH, 1, 1]))
   :rtype: tuple[jax.Array, tuple[jax.Array, jax.Array]]


