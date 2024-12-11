xlstm_jax.models.xlstm_pytorch.blocks.mlstm.backend.simple
==========================================================

.. py:module:: xlstm_jax.models.xlstm_pytorch.blocks.mlstm.backend.simple


Classes
-------

.. autoapisummary::

   xlstm_jax.models.xlstm_pytorch.blocks.mlstm.backend.simple.mLSTMBackendTorchConfig
   xlstm_jax.models.xlstm_pytorch.blocks.mlstm.backend.simple.mLSTMBackendTorch


Functions
---------

.. autoapisummary::

   xlstm_jax.models.xlstm_pytorch.blocks.mlstm.backend.simple.parallel_stabilized_simple
   xlstm_jax.models.xlstm_pytorch.blocks.mlstm.backend.simple.recurrent_step_stabilized_simple


Module Contents
---------------

.. py:function:: parallel_stabilized_simple(queries, keys, values, igate_preact, fgate_preact, lower_triangular_matrix = None, stabilize_rowwise = True, eps = 1e-06)

   This is the mLSTM cell in parallel form.

   This version is stabilized. We control the range of exp() arguments by
   ensuring that they are always smaller than 0.0 by subtracting the maximum.

   :param queries: (B, NH, S, DH)
   :type queries: torch.Tensor
   :param keys: (B, NH, S, DH)
   :type keys: torch.Tensor
   :param values: (B, NH, S, DH)
   :type values: torch.Tensor
   :param igate_preact: (B, NH, S, 1)
   :type igate_preact: torch.Tensor
   :param fgate_preact: (B, NH, S, 1)
   :type fgate_preact: torch.Tensor
   :param lower_triangular_matrix: (S,S). Defaults to None.
   :type lower_triangular_matrix: torch.Tensor, optional
   :param stabilize_rowwise: Wether to stabilize the combination matrix C rowwise (take maximum per row).
                             Alternative: Subtract the maximum over all rows. Defaults to True.
   :type stabilize_rowwise: bool, optional
   :param eps: Epsilon value. Defaults to 1e-6.

   :returns: (B, NH, S, DH), h_tilde_state
   :rtype: torch.Tensor


.. py:class:: mLSTMBackendTorchConfig

   .. py:attribute:: context_length
      :type:  int
      :value: -1



   .. py:method:: assign_model_config_params(model_config)


.. py:class:: mLSTMBackendTorch(config)

   Bases: :py:obj:`xlstm_jax.models.xlstm_pytorch.blocks.mlstm.backend.config.mLSTMBackend`


   .. py:attribute:: config_class


   .. py:method:: forward(q, k, v, i, f)


   .. py:method:: reset_parameters()


.. py:function:: recurrent_step_stabilized_simple(c_state, n_state, m_state, q, k, v, igate_preact, fgate_preact, eps = 1e-06)

   This is a single step of the mLSTM operation in recurrent form.

   :param c_state: (B, NH, DH, DH)
   :type c_state: torch.Tensor
   :param n_state: (B, NH, DH, 1)
   :type n_state: torch.Tensor
   :param m_state: (B, NH, 1, 1)
   :type m_state: torch.Tensor
   :param q: (B, NH, 1, DH)
   :type q: torch.Tensor
   :param k: (B, NH, 1, DH)
   :type k: torch.Tensor
   :param v: (B, NH, 1, DH)
   :type v: torch.Tensor
   :param igate_preact: (B, NH, 1, 1)
   :type igate_preact: torch.Tensor
   :param fgate_preact: (B, NH, 1, 1)
   :type fgate_preact: torch.Tensor
   :param eps: Epsilon value. Defaults to 1e-6.

   :returns:

                 (hidden_state [B, NH, DH],
                  (c_state_new [B, NH, DH, DH], n_state_new [B, NH, DH, 1], m_state_new [B, NH, 1, 1]))
   :rtype: tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]


