xlstm_jax.models.xlstm_pytorch.blocks.mlstm.backend.triton_chunk
================================================================

.. py:module:: xlstm_jax.models.xlstm_pytorch.blocks.mlstm.backend.triton_chunk


Attributes
----------

.. autoapisummary::

   xlstm_jax.models.xlstm_pytorch.blocks.mlstm.backend.triton_chunk.mLSTMFunction


Functions
---------

.. autoapisummary::

   xlstm_jax.models.xlstm_pytorch.blocks.mlstm.backend.triton_chunk.chunk_mlstm_fwd_kernel_C
   xlstm_jax.models.xlstm_pytorch.blocks.mlstm.backend.triton_chunk.chunk_mlstm_fwd_kernel_h
   xlstm_jax.models.xlstm_pytorch.blocks.mlstm.backend.triton_chunk.chunk_mlstm_bwd_kernel_dC
   xlstm_jax.models.xlstm_pytorch.blocks.mlstm.backend.triton_chunk.chunk_mlstm_bwd_kernel_dqkvif
   xlstm_jax.models.xlstm_pytorch.blocks.mlstm.backend.triton_chunk.mLSTMFunc
   xlstm_jax.models.xlstm_pytorch.blocks.mlstm.backend.triton_chunk.mlstm_triton


Module Contents
---------------

.. py:function:: chunk_mlstm_fwd_kernel_C(k, v, C, n, m, i, f, initial_C, initial_n, initial_m, final_C, final_n, final_m, s_qk_h, s_qk_t, s_qk_d, s_vh_h, s_vh_t, s_vh_d, s_C_h, s_C_t, s_n_h, H, T, K, V, BT, BK, BV, NT, USE_INITIAL_STATE, STORE_FINAL_STATE)

.. py:function:: chunk_mlstm_fwd_kernel_h(q, k, v, C, n, m, m_total, i, f, h, norm, s_qk_h, s_qk_t, s_qk_d, s_vh_h, s_vh_t, s_vh_d, s_C_h, s_C_t, s_n_h, scale, H, T, K, V, BT, BK, BV, NT)

.. py:function:: chunk_mlstm_bwd_kernel_dC(q, f, m, m_total, norm, dh, dC, final_dC, final_m, initial_dC, initial_m, s_qk_h, s_qk_t, s_qk_d, s_vh_h, s_vh_t, s_vh_d, s_C_h, s_C_t, scale, H, T, K, V, BT, BK, BV, NT)

.. py:function:: chunk_mlstm_bwd_kernel_dqkvif(q, k, v, C, m, m_total, norm, i, f, dh, dC, dq, dk, dv, s_qk_h, s_qk_t, s_qk_d, s_vh_h, s_vh_t, s_vh_d, s_C_h, s_C_t, scale, B, H, T, K, V, BT, BK, BV, NT)

.. py:function:: mLSTMFunc(chunk_size, save_states = True)

.. py:data:: mLSTMFunction

.. py:function:: mlstm_triton(q, k, v, i, f, initial_C = None, initial_n = None, initial_m = None, output_final_state = False)

