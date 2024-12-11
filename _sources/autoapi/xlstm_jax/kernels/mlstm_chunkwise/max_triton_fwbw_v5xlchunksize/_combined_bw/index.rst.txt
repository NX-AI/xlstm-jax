xlstm_jax.kernels.mlstm_chunkwise.max_triton_fwbw_v5xlchunksize._combined_bw
============================================================================

.. py:module:: xlstm_jax.kernels.mlstm_chunkwise.max_triton_fwbw_v5xlchunksize._combined_bw

.. autoapi-nested-parse::

   This file contains the kernel that combines the recurrent and parallel
   part of the forward pass of the mLSTM chunkwise formulation.
   It should allow arbitrary large chunk sizes and head dimensions.



Functions
---------

.. autoapisummary::

   xlstm_jax.kernels.mlstm_chunkwise.max_triton_fwbw_v5xlchunksize._combined_bw.mlstm_chunkwise_bw


Module Contents
---------------

.. py:function:: mlstm_chunkwise_bw(matQ, matK, matV, vecI, vecF, matC_initial = None, vecN_initial = None, scaM_initial = None, matC_all = None, vecN_all = None, scaM_all = None, vecN_out = None, vecM_out = None, matDeltaH = None, matDeltaC_last = None, qk_scale = None, chunk_size_inter = None, chunk_size_intra = None, siz_b_L_parallel = None, siz_b_L_loop = None, siz_b_DH_parallel = None, siz_b_DH_loop = None, num_warps_intra = None, num_warps_inter = None, num_stages_intra = None, num_stages_inter = None, eps = 0.0)

