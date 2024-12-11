xlstm_jax.kernels.mlstm_chunkwise.max_triton_fwbw_v5xlchunksize._recurrent_bw
=============================================================================

.. py:module:: xlstm_jax.kernels.mlstm_chunkwise.max_triton_fwbw_v5xlchunksize._recurrent_bw


Functions
---------

.. autoapisummary::

   xlstm_jax.kernels.mlstm_chunkwise.max_triton_fwbw_v5xlchunksize._recurrent_bw.mlstm_chunkwise__recurrent_bw_dC


Module Contents
---------------

.. py:function:: mlstm_chunkwise__recurrent_bw_dC(matQ, vecF, scaM_inter, vecM_combine, matDeltaH, vecN_out, matDeltaC_last = None, qk_scale = None, chunk_size = 64, save_states_every_nth_chunk = 1, num_warps = None, num_stages = None, eps = 0.0)

   Computes only the deltaC gradients for the backward pass.

   The other gradients are computed in the other (kernel) function.
   We do not need to compute the gradients for the denominator, as it cancels out in the forward in the groupnorm.

   :param matQ: Tensor containing the query vectors. Shape (B, NH, S, DHQK).
   :param vecF: Tensor containing the forget gate pre-activations. Shape (B, NH, NC * L) = (B, NH, S).
   :param scaM_inter: States of the M scalar. Shape (B, NH, NC+1).
   :param vecM_combine: Combined M states. Shape (B, NH, S).
   :param matDeltaH: Tensor containing the H gradients. Shape (B, NH, S, DHHV).
   :param vecN_out: States of the N vector. Shape (B, NH, NC * DHQK).
   :param matDeltaC_last: Tensor containing the last C gradients. Shape (B, NH, DHQK, DHHV).
                          Defaults to None.
   :param qk_scale: Scale factor for the QK matrix. Defaults to None.
   :param chunk_size: Chunk size. Defaults to 64.
   :param save_states_every_nth_chunk: Save the states every nth chunk. Defaults to 1.
   :param num_warps: Number of warps. Defaults to None.
   :param num_stages: Number of stages. Defaults to None.
   :param eps: Epsilon value. Defaults to 1e-6.

   :returns: Tensor containing the C gradients. Shape (B, NH, (NC+1) * DHQK, DHHV).


