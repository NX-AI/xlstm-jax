xlstm_jax.kernels.mlstm_chunkwise.triton_stablef._triton_bw
===========================================================

.. py:module:: xlstm_jax.kernels.mlstm_chunkwise.triton_stablef._triton_bw

.. autoapi-nested-parse::

   Triton backend for the backward pass of the mLSTM chunkwise formulation.

   This file has been adapted from the original PyTorch Triton implementation to JAX.
   For the Triton kernels, see mlstm_kernels/mlstm_kernels/mlstm/chunkwise/triton_fwbw_stablef.py.

   In this file, we use the following notation:

   Dimensions:
       B: batch size
       H: number of heads
       S: sequence length (K, V)
       T: sequence length (Q)
       K: hidden dimension (Q, K)
       V: hidden dimension (H, V)
       NT: number of chunks
       BT: chunk size



Functions
---------

.. autoapisummary::

   xlstm_jax.kernels.mlstm_chunkwise.triton_stablef._triton_bw._mlstm_chunkwise__recurrent_bw_dC
   xlstm_jax.kernels.mlstm_chunkwise.triton_stablef._triton_bw._mlstm_chunkwise__parallel_bw_dQKV
   xlstm_jax.kernels.mlstm_chunkwise.triton_stablef._triton_bw._mlstm_chunkwise_bw


Module Contents
---------------

.. py:function:: _mlstm_chunkwise__recurrent_bw_dC(matQ, vecF, scaM_inter, vecM, vecN_out, matDeltaH, matDeltaC_last = None, qk_scale = None, chunk_size = 64, num_chunks = 1, store_initial_state = False)

   Computes only the deltaC gradients for the backward pass.

   The other gradients are computed in the other (kernel) function.
   We do not need to compute the gradients for the denominator, as it cancels out in the forward in the groupnorm.

   :param matQ: Tensor containing the query vectors. Shape (B, H, T, K).
   :param vecF: Tensor containing the log forget gate activations. Shape (B, H, NT, BT).
   :param scaM_inter: States of the M scalar. Shape (B, H, NT+1).
   :param vecM: M states. Shape (B, H, T).
   :param matDeltaH: Tensor containing the H gradients. Shape (B, H, T, V).
   :param vecN_out: States of the N vector. Shape (B, H, NT * DHQK).
   :param matDeltaC_last: Tensor containing the last C gradients. Shape (B, H, DHQK, DHHV).
                          Defaults to None.
   :param qk_scale: Scale factor for the QK matrix. Defaults to None.
   :param chunk_size: Chunk size. Defaults to 64.
   :param num_chunks: Number of chunks. Defaults to 1.
   :param store_initial_state: Whether to store the inital state gradient and logscale (m state)

   :returns: Tensor containing the C gradients and the C_first gradients.
             Shapes (B, H, NT * DHQK, DHHV), (B, H, DHQK, DHHV).


.. py:function:: _mlstm_chunkwise__parallel_bw_dQKV(matQ, matK, matV, vecI, vecF, vecM_combine, scaM_inter, matC_states, matDeltaH, vecN_out, matDeltaC_states, qk_scale = None, chunk_size = 64, num_chunks = 1)

   Computes the gradients for the query, key and value matrices.

   :param matQ: Tensor containing the query vectors. Shape (B, H, T, K).
   :param matK: Tensor containing the key vectors. Shape (B, H, T, K).
   :param matV: Tensor containing the value vectors. Shape (B, H, T, V).
   :param vecF: Tensor containing the summed log forget gate activations. Shape (B, H, NT, BT).
   :param vecI: Tensor containing the input gate pre-activations. Shape (B, H, NT, BT).
   :param vecM_combine: Combined M states. Shape (B, H, T).
   :param scaM_inter: States of the M scalar. Shape (B, H, NT+1).
   :param matC_states: States of the C matrix. Shape (B, H, NT * DHQK, DHHV).
   :param matDeltaH: Tensor containing the H gradients. Shape (B, H, T, V).
   :param vecN_out: States of the N vector. Shape (B, H, T).
   :param matDeltaC_states: Tensor containing the C gradients. Shape (B, H, (NT+1) * DHQK, DHHV).
   :param qk_scale: Scale factor for the QK matrix. Defaults to None.
   :param chunk_size: Chunk size. Defaults to 64.
   :type chunk_size: int, optional
   :param num_chunks: Number of chunks. Defaults to 1.
   :type num_chunks: int, optional

   :returns: Gradients for the query, key and value matrices. Shapes (B, H, T, K), (B, H, T, K), (B, H, T, V).


.. py:function:: _mlstm_chunkwise_bw(matQ, matK, matV, vecI, vecF, matC_initial = None, vecN_initial = None, scaM_initial = None, qk_scale = None, matC_states = None, scaM_states = None, vecN_out = None, vecM_out = None, matDeltaH = None, matDeltaC_last = None, chunk_size = 64)

   Computes the backward pass of the mLSTM chunkwise formulation.

   :param matQ: Tensor containing the query vectors. Shape (B, H, T, K).
   :param matK: Tensor containing the key vectors. Shape (B, H, T, K).
   :param matV: Tensor containing the value vectors. Shape (B, H, S, DHV).
   :param vecI: Tensor containing the input gate pre-activations. Shape (B, H, T).
   :param vecF: Tensor containing the forget gate pre-activations. Shape (B, H, T).
   :param matC_initial: Tensor containing the initial C states. Shape (B, H, DHQK, DHV).
                        Defaults to None.
   :param vecN_initial: Tensor containing the initial N states. Shape (B, H, DHQK).
                        Defaults to None.
   :param scaM_initial: Tensor containing the initial M states. Shape (B, NH).
                        Defaults to None.
   :param qk_scale: Scale factor for the QK matrix. Defaults to None.
   :param matC_states: Tensor containing all C states. Shape (B, H, NT * DHQK, DHV).
                       Defaults to None.
   :param scaM_states: Tensor containing all M states. Shape (B, H, NC).
                       Defaults to None.
   :param vecN_out: Tensor containing the N states for the output. Shape (B, H, T).
                    Defaults to None.
   :param vecM_out: Tensor containing the M states for the output. Shape (B, H, T).
                    Defaults to None.
   :param matDeltaH: Tensor containing the H gradients. Shape (B, H, S, DHV).
                     Defaults to None.
   :param matDeltaC_last: Tensor containing the last C gradients. Shape (B, H, DHQK, DHV).
                          Defaults to None.
   :param chunk_size: Chunk size. Defaults to 64.

   :returns: Gradients for the query, key, value, vecI and vecF matrices. Shapes (B, H, T, K),
             (B, H, T, K), (B, H, S, DHV), (B, H, T), (B, H, T). If initial states are provided,
             the function also returns the gradients for the initial C, N and M states.


