xlstm_jax.kernels.mlstm_chunkwise.max_triton_fwbw_v3noslice._triton_bw
======================================================================

.. py:module:: xlstm_jax.kernels.mlstm_chunkwise.max_triton_fwbw_v3noslice._triton_bw

.. autoapi-nested-parse::

   Triton backend for the backward pass of the mLSTM chunkwise formulation.

   This file has been adapted from the original PyTorch Triton implementation to JAX.
   For the Triton kernels, see mlstm_kernels/mlstm_kernels/mlstm/chunkwise/max_triton_fwbw_v3/_triton_bw.py.

   In this file, we use the following notation:

   Dimensions:
       B: batch size
       NH: number of heads
       S: sequence length (K, V)
       T: sequence length (Q)
       DHQK: hidden dimension (Q, K)
       DHHV: hidden dimension (H, V)
       NC: number of chunks
       L: chunk size

   Variables:
       vecA, a: forward gate contribution, contribution of forget gates from last chunk state C_{k-1} to
               current timestep t
       vecB, b: backward gate contribution, contribution of forget and input gates up to next chunk
               state C_k (form current timestep t)
       scaG, g: "go through" gate contribution, contribution of forget gates from C_{k-1} to C_k.
       matD, D: gating matrix for the parallel form.



Functions
---------

.. autoapisummary::

   xlstm_jax.kernels.mlstm_chunkwise.max_triton_fwbw_v3noslice._triton_bw._mlstm_chunkwise__recurrent_bw_dC
   xlstm_jax.kernels.mlstm_chunkwise.max_triton_fwbw_v3noslice._triton_bw._mlstm_chunkwise__parallel_bw_dQKV
   xlstm_jax.kernels.mlstm_chunkwise.max_triton_fwbw_v3noslice._triton_bw._mlstm_chunkwise_bw


Module Contents
---------------

.. py:function:: _mlstm_chunkwise__recurrent_bw_dC(matQ, vecB, scaM_inter, vecM_combine, matDeltaH, vecN_out, matDeltaC_last = None, qk_scale = None, CHUNK_SIZE = 64, NUM_CHUNKS = 1, EPS = 1e-06)

   Computes only the deltaC gradients for the backward pass.

   The other gradients are computed in the other (kernel) function.
   We do not need to compute the gradients for the denominator, as it cancels out in the forward in the groupnorm.

   :param matQ: Tensor containing the query vectors. Shape (B, NH, S, DHQK).
   :param vecB: Tensor containing the summed log forget gate activations. Shape (B, NH, NC, L).
   :param scaM_inter: States of the M scalar. Shape (B, NH, NC+1).
   :param vecM_combine: Combined M states. Shape (B, NH, S).
   :param matDeltaH: Tensor containing the H gradients. Shape (B, NH, S, DHHV).
   :param vecN_out: States of the N vector. Shape (B, NH, NC * DHQK).
   :param matDeltaC_last: Tensor containing the last C gradients. Shape (B, NH, DHQK, DHHV).
                          Defaults to None.
   :param qk_scale: Scale factor for the QK matrix. Defaults to None.
   :param CHUNK_SIZE: Chunk size. Defaults to 64.
   :param NUM_CHUNKS: Number of chunks. Defaults to 1.
   :param EPS: Epsilon value. Defaults to 1e-6.

   :returns: Tensor containing the C gradients. Shape (B, NH, (NC+1) * DHQK, DHHV).


.. py:function:: _mlstm_chunkwise__parallel_bw_dQKV(matQ, matK, matV, vecB, vecI, vecM_combine, scaM_inter, matC_states, matDeltaH, vecN_out, matDeltaC_states, qk_scale = None, CHUNK_SIZE = 64, NUM_CHUNKS = 1, EPS = 1e-06)

   Computes the gradients for the query, key and value matrices.

   :param matQ: Tensor containing the query vectors. Shape (B, NH, S, DHQK).
   :param matK: Tensor containing the key vectors. Shape (B, NH, S, DHQK).
   :param matV: Tensor containing the value vectors. Shape (B, NH, S, DHHV).
   :param vecB: Tensor containing the summed log forget gate activations. Shape (B, NH, NC, L).
   :param vecI: Tensor containing the input gate pre-activations. Shape (B, NH, NC, L).
   :param vecM_combine: Combined M states. Shape (B, NH, S).
   :param scaM_inter: States of the M scalar. Shape (B, NH, NC+1).
   :param matC_states: States of the C matrix. Shape (B, NH, NC * DHQK, DHHV).
   :param matDeltaH: Tensor containing the H gradients. Shape (B, NH, S, DHHV).
   :param vecN_out: States of the N vector. Shape (B, NH, S).
   :param matDeltaC_states: Tensor containing the C gradients. Shape (B, NH, (NC+1) * DHQK, DHHV).
   :param qk_scale: Scale factor for the QK matrix. Defaults to None.
   :param CHUNK_SIZE: Chunk size. Defaults to 64.
   :type CHUNK_SIZE: int, optional
   :param NUM_CHUNKS: Number of chunks. Defaults to 1.
   :type NUM_CHUNKS: int, optional
   :param EPS: Epsilon value. Defaults to 1e-6.

   :returns: Gradients for the query, key and value matrices. Shapes (B, NH, S, DHQK), (B, NH, S, DHQK), (B, NH, S, DHHV).


.. py:function:: _mlstm_chunkwise_bw(matQ, matK, matV, vecI, vecF, matC_initial = None, vecN_initial = None, scaM_initial = None, qk_scale = None, matC_all = None, vecN_all = None, scaM_all = None, vecN_out = None, vecM_out = None, matDeltaH = None, matDeltaC_last = None, CHUNK_SIZE = 64, EPS = 1e-06, reduce_slicing = False)

   Computes the backward pass of the mLSTM chunkwise formulation.

   :param matQ: Tensor containing the query vectors. Shape (B, NH, S, DHQK).
   :param matK: Tensor containing the key vectors. Shape (B, NH, S, DHQK).
   :param matV: Tensor containing the value vectors. Shape (B, NH, S, DHV).
   :param vecI: Tensor containing the input gate pre-activations. Shape (B, NH, S).
   :param vecF: Tensor containing the forget gate pre-activations. Shape (B, NH, S).
   :param matC_initial: Tensor containing the initial C states. Shape (B, NH, DHQK, DHV).
                        Defaults to None.
   :param vecN_initial: Tensor containing the initial N states. Shape (B, NH, DHQK).
                        Defaults to None.
   :param scaM_initial: Tensor containing the initial M states. Shape (B, NH).
                        Defaults to None.
   :param qk_scale: Scale factor for the QK matrix. Defaults to None.
   :param matC_all: Tensor containing all C states. Shape (B, NH, NC * DHQK, DHV).
                    Defaults to None.
   :param vecN_all: Tensor containing all N states. Shape (B, NH, NC * DHQK).
                    Defaults to None.
   :param scaM_all: Tensor containing all M states. Shape (B, NH, NC).
                    Defaults to None.
   :param vecN_out: Tensor containing the N states for the output. Shape (B, NH, S).
                    Defaults to None.
   :param vecM_out: Tensor containing the M states for the output. Shape (B, NH, S).
                    Defaults to None.
   :param matDeltaH: Tensor containing the H gradients. Shape (B, NH, S, DHV).
                     Defaults to None.
   :param matDeltaC_last: Tensor containing the last C gradients. Shape (B, NH, DHQK, DHV).
                          Defaults to None.
   :param CHUNK_SIZE: Chunk size. Defaults to 64.
   :param EPS: Epsilon value. Defaults to 1e-6.
   :param reduce_slicing: If True, reduces the slicing operations taken in the preprocessing to
                          the kernel. This leads to performance improvements during training while returning
                          the same results. Defaults to False.

   :returns: Gradients for the query, key, value, vecI and vecF matrices. Shapes (B, NH, S, DHQK),
             (B, NH, S, DHQK), (B, NH, S, DHV), (B, NH, S), (B, NH, S). If initial states are provided,
             the function also returns the gradients for the initial C, N and M states.


