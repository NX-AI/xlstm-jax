xlstm_jax.kernels.mlstm_chunkwise.max_triton_fwbw_v3noslice._triton_fw
======================================================================

.. py:module:: xlstm_jax.kernels.mlstm_chunkwise.max_triton_fwbw_v3noslice._triton_fw

.. autoapi-nested-parse::

   Triton backend for the forward pass of the mLSTM chunkwise formulation.

   This file has been adapted from the original PyTorch Triton implementation to JAX.
   For the Triton kernels, see mlstm_kernels/mlstm_kernels/mlstm/chunkwise/max_triton_fwbw_v3/_triton_fw.py.

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

   xlstm_jax.kernels.mlstm_chunkwise.max_triton_fwbw_v3noslice._triton_fw._mlstm_chunkwise__recurrent_fw_C
   xlstm_jax.kernels.mlstm_chunkwise.max_triton_fwbw_v3noslice._triton_fw._mlstm_chunkwise__parallel_fw_H
   xlstm_jax.kernels.mlstm_chunkwise.max_triton_fwbw_v3noslice._triton_fw._mlstm_chunkwise_fw


Module Contents
---------------

.. py:function:: _mlstm_chunkwise__recurrent_fw_C(matK, matV, vecB, vecI, matC_states = None, vecN_states = None, scaMinter_states = None, matC_initial = None, vecN_initial = None, scaMinter_initial = None, qk_scale = None, CHUNK_SIZE = 64, NUM_CHUNKS = 1)

   Execute the recurrent forward kernel for the C computation in the mLSTM chunkwise formulation.

   This function defines the grid and block sizes for the kernel launch and calls the kernel. See
   the fwbw backend implementation and the triton kernels for more information.

   :param matK: Tensor containing the keys. Shape (B, NH, S, DHQK).
   :param matV: Tensor containing the values. Shape (B, NH, S, DHHV).
   :param vecB: Tensor containing the summed log forget gate activations. Shape (B, NH, NC, L).
   :param vecI: Tensor containing the input gate. Shape (B, NH, NC, L).
   :param matC_states: Buffer for the states of the C matrix.
                       Shape (B, NH, (NC + 1) * DHQK, DHHV). Defaults to None.
   :param vecN_states: Buffer for the states of the N vector. Shape (B, NH, (NC + 1) * DHQK).
                       Defaults to None.
   :param scaMinter_states: Buffer for the states of the M scalar. Shape (B, NH, (NC + 1)).
                            Defaults to None.
   :param matC_initial: Initial state of the C matrix. Shape (B, NH, DHQK, DHHV).
                        Defaults to None.
   :param vecN_initial: Initial state of the N vector. Shape (B, NH, DHQK).
                        Defaults to None.
   :param scaMinter_initial: Initial state of the M scalar. Shape (B, NH).
                             Defaults to None.
   :param qk_scale: Scaling factor for the QK matrix. Defaults to None and will be inferred.
   :param CHUNK_SIZE: Chunk size for the kernel. Defaults to 64.
   :param NUM_CHUNKS: Number of chunks. Defaults to 1.

   :returns: Tuple containing the states of the C matrix, the N vector and the M scalar.


.. py:function:: _mlstm_chunkwise__parallel_fw_H(matQ, matK, matV, matC_states, vecN_states, scaMinter_states, vecI, vecB, qk_scale = None, CHUNK_SIZE = 64, NUM_CHUNKS = 1, EPS = 1e-06)

   Execute the parallel forward kernel for the H computation in the mLSTM chunkwise formulation.

   This function defines the grid and block sizes for the kernel launch and calls the kernel. See
   the fwbw backend implementation and the triton kernels for more information.

   :param matQ: Tensor containing the queries. Shape (B, NH, S, DHQK).
   :param matK: Tensor containing the keys. Shape (B, NH, S, DHQK).
   :param matV: Tensor containing the values. Shape (B, NH, S, DHHV).
   :param matC_states: States of the C matrix. Shape (B, NH, NC * DHQK, DHHV).
                       This state and following states must be all states up to the last chunk, i.e. :-1.
   :param vecN_states: States of the N vector. Shape (B, NH, NC * DHQK).
   :param scaMinter_states: States of the M scalar. Shape (B, NH, NC).
   :param vecI: Tensor containing the input gate. Shape (B, NH, NC, L).
   :param vecB: Tensor containing the summed log forget gate activations. Shape (B, NH, NC, L).
   :param qk_scale: Scaling factor for the QK matrix. Defaults to None and will be inferred.
   :param CHUNK_SIZE: Chunk size for the kernel. Defaults to 64.
   :param NUM_CHUNKS: Number of chunks. Defaults to 1.
   :param EPS: Small value to avoid division by zero. Defaults to 1e-6.

   :returns: Tuple containing the output matrix H (shape (B, NH, S, DHHV)) and the N vector (shape (B, NH, S)).


.. py:function:: _mlstm_chunkwise_fw(matQ, matK, matV, vecI, vecF, matC_initial = None, vecN_initial = None, scaM_initial = None, qk_scale = None, return_last_states = False, return_all_states = False, CHUNK_SIZE = 64, EPS = 1e-06)

   Execute the forward pass of the mLSTM chunkwise formulation.

   :param matQ: Tensor containing the queries. Shape (B, NH, S, DHQK).
   :param matK: Tensor containing the keys. Shape (B, NH, S, DHQK).
   :param matV: Tensor containing the values. Shape (B, NH, S, DHV).
   :param vecI: Tensor containing the input gate. Shape (B, NH, S).
   :param vecF: Tensor containing the forget gate. Shape (B, NH, S).
   :param matC_initial: Initial state of the C matrix. Shape (B, NH, DHQK, DHV).
                        Defaults to None.
   :param vecN_initial: Initial state of the N vector. Shape (B, NH, DHQK).
                        Defaults to None.
   :param scaM_initial: Initial state of the M scalar. Shape (B, NH).
                        Defaults to None.
   :param qk_scale: Scaling factor for the QK matrix. Defaults to None and will be inferred.
   :param return_last_states: Whether to return the last states. Defaults to False.
   :param return_all_states: Whether to return all states. Defaults to False.
   :param CHUNK_SIZE: Chunk size for the kernel. Defaults to 64.
   :param EPS: Small value to avoid division by zero. Defaults to 1e-6.

   :returns: Tuple containing the output matrix H (shape (B, NH, S, DHV)), the N vector (shape (B, NH, S)),
             the M scalar (shape (B, NH)). Optionally, it might contain last states (matC_states,
             vecN_states, scaMinter_states) and optional all states (matC_states, vecN_states,
             scaMinter_states).


