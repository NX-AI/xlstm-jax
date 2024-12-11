xlstm_jax.kernels.mlstm_chunkwise.triton_stablef._triton_fw
===========================================================

.. py:module:: xlstm_jax.kernels.mlstm_chunkwise.triton_stablef._triton_fw

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

   xlstm_jax.kernels.mlstm_chunkwise.triton_stablef._triton_fw.assert_equal
   xlstm_jax.kernels.mlstm_chunkwise.triton_stablef._triton_fw._mlstm_chunkwise__recurrent_fw_C
   xlstm_jax.kernels.mlstm_chunkwise.triton_stablef._triton_fw._mlstm_chunkwise__parallel_fw_H
   xlstm_jax.kernels.mlstm_chunkwise.triton_stablef._triton_fw._mlstm_chunkwise_fw


Module Contents
---------------

.. py:function:: assert_equal(a, b)

.. py:function:: _mlstm_chunkwise__recurrent_fw_C(matK, matV, vecF, vecI, matC_initial = None, vecN_initial = None, scaMinter_initial = None, chunk_size = 64, num_chunks = 1, store_final_state = False)

   Execute the recurrent forward kernel for the C computation in the mLSTM chunkwise formulation.

   This function defines the grid and block sizes for the kernel launch and calls the kernel. See
   the fwbw backend implementation and the triton kernels for more information.

   :param matK: Tensor containing the keys. Shape (B, H, S, K).
   :param matV: Tensor containing the values. Shape (B, H, S, V).
   :param vecB: Tensor containing the summed log forget gate activations. Shape (B, H, NT, BT).
   :param vecI: Tensor containing the input gate. Shape (B, H, NT, BT).
   :param matC_states: Buffer for the states of the C matrix.
                       Shape (B, H, NT * K, V). Defaults to None.
   :param vecN_states: Buffer for the states of the N vector. Shape (B, H, NT * K).
                       Defaults to None.
   :param scaMinter_states: Buffer for the states of the M scalar. Shape (B, H, (NT + 1)).
                            Defaults to None.
   :param matC_initial: Initial state of the C matrix. Shape (B, H, K, V).
                        Defaults to None.
   :param vecN_initial: Initial state of the N vector. Shape (B, H, K).
                        Defaults to None.
   :param scaMinter_initial: Initial state of the M scalar. Shape (B, H).
                             Defaults to None.
   :param qk_scale: Scaling factor for the QK matrix. Defaults to None and will be inferred.
   :param chunk_size: Chunk size for the kernel. Defaults to 64.
   :param num_chunks: Number of chunks. Defaults to 1.
   :param store_final_state: Whether to return the final state

   :returns: Tuple containing the states of the C matrix, the N vector and the M scalar.


.. py:function:: _mlstm_chunkwise__parallel_fw_H(matQ, matK, matV, matC_states, vecN_states, scaMinter_states, vecI, vecF, qk_scale = None, chunk_size = 64, num_chunks = 1, eps = 1e-06, stabilize_correctly = True, norm_val = 1.0)

   Execute the parallel forward kernel for the H computation in the mLSTM chunkwise formulation.

   This function defines the grid and block sizes for the kernel launch and calls the kernel. See
   the fwbw backend implementation and the triton kernels for more information.

   :param matQ: Tensor containing the queries. Shape (B, H, S, K).
   :param matK: Tensor containing the keys. Shape (B, H, S, K).
   :param matV: Tensor containing the values. Shape (B, H, S, V).
   :param matC_states: States of the C matrix. Shape (B, H, NT * K, V).
                       This state and following states must be all states up to the last chunk, i.e. :-1.
   :param vecN_states: States of the N vector. Shape (B, H, NT * K).
   :param scaMinter_states: States of the M scalar. Shape (B, H, NT + 1).
   :param vecI: Tensor containing the input gate. Shape (B, H, NT, BT).
   :param vecB: Tensor containing the summed log forget gate activations. Shape (B, H, NT, BT).
   :param qk_scale: Scaling factor for the QK matrix. Defaults to None and will be inferred.
   :param CHUNK_SIZE: Chunk size for the kernel. Defaults to 64.
   :param NUM_CHUNKS: Number of chunks. Defaults to 1.
   :param EPS: Small value to avoid division by zero. Defaults to 1e-6.

   :returns: Tuple containing the output matrix H (shape (B, H, S, V)) and the N vector (shape (B, H, S)).


.. py:function:: _mlstm_chunkwise_fw(matQ, matK, matV, vecI, vecF, matC_initial = None, vecN_initial = None, scaM_initial = None, qk_scale = None, return_last_states = False, return_all_states = False, chunk_size = 64, stabilize_correctly = True, norm_val = 1.0, eps = 1e-06)

   Execute the forward pass of the mLSTM chunkwise formulation.

   :param matQ: Tensor containing the queries. Shape (B, H, S, K).
   :param matK: Tensor containing the keys. Shape (B, H, S, K).
   :param matV: Tensor containing the values. Shape (B, H, S, V).
   :param vecI: Tensor containing the input gate. Shape (B, H, S).
   :param vecF: Tensor containing the forget gate. Shape (B, H, S).
   :param matC_initial: Initial state of the C matrix. Shape (B, H, K, V).
                        Defaults to None.
   :param vecN_initial: Initial state of the N vector. Shape (B, H, K).
                        Defaults to None.
   :param scaM_initial: Initial state of the M scalar. Shape (B, H).
                        Defaults to None.
   :param qk_scale: Scaling factor for the QK matrix. Defaults to None and will be inferred.
   :param return_last_states: Whether to return the last states. Defaults to False.
   :param return_all_states: Whether to return all states. Defaults to False.
   :param chunk_size: Chunk size for the kernel. Defaults to 64.
   :param stabilize_correctly: Whether to stabilize with max(norm_val*e^-m, |nk|) instead of max(norm_val, |nk|)
   :param norm_val: Norm scale in the max formula above.

   :returns: Tuple containing the output matrix H (shape (B, H, S, V)), the N vector (shape (B, H, S)),
             the M scalar (shape (B, H)). Optionally, it might contain last states (matC_states,
             vecN_states, scaMinter_states) and optional all states (matC_states, vecN_states,
             scaMinter_states).


