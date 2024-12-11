xlstm_jax.kernels.mlstm_chunkwise.max_triton_fwbw_v5xlchunksize._recurrent_fw
=============================================================================

.. py:module:: xlstm_jax.kernels.mlstm_chunkwise.max_triton_fwbw_v5xlchunksize._recurrent_fw


Functions
---------

.. autoapisummary::

   xlstm_jax.kernels.mlstm_chunkwise.max_triton_fwbw_v5xlchunksize._recurrent_fw.mlstm_chunkwise__recurrent_fw_C


Module Contents
---------------

.. py:function:: mlstm_chunkwise__recurrent_fw_C(matK, matV, vecF, vecI, matC_initial = None, vecN_initial = None, scaMinter_initial = None, chunk_size = 64, num_stages = None, num_warps = None, save_states_every_nth_chunk = 1)

   Execute the recurrent forward kernel for the C computation in the mLSTM chunkwise formulation.

   This function defines the grid and block sizes for the kernel launch and calls the kernel. See
   the fwbw backend implementation and the triton kernels for more information.

   :param matK: Tensor containing the keys. Shape (B, NH, S, DHQK).
   :param matV: Tensor containing the values. Shape (B, NH, S, DHHV).
   :param vecF: Tensor containing the forget gate. Shape (B, NH, NC, L).
   :param vecI: Tensor containing the input gate. Shape (B, NH, NC, L).
   :param matC_initial: Initial state of the C matrix. Shape (B, NH, DHQK, DHHV).
                        Defaults to None.
   :param vecN_initial: Initial state of the N vector. Shape (B, NH, DHQK).
                        Defaults to None.
   :param scaMinter_initial: Initial state of the M scalar. Shape (B, NH).
                             Defaults to None.
   :param chunk_size: Chunk size for the kernel. Defaults to 64.
   :param num_stages: Number of stages of the kernel. Defaults to None.
   :param num_warps: Number of warps of the kernel. Defaults to None.
   :param save_states_every_nth_chunk: Save the states every nth chunk. Defaults to 1.

   :returns: Tuple containing the states of the C matrix, the N vector and the M scalar.


