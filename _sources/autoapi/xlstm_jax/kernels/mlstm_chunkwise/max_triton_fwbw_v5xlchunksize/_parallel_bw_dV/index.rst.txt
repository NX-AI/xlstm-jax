xlstm_jax.kernels.mlstm_chunkwise.max_triton_fwbw_v5xlchunksize._parallel_bw_dV
===============================================================================

.. py:module:: xlstm_jax.kernels.mlstm_chunkwise.max_triton_fwbw_v5xlchunksize._parallel_bw_dV


Functions
---------

.. autoapisummary::

   xlstm_jax.kernels.mlstm_chunkwise.max_triton_fwbw_v5xlchunksize._parallel_bw_dV.mlstm_chunkwise__parallel_bw_dV


Module Contents
---------------

.. py:function:: mlstm_chunkwise__parallel_bw_dV(matQ, matK, matV, vecI, vecA, vecB, matC_all, vecN_all, scaM_all, vecN_out, vecM_out, matDeltaH, matDeltaC_states, qk_scale = None, chunk_size = 64, siz_b_LQ = 32, siz_b_LKV = 32, siz_b_DHQK = None, siz_b_DHHV = None, num_warps = None, num_stages = None, eps = 0.0, output_dtype = jnp.float32)

   Computes only the deltaK gradients for the backward pass. The other gradients are computed in
   the other (kernel) function.

   This function defines the grid and block sizes for the kernel launch and calls the kernel.
   chunk parallel size:        siz_b_LKV
   chunk loop size:            siz_b_LQ
   head dim parallel size:     siz_b_DHHV
   head dim loop size:         siz_b_DHQK

   :param matQ: Tensor containing the query vectors. Shape (B, NH, S, DHQK).
   :param matK: Tensor containing the key vectors. Shape (B, NH, S, DHQK).
   :param matV: Tensor containing the value vectors. Shape (B, NH, S, DHHV).
   :param vecI: Tensor containing the input gate pre-activations. Shape (B, NH, NC, L).
   :param vecA: Tensor containing the summed input and cumulative forget gate pre-activations. Shape (B, NH, NC, L).
   :param vecB: Tensor containing the cumulative forget gate pre-activations. Shape (B, NH, NC, L).
   :param matC_all: Tensor containing the C states at the chunk borders. Shape (B, NH, (NC+1) * DHQK, DHHV).
   :param vecN_all: Tensor containing the N states at the chunk borders. Shape (B, NH, (NC+1) * DHQK).
   :param scaM_all: Tensor containing the M states at the chunk borders. Shape (B, NH, (NC+1)).
   :param vecN_out: Tensor containing the normalizer output N. Shape (B, NH, S).
   :param vecM_out: Tensor containing the max state M. Shape (B, NH, S).
   :param matDeltaH: Tensor containing the incoming H gradients. Shape (B, NH, S, DHHV).
   :param matDeltaC_states: Tensor containing the incoming C gradients. Shape (B, NH, (NC+1) * DHQK, DHHV).
   :param qk_scale: Scaling factor for the QK matrix. Defaults to None.
   :param chunk_size: Chunk size. Defaults to 64.
   :param siz_b_LQ: Block size for the chunk dimension LQ. Defaults to 32.
   :param siz_b_LKV: Block size for the chunk dimension LKV. Defaults to 32.
   :param siz_b_DHQK: Block size for the head dimension DHQK. Defaults to None.
   :param siz_b_DHHV: Block size for the head dimension DHHV. Defaults to None.
   :param num_warps: Number of warps. Defaults to None.
   :param num_stages: Number of stages. Defaults to None.
   :param eps: Epsilon value. Defaults to 1e-6.
   :param output_dtype: Output data type. Defaults to jnp.float32.

   :returns: Tensor containing the K gradients. Shape (B, NH, S, DHQK).


