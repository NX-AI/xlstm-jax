xlstm_jax.kernels.mlstm_chunkwise.max_triton_fwbw_v5xlchunksize._parallel_fw
============================================================================

.. py:module:: xlstm_jax.kernels.mlstm_chunkwise.max_triton_fwbw_v5xlchunksize._parallel_fw


Functions
---------

.. autoapisummary::

   xlstm_jax.kernels.mlstm_chunkwise.max_triton_fwbw_v5xlchunksize._parallel_fw.mlstm_chunkwise__parallel_fw_Hintra


Module Contents
---------------

.. py:function:: mlstm_chunkwise__parallel_fw_Hintra(matQ, matK, matV, vecI, vecF, matC_all, vecN_all, scaM_all, qk_scale = None, chunk_size = 64, siz_b_LQ = 32, siz_b_LKV = 32, siz_b_DHQK = None, siz_b_DHHV = None, num_warps = None, num_stages = None, eps = 1e-06, output_dtype = jnp.float32)

   Execute the parallel forward kernel for the H computation in the mLSTM chunkwise formulation.

   This function defines the grid and block sizes for the kernel launch and calls the kernel. See
   the fwbw backend implementation and the triton kernels for more information.
   chunk parallel size:        siz_b_LQ
   chunk loop size:            siz_b_LKV
   head dim parallel size:     siz_b_DHHV
   head dim loop size:         siz_b_DHQK

   :param matQ: Tensor containing the queries. Shape (B, NH, S, DHQK).
   :param matK: Tensor containing the keys. Shape (B, NH, S, DHQK).
   :param matV: Tensor containing the values. Shape (B, NH, S, DHHV).
   :param vecI: Tensor containing the input gate. Shape (B, NH, NC, L).
   :param vecF: Tensor containing the forget gate preactivations. Shape (B, NH, NC * L) = (B, NH, S).
   :param matC_all: Tensor containing the C states at the chunk borders. Shape (B, NH, (NC+1) * DHQK, DHHV).
   :param vecN_all: Tensor containing the N states at the chunk borders. Shape (B, NH, (NC+1) * DHQK).
   :param scaM_all: Tensor containing the M states at the chunk borders. Shape (B, NH, (NC+1)).
   :param qk_scale: Scaling factor for the QK matrix. Defaults to None and will be inferred.
   :param chunk_size: Chunk size. Defaults to 64.
   :param siz_b_LQ: Block size for the chunk dimension LQ. Defaults to 32.
   :param siz_b_LKV: Block size for the chunk dimension LKV. Defaults to 32.
   :param siz_b_DHQK: Block size for the head dimension DHQK. Defaults to None.
   :param siz_b_DHHV: Block size for the head dimension DHHV. Defaults to None.
   :param num_warps: Number of warps. Defaults to None.
   :param num_stages: Number of stages. Defaults to None.
   :param eps: Epsilon value. Defaults to 1e-6.
   :param output_dtype: Output data type. Defaults to jnp.float32.

   :returns: Tuple containing the output matrix H (shape (B, NH, S, DHHV)) and the N vector (shape (B, NH, S)).


