xlstm_jax.kernels.mlstm_chunkwise.max_triton_fwbw_v5xlchunksize._combined_fw
============================================================================

.. py:module:: xlstm_jax.kernels.mlstm_chunkwise.max_triton_fwbw_v5xlchunksize._combined_fw


Functions
---------

.. autoapisummary::

   xlstm_jax.kernels.mlstm_chunkwise.max_triton_fwbw_v5xlchunksize._combined_fw.mlstm_chunkwise_fw


Module Contents
---------------

.. py:function:: mlstm_chunkwise_fw(matQ, matK, matV, vecI, vecF, matC_initial = None, vecN_initial = None, scaM_initial = None, qk_scale = None, return_last_states = False, return_all_states = False, chunk_size_inter = None, chunk_size_intra = None, siz_b_L_parallel = None, siz_b_L_loop = None, siz_b_DH_parallel = None, siz_b_DH_loop = None, num_warps_intra = None, num_warps_inter = None, num_stages_intra = None, num_stages_inter = None, output_dtype = jnp.float32, eps = 0.0)

   Execute the forward pass of the mLSTM chunkwise formulation.

   :param matQ: Tensor containing the queries. Shape (B, NH, S, DHQK).
   :param matK: Tensor containing the keys. Shape (B, NH, S, DHQK).
   :param matV: Tensor containing the values. Shape (B, NH, S, DHV).
   :param vecI: Tensor containing the input gate. Shape (B, NH, S).
   :param vecF: Tensor containing the forget gate. Shape (B, NH, S).
   :param matC_initial: Initial state of the C matrix. Shape (B, NH, DHQK, DHV). Defaults to None.
   :param vecN_initial: Initial state of the N vector. Shape (B, NH, DHQK). Defaults to None.
   :param scaM_initial: Initial state of the M scalar. Shape (B, NH). Defaults to None.
   :param qk_scale: Scaling factor for the QK matrix. Defaults to None and will be inferred.
   :param return_last_states: Whether to return the last states. Defaults to False.
   :param return_all_states: Whether to return all states. Defaults to False.
   :param chunk_size_inter: Chunk size for the kernel inter chunk (recurrent) kernel. Defaults to None.
   :param chunk_size_intra: Chunk size for the kernel intra chunk (parallel) kernel. Defaults to None.
   :param siz_b_L_parallel: Size of the parallel L dimension for the parallel kernel. Defaults to None.
   :param siz_b_L_loop: Size of the loop L dimension for the parallel kernel. Defaults to None.
   :param siz_b_DH_parallel: Size of the parallel DH dimension for the parallel kernel. Defaults to None.
   :param siz_b_DH_loop: Size of the loop DH dimension for the parallel kernel. Defaults to None.
   :param num_warps_intra: Number of warps for the intra chunk kernel. Defaults to None.
   :param num_warps_inter: Number of warps for the inter chunk kernel. Defaults to None.
   :param num_stages_intra: Number of stages for the intra chunk kernel. Defaults to None.
   :param num_stages_inter: Number of stages for the inter chunk kernel. Defaults to None.
   :param output_dtype: Data type for the output. Defaults to jnp.float32.
   :param eps: Small value to avoid division by zero. Defaults to 1e-6.

   :returns: Tuple containing the output matrix H (shape (B, NH, S, DHV)), the N vector (shape (B, NH, S)),
             the M scalar (shape (B, NH)). Optionally, it might contain last states (matC_states,
             vecN_states, scaMinter_states) and optional all states (matC_states, vecN_states,
             scaMinter_states).


