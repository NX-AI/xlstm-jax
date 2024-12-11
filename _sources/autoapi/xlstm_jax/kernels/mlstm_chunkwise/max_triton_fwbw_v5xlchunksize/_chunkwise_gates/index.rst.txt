xlstm_jax.kernels.mlstm_chunkwise.max_triton_fwbw_v5xlchunksize._chunkwise_gates
================================================================================

.. py:module:: xlstm_jax.kernels.mlstm_chunkwise.max_triton_fwbw_v5xlchunksize._chunkwise_gates

.. autoapi-nested-parse::

   In this file we compute the chunkwise or cumulative gates (i.e. vecA and vecB)
   for the forward and backward pass of the mLSTM.
   We use the stable formulations, i.e. we avoid subtraction of forget gates.



Functions
---------

.. autoapisummary::

   xlstm_jax.kernels.mlstm_chunkwise.max_triton_fwbw_v5xlchunksize._chunkwise_gates.compute_chunkwise_log_gates_vecB_vecA


Module Contents
---------------

.. py:function:: compute_chunkwise_log_gates_vecB_vecA(vecI, vecF, chunk_size, return_vecB_only = False)

