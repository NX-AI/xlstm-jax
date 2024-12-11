xlstm_jax.kernels.mlstm_chunkwise.triton_stablef.triton_fwbw
============================================================

.. py:module:: xlstm_jax.kernels.mlstm_chunkwise.triton_stablef.triton_fwbw

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

   xlstm_jax.kernels.mlstm_chunkwise.triton_stablef.triton_fwbw._mlstm_chunkwise_fwbw_generator
   xlstm_jax.kernels.mlstm_chunkwise.triton_stablef.triton_fwbw._get_chunkwise_fwbw_kernel
   xlstm_jax.kernels.mlstm_chunkwise.triton_stablef.triton_fwbw.mlstm_chunkwise_triton_stablef


Module Contents
---------------

.. py:function:: _mlstm_chunkwise_fwbw_generator(autocast_kernel_dtype = jnp.bfloat16, return_last_states = False, recompute_states_in_bw = True, chunk_size = 64, eps = 1e-06, stabilize_correctly = True, norm_val = 1.0)

   Generate a forward and backward pass function for the mLSTM kernels with chunkwise formulation.

   :param autocast_kernel_dtype: The dtype to use for the kernel computation. All inputs arguments up to vecF
                                 are cast to this dtype. vecF is automatically casted to float32 in the kernels.
   :param return_last_states: Whether to return the last states of the mLSTM.
   :param recompute_states_in_bw: Whether to recompute the mLSTM states in the backward pass.
   :param chunk_size: The chunk size to use for the mLSTM computation.
   :param eps: The epsilon value to use for numerical stability.
   :param stabilize_correctly: Whether to stabilize with max(norm_val*e^-m, |nk|) instead of max(norm_val, |nk|)
   :param norm_val: Norm scale in the max formula above

   :returns: A function that computes the forward pass of the mLSTM chunkwise formulation, which custom gradients for the
             backward pass. The function input signatures is:

                 forward(
                     matQ: jax.Array,  # (B, NH, S, DHQK)
                     matK: jax.Array,  # (B, NH, S, DHQK)
                     matV: jax.Array,  # (B, NH, S, DHV)
                     vecI: jax.Array,  # (B, NH, S)
                     vecF: jax.Array,  # (B, NH, S)
                     matC_initial: jax.Array | None = None,  # (B, NH, DHQK, DHV)
                     vecN_initial: jax.Array | None = None,  # (B, NH, DHQK)
                     scaM_initial: jax.Array | None = None,  # (B, NH)
                 ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
             The function returns the output of the mLSTM computation, and the last states internal states of C, N and M.


.. py:function:: _get_chunkwise_fwbw_kernel(autocast_kernel_dtype, **kwargs)

   Get the forward and backward pass function for the mLSTM kernels with chunkwise formulation.

   :param autocast_kernel_dtype: The dtype to use for the kernel computation. All inputs arguments up to vecF and vecI
                                 are cast to this dtype. vecF is automatically casted to float32 in the kernels.
   :param \*\*kwargs: Additional keyword arguments to pass to the kernel function.

   :returns: A function that computes the forward pass of the mLSTM chunkwise formulation, which custom gradients for the
             backward pass. See _mlstm_chunkwise_fwbw_generator for the function signature.


.. py:function:: mlstm_chunkwise_triton_stablef(q, k, v, i, f, c_initial = None, n_initial = None, m_initial = None, return_last_states = False, eps = 1e-06, chunk_size = 64, autocast_kernel_dtype = jnp.float32, stabilize_correctly = True, norm_val = 1.0)

   Apply the mLSTM chunkwise formulation with Triton kernels.

   Supports autograd application.

   :param q: The query tensor of shape (B, NH, S, DHQK).
   :param k: The key tensor of shape (B, NH, S, DHQK).
   :param v: The value tensor of shape (B, NH, S, DHV).
   :param i: The input gate preactivation tensor of shape (B, NH, S).
   :param f: The forget gate preactivation tensor of shape (B, NH, S).
   :param c_initial: The initial chunk state tensor of shape (B, NH, DHQK, DHV).
   :param n_initial: The initial chunk state tensor of shape (B, NH, DHQK).
   :param m_initial: The initial chunk state tensor of shape (B, NH).
   :param return_last_states: Whether to return the last states of the mLSTM.
   :param eps: The epsilon value to use for numerical stability.
   :param chunk_size: The chunk size to use for the mLSTM computation.
   :param autocast_kernel_dtype: The dtype to use for the kernel computation. All inputs arguments up
                                 to vecF are cast to this dtype. vecF is automatically casted to float32 in the kernels.
   :param stabilize_correctly: Whether to stabilize with max(norm_val*e^-m, |nk|) instead of max(norm_val, |nk|)
   :param norm_val: Norm scale in the max formula above

   :returns: The output of the mLSTM computation. If return_last_states is True, the last states of the
             mLSTM are also returned.


