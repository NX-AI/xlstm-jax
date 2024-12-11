xlstm_jax.kernels.mlstm_chunkwise.max_triton_fwbw_v3.triton_fwbw
================================================================

.. py:module:: xlstm_jax.kernels.mlstm_chunkwise.max_triton_fwbw_v3.triton_fwbw

.. autoapi-nested-parse::

   Triton backend for the forward and backward pass of the mLSTM chunkwise formulation.

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

   xlstm_jax.kernels.mlstm_chunkwise.max_triton_fwbw_v3.triton_fwbw._mlstm_chunkwise_fwbw_generator
   xlstm_jax.kernels.mlstm_chunkwise.max_triton_fwbw_v3.triton_fwbw._get_chunkwise_fwbw_kernel
   xlstm_jax.kernels.mlstm_chunkwise.max_triton_fwbw_v3.triton_fwbw.mlstm_chunkwise_max_triton


Module Contents
---------------

.. py:function:: _mlstm_chunkwise_fwbw_generator(autocast_kernel_dtype = jnp.bfloat16, return_last_states = False, recompute_states_in_bw = True, chunk_size = 64, eps = 1e-06, reduce_slicing = False)

   Generate a forward and backward pass function for the mLSTM kernels with chunkwise formulation.

   :param autocast_kernel_dtype: The dtype to use for the kernel computation. All inputs arguments up to vecF
                                 are cast to this dtype. vecF is automatically casted to float32 in the kernels.
   :param return_last_states: Whether to return the last states of the mLSTM.
   :param recompute_states_in_bw: Whether to recompute the mLSTM states in the backward pass.
   :param chunk_size: The chunk size to use for the mLSTM computation.
   :param eps: The epsilon value to use for numerical stability.
   :param reduce_slicing: If True, reduces the slicing operations taken in the preprocessing to
                          the kernel. This leads to performance improvements during training while returning
                          the same results. Defaults to False.

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

   :param autocast_kernel_dtype: The dtype to use for the kernel computation. All inputs arguments up to vecF
                                 are cast to this dtype. vecF is automatically casted to float32 in the kernels.
   :param \*\*kwargs: Additional keyword arguments to pass to the kernel function.

   :returns: A function that computes the forward pass of the mLSTM chunkwise formulation, which custom gradients for the
             backward pass. See _mlstm_chunkwise_fwbw_generator for the function signature.


.. py:function:: mlstm_chunkwise_max_triton(q, k, v, i, f, c_initial = None, n_initial = None, m_initial = None, return_last_states = False, eps = 1e-06, chunk_size = 64, autocast_kernel_dtype = jnp.float32, reduce_slicing = False)

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
   :param reduce_slicing: If True, reduces the slicing operations taken in the preprocessing to
                          the kernel. This leads to performance improvements during training while returning
                          the same results. Defaults to False.

   :returns: The output of the mLSTM computation. If return_last_states is True, the last states of the
             mLSTM are also returned.


