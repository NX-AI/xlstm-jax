xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.fwbw
=========================================================

.. py:module:: xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.fwbw


Classes
-------

.. autoapisummary::

   xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.fwbw.mLSTMBackendFwbwConfig
   xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.fwbw.mLSTMBackendFwbw


Functions
---------

.. autoapisummary::

   xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.fwbw.rev_cumsum_off
   xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.fwbw.rev_cumsum
   xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.fwbw.causal_forget_matrix
   xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.fwbw.fwbw_forward
   xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.fwbw.fwbw_backward
   xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.fwbw.mlstm_fwbw_custom_grad


Module Contents
---------------

.. py:class:: mLSTMBackendFwbwConfig

   Configuration object for the mLSTM fwbw backend.


   .. py:attribute:: chunk_size
      :type:  int | None
      :value: 512


      Chunk size for the kernel computation.


   .. py:attribute:: return_state
      :type:  bool
      :value: False


      Whether to return the last state. USeful for inference.


   .. py:attribute:: use_initial_state
      :type:  bool
      :value: False


      Whether to start from an initial state or zeros.


   .. py:attribute:: keep_G
      :type:  bool
      :value: False


      Whether to save the G matrix for the backward pass.


   .. py:attribute:: keep_gates
      :type:  bool
      :value: True


      Whether to save the gates for the backward pass.


   .. py:attribute:: keep_M
      :type:  bool
      :value: False


      Whether to save the M matrix for the backward pass.


   .. py:attribute:: keep_c
      :type:  bool
      :value: False


      Whether to save the c matrix for the backward pass.


   .. py:attribute:: stabilize_correctly
      :type:  bool
      :value: False


      Whether to stabilize the output correctly. This is only needed if no GroupNorm is applied after the mLSTM.
      If GroupNorm is applied, this can be set to False, as results after GroupNorm will be the same.


   .. py:method:: assign_model_config_params(model_config)


.. py:function:: rev_cumsum_off(x)

   Compute the reverse cumulative sum of a tensor with an offset.


.. py:function:: rev_cumsum(x)

   Compute the reverse cumulative sum of a tensor.


.. py:function:: causal_forget_matrix(forget_gates)

   Compute the causal forget matrix from the forget gates.


.. py:function:: fwbw_forward(q, k, v, i, f, config, initial_C = None, initial_n = None, initial_m = None)

   Forward pass of the mLSTM fwbw backend.

   :param q: query tensor
   :param k: key tensor
   :param v: value tensor
   :param i: input gate tensor
   :param f: forget gate tensor
   :param config: configuration object
   :param initial_C: initial chunk tensor. Defaults to None.
   :param initial_n: initial n tensor. Defaults to None.
   :param initial_m: initial m tensor. Defaults to None.

   :returns: Output tensor and context for backward.


.. py:function:: fwbw_backward(ctx, dh, config, dc_last = None, dn_last = None, dm_last = None)

   Backward pass of the mLSTM fwbw backend.

   :param ctx: context from forward pass.
   :type ctx: Sequence[jax.Array]
   :param dh: gradient tensor.
   :type dh: jax.Array
   :param config: configuration object.
   :type config: mLSTMfwbwConfig
   :param dc_last: last chunk tensor. Defaults to None.
   :type dc_last: jax.Array, optional
   :param dn_last: last n tensor. Defaults to None.
   :type dn_last: jax.Array, optional
   :param dm_last: last m tensor. Defaults to None.
   :type dm_last: jax.Array, optional

   :returns:

             tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array,
                   jax.Array | None, jax.Array | None, jax.Array | None]: gradients.


.. py:function:: mlstm_fwbw_custom_grad(config)

   Returns an autograd function that computes the gradient itself.

   :param config: configuration object.
   :type config: mLSTMfwbwConfig

   :returns: autograd function.
   :rtype: function


.. py:class:: mLSTMBackendFwbw

   Bases: :py:obj:`xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.config.mLSTMBackend`


   .. py:attribute:: config_class


   .. py:property:: can_vmap_over_heads
      :type: bool


      Whether the backend can be vmaped over the heads dimension.

      The backend is written independent of the heads dimension, and thus can be vmapped.

      :returns: True
      :rtype: bool


