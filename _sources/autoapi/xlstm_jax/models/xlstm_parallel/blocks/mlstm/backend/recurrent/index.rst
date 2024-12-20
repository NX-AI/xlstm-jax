xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.recurrent
==============================================================

.. py:module:: xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.recurrent


Classes
-------

.. autoapisummary::

   xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.recurrent.mLSTMBackendRecurrentConfig
   xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.recurrent.mLSTMBackendRecurrent


Functions
---------

.. autoapisummary::

   xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.recurrent.recurrent_step_fw
   xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.recurrent.recurrent_sequence_fw


Module Contents
---------------

.. py:function:: recurrent_step_fw(c, n, m, q, k, v, i, f, eps = 1e-06)

   This is a single step of the mLSTM operation in recurrent form.

   :param c: Memory state tensor of shape (B, NH, DHQK, DHV).
   :param n: Normalizer state tensor of shape (B, NH, DHQK).
   :param m: Max state tensor of shape (B, NH, 1).
   :param q: Queries tensor of shape (B, NH, DHQK).
   :param k: Keys tensor of shape (B, NH, DHQK).
   :param v: Values tensor of shape (B, NH, DHV).
   :param i: Input gate tensor of shape (B, NH, 1).
   :param f: Forget gate tensor of shape (B, NH, 1).
   :param eps: Used for building the forgetgate matrix. Defaults to 1e-6.

   :returns: The hidden state and the new states (matC_state_new, vecN_state_new, vecM_state_new).


.. py:function:: recurrent_sequence_fw(queries, keys, values, igate_preact, fgate_preact, c_initial = None, n_initial = None, m_initial = None, return_last_states = False, eps = 1e-06, state_dtype = None, use_scan = False, mlstm_step_fn = recurrent_step_fw)

   Forward pass of the mLSTM cell in recurrent form on a full sequence.

   :param queries: Queries tensor of shape (B, NH, S, DHQK).
   :param keys: Keys tensor of shape (B, NH, S, DHQK).
   :param values: Values tensor of shape (B, NH, S, DHV).
   :param igate_preact: Input gate pre-activation tensor of shape (B, NH, S, 1).
   :param fgate_preact: Forget gate pre-activation tensor of shape (B, NH, S, 1).
   :param c_initial: Initial memory state tensor of shape (B, NH, DHQK, DHV). If None, initialized to zeros.
   :param n_initial: Initial normalizer state tensor of shape (B, NH, DHQK). If None, initialized to zeros.
   :param m_initial: Initial max state tensor of shape (B, NH). If None, initialized to zeros.
   :param return_last_states: Whether to return the last states. Defaults to False.
   :param eps: Epsilon value for numerical stability. Defaults to 1e-6.
   :param state_dtype: Dtype of the states. If None, uses the dtype of the initial states if provided, or other
                       the dtypes of the pre-activations. If initial states are provided, the return dtype will be the same as the
                       initial states. Defaults to None.
   :param use_scan: Whether to use `jax.lax.scan` for the loop. The scan reduces compilation time, but may be slower for
                    kernels without XLA compiler support and introduces memory copy overhead.
   :param mlstm_step_fn: Function to compute a single mLSTM step. By default, set to `recurrent_step_fw` in this backend.

   :returns: Hidden states tensor of shape (B, NH, S, DHV) if `return_last_states` is False.
             Tuple of hidden states tensor and tuple of last states tensors if `return_last_states` is True.


.. py:class:: mLSTMBackendRecurrentConfig

   .. py:attribute:: context_length
      :type:  int
      :value: -1



   .. py:attribute:: eps
      :type:  float
      :value: 1e-06



   .. py:attribute:: state_dtype
      :type:  str | None
      :value: None



   .. py:attribute:: use_scan
      :type:  bool
      :value: False



   .. py:method:: assign_model_config_params(model_config)


.. py:class:: mLSTMBackendRecurrent

   Bases: :py:obj:`xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.config.mLSTMBackend`


   .. py:attribute:: config_class


   .. py:property:: can_vmap_over_heads
      :type: bool


      Whether the backend can be vmaped over the heads dimension.

      The backend was not written to be independent of the heads dimension, and thus cannot be vmapped.

      :returns: False
      :rtype: bool


