xlstm_jax.models.xlstm_parallel.blocks.mlstm.cell
=================================================

.. py:module:: xlstm_jax.models.xlstm_parallel.blocks.mlstm.cell


Classes
-------

.. autoapisummary::

   xlstm_jax.models.xlstm_parallel.blocks.mlstm.cell.mLSTMCellConfig
   xlstm_jax.models.xlstm_parallel.blocks.mlstm.cell.mLSTMCell


Module Contents
---------------

.. py:class:: mLSTMCellConfig

   Bases: :py:obj:`xlstm_jax.models.configs.SubModelConfig`


   Sub-model configuration.

   This class is currently a quick fix to allow for post-init style model configs, like the xlstm-clean we ported from
   the original xlstm codebase. Once the config system is more mature, we should remove this and all becomes a subclass
   of ModelConfig.


   .. py:attribute:: context_length
      :type:  int
      :value: -1



   .. py:attribute:: embedding_dim
      :type:  int
      :value: -1



   .. py:attribute:: num_heads
      :type:  int
      :value: -1



   .. py:attribute:: backend
      :type:  xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.mLSTMBackendNameAndKwargs


   .. py:attribute:: norm_eps
      :type:  float
      :value: 1e-06


      Epsilon value for numerical stability in layer norm.


   .. py:attribute:: norm_type
      :type:  Literal['layernorm', 'rmsnorm']
      :value: 'layernorm'


      Type of normalization layer to use.


   .. py:attribute:: norm_type_v1
      :type:  Literal['layernorm', 'rmsnorm']
      :value: 'layernorm'


      this is only used in the 'mlstm_v1' layer_type. Due to a bug, the
      'norm_type' was not used correctly in the v1 version. To keep the same behavior, we introduce a separate parameter
      for the normalization layer.

      :type: Type of normalization layer to use. NOTE


   .. py:attribute:: dtype
      :type:  str
      :value: 'bfloat16'



   .. py:attribute:: gate_dtype
      :type:  str
      :value: 'float32'



   .. py:attribute:: gate_soft_cap
      :type:  float | None
      :value: None


      Soft cap for the gate pre-activations. If None, no cap is applied.


   .. py:attribute:: gate_linear_headwise
      :type:  bool
      :value: False


      If True, the gate pre-activations are computed with a linear headwise layer, similar to QKV.
      Otherwise, each gate head takes as input the full features across all heads.


   .. py:attribute:: igate_bias_init_range
      :type:  tuple[float, float] | float | None
      :value: None


      Input gate bias initialization. If a tuple, the bias is initialized with a linspace in the given range.
      If a float, the bias is initialized with the given value. If None, the bias is initialized with normal(0.1).


   .. py:attribute:: fgate_bias_init_range
      :type:  tuple[float, float] | float | None
      :value: (3.0, 6.0)


      Forget gate bias initialization. If a tuple, the bias is initialized with a linspace in the given range.
      If a float, the bias is initialized with the given value. If None, the bias is initialized with normal(0.1).


   .. py:attribute:: add_qk_norm
      :type:  bool
      :value: False


      If True, adds a normalization layer on the query and key vectors before the mLSTM cell.


   .. py:attribute:: reset_at_document_boundaries
      :type:  bool
      :value: False


      If True, the memory is reset at the beginning of each document.


   .. py:attribute:: reset_fgate_value
      :type:  float
      :value: -25.0


      Value to set the forget gate to at document boundaries.


   .. py:attribute:: parallel
      :type:  xlstm_jax.models.configs.ParallelConfig | None
      :value: None


      Parallel configuration for the mLSTM cell.


   .. py:property:: _dtype
      :type: jax.numpy.dtype


      Returns the real dtype instead of the str from configs.

      :returns: The jnp dtype corresponding to the string value.


   .. py:property:: _gate_dtype
      :type: jax.numpy.dtype



   .. py:method:: to_dict()

      Converts the config to a dictionary.

      Helpful for saving to disk or logging.



.. py:class:: mLSTMCell

   Bases: :py:obj:`flax.linen.Module`


   .. py:attribute:: config
      :type:  mLSTMCellConfig


