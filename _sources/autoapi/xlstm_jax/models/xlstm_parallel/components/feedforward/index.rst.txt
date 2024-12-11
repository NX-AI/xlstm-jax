xlstm_jax.models.xlstm_parallel.components.feedforward
======================================================

.. py:module:: xlstm_jax.models.xlstm_parallel.components.feedforward


Attributes
----------

.. autoapisummary::

   xlstm_jax.models.xlstm_parallel.components.feedforward._act_fn_registry


Classes
-------

.. autoapisummary::

   xlstm_jax.models.xlstm_parallel.components.feedforward.FeedForwardConfig
   xlstm_jax.models.xlstm_parallel.components.feedforward.GatedFeedForward
   xlstm_jax.models.xlstm_parallel.components.feedforward.FeedForward


Functions
---------

.. autoapisummary::

   xlstm_jax.models.xlstm_parallel.components.feedforward.get_act_fn
   xlstm_jax.models.xlstm_parallel.components.feedforward.create_feedforward


Module Contents
---------------

.. py:data:: _act_fn_registry

.. py:function:: get_act_fn(act_fn_name)

.. py:class:: FeedForwardConfig

   Bases: :py:obj:`xlstm_jax.models.xlstm_parallel.utils.UpProjConfigMixin`


   Sub-model configuration.

   This class is currently a quick fix to allow for post-init style model configs, like the xlstm-clean we ported from
   the original xlstm codebase. Once the config system is more mature, we should remove this and all becomes a subclass
   of ModelConfig.


   .. py:attribute:: proj_factor
      :type:  float
      :value: 1.3



   .. py:attribute:: act_fn
      :type:  str
      :value: 'gelu'



   .. py:attribute:: embedding_dim
      :type:  int
      :value: -1



   .. py:attribute:: dropout
      :type:  float
      :value: 0.0



   .. py:attribute:: bias
      :type:  bool
      :value: False



   .. py:attribute:: init_distribution
      :type:  xlstm_jax.models.shared.InitDistribution
      :value: 'normal'


      Distribution type from which to sample the weights.


   .. py:attribute:: output_init_fn
      :type:  xlstm_jax.models.shared.InitFnName
      :value: 'wang'


      Initialization function for the output projection layer.


   .. py:attribute:: ff_type
      :type:  Literal['ffn_gated', 'ffn']
      :value: 'ffn_gated'



   .. py:attribute:: dtype
      :type:  str
      :value: 'bfloat16'



   .. py:attribute:: parallel
      :type:  xlstm_jax.models.configs.ParallelConfig | None
      :value: None



   .. py:attribute:: _num_blocks
      :type:  int
      :value: 1



   .. py:property:: _dtype
      :type: jax.numpy.dtype


      Returns the real dtype instead of the str from configs.

      :returns: The jnp dtype corresponding to the string value.


   .. py:attribute:: round_proj_up_dim_up
      :type:  bool
      :value: True



   .. py:attribute:: round_proj_up_to_multiple_of
      :type:  int
      :value: 64



   .. py:attribute:: _proj_up_dim
      :type:  int | None
      :value: None



   .. py:method:: _set_proj_up_dim(embedding_dim)


   .. py:method:: to_dict()

      Converts the config to a dictionary.

      Helpful for saving to disk or logging.



.. py:class:: GatedFeedForward

   Bases: :py:obj:`flax.linen.Module`


   .. py:attribute:: config
      :type:  FeedForwardConfig


.. py:class:: FeedForward

   Bases: :py:obj:`flax.linen.Module`


   .. py:attribute:: config
      :type:  FeedForwardConfig


.. py:function:: create_feedforward(config, name = 'ffn')

