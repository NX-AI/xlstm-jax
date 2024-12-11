xlstm_jax.models.xlstm_pytorch.utils
====================================

.. py:module:: xlstm_jax.models.xlstm_pytorch.utils


Classes
-------

.. autoapisummary::

   xlstm_jax.models.xlstm_pytorch.utils.UpProjConfigMixin
   xlstm_jax.models.xlstm_pytorch.utils.WeightDecayOptimGroupMixin


Module Contents
---------------

.. py:class:: UpProjConfigMixin

   .. py:attribute:: proj_factor
      :type:  float
      :value: None



   .. py:attribute:: round_proj_up_dim_up
      :type:  bool
      :value: True



   .. py:attribute:: round_proj_up_to_multiple_of
      :type:  int
      :value: 64



   .. py:attribute:: _proj_up_dim
      :type:  int
      :value: None



   .. py:method:: _set_proj_up_dim(embedding_dim)


.. py:class:: WeightDecayOptimGroupMixin

   Bases: :py:obj:`torch.nn.Module`, :py:obj:`abc.ABC`


   Helper class that provides a standard way to create an ABC using
   inheritance.


   .. py:method:: get_weight_decay_optim_groups()

      Return a tuple of two sequences, one for parameters with weight decay and one for parameters without weight
      decay.

      Performs checks to ensure that each parameter is only in one of the two sequences.



   .. py:method:: get_weight_decay_optim_group_param_names()

      Return a tuple of two sequences, one for parameter names with weight decay and one for parameter names without
      weight decay.

      Performs checks to ensure that each parameter is only in one of the two sequences.



   .. py:method:: _create_weight_decay_optim_groups()

      Return a tuple of two sequences, one for parameters with weight decay and one for parameters without weight
      decay.

      Default separation:
      - weight decay: all parameters which have > 1 dimensions.
      - no weight decay: all parameters which have = 1 dimension, e.g. biases.



   .. py:method:: _get_weight_decay_optim_groups_for_modules(modules, **kwargs)
      :staticmethod:



