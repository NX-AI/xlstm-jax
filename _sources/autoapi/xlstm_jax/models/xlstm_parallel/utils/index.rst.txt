xlstm_jax.models.xlstm_parallel.utils
=====================================

.. py:module:: xlstm_jax.models.xlstm_parallel.utils


Classes
-------

.. autoapisummary::

   xlstm_jax.models.xlstm_parallel.utils.UpProjConfigMixin


Module Contents
---------------

.. py:class:: UpProjConfigMixin

   Bases: :py:obj:`xlstm_jax.models.configs.SubModelConfig`


   Sub-model configuration.

   This class is currently a quick fix to allow for post-init style model configs, like the xlstm-clean we ported from
   the original xlstm codebase. Once the config system is more mature, we should remove this and all becomes a subclass
   of ModelConfig.


   .. py:attribute:: proj_factor
      :type:  float | None
      :value: None



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



