xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.config_utils
=================================================================

.. py:module:: xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.config_utils


Classes
-------

.. autoapisummary::

   xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.config_utils.NameAndKwargs


Module Contents
---------------

.. py:class:: NameAndKwargs

   Bases: :py:obj:`xlstm_jax.models.configs.SubModelConfig`


   Sub-model configuration.

   This class is currently a quick fix to allow for post-init style model configs, like the xlstm-clean we ported from
   the original xlstm codebase. Once the config system is more mature, we should remove this and all becomes a subclass
   of ModelConfig.


   .. py:attribute:: name
      :type:  str
      :value: ''



   .. py:attribute:: kwargs
      :type:  dict[str, Any] | None


   .. py:attribute:: _registry
      :type:  dict[str, type] | None


   .. py:method:: get_config_class_for_kwargs()


   .. py:method:: get_class_for_name()


   .. py:method:: create_name_instance()


   .. py:method:: to_dict()

      Converts the config to a dictionary.

      Helpful for saving to disk or logging.



