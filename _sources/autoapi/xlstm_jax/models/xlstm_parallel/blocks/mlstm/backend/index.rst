xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend
====================================================

.. py:module:: xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/xlstm_jax/models/xlstm_parallel/blocks/mlstm/backend/attention/index
   /autoapi/xlstm_jax/models/xlstm_parallel/blocks/mlstm/backend/config/index
   /autoapi/xlstm_jax/models/xlstm_parallel/blocks/mlstm/backend/config_utils/index
   /autoapi/xlstm_jax/models/xlstm_parallel/blocks/mlstm/backend/fwbw/index
   /autoapi/xlstm_jax/models/xlstm_parallel/blocks/mlstm/backend/layer_factory/index
   /autoapi/xlstm_jax/models/xlstm_parallel/blocks/mlstm/backend/recurrent/index
   /autoapi/xlstm_jax/models/xlstm_parallel/blocks/mlstm/backend/recurrent_triton/index
   /autoapi/xlstm_jax/models/xlstm_parallel/blocks/mlstm/backend/simple/index
   /autoapi/xlstm_jax/models/xlstm_parallel/blocks/mlstm/backend/triton_kernels/index


Attributes
----------

.. autoapisummary::

   xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend._mlstm_backend_registry
   xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.BackendType
   xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.create_mlstm_backend


Classes
-------

.. autoapisummary::

   xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.mLSTMBackendNameAndKwargs


Package Contents
----------------

.. py:data:: _mlstm_backend_registry

.. py:data:: BackendType

.. py:class:: mLSTMBackendNameAndKwargs

   Bases: :py:obj:`config_utils.NameAndKwargs`


   Sub-model configuration.

   This class is currently a quick fix to allow for post-init style model configs, like the xlstm-clean we ported from
   the original xlstm codebase. Once the config system is more mature, we should remove this and all becomes a subclass
   of ModelConfig.


   .. py:attribute:: _registry
      :type:  dict[str, type]


   .. py:attribute:: name
      :type:  str
      :value: ''



   .. py:attribute:: kwargs
      :type:  dict[str, Any] | None


   .. py:method:: get_config_class_for_kwargs()


   .. py:method:: get_class_for_name()


   .. py:method:: create_name_instance()


   .. py:method:: to_dict()

      Converts the config to a dictionary.

      Helpful for saving to disk or logging.



.. py:data:: create_mlstm_backend

