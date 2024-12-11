xlstm_jax.models.xlstm_pytorch.blocks.mlstm.backend
===================================================

.. py:module:: xlstm_jax.models.xlstm_pytorch.blocks.mlstm.backend


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/xlstm_jax/models/xlstm_pytorch/blocks/mlstm/backend/config/index
   /autoapi/xlstm_jax/models/xlstm_pytorch/blocks/mlstm/backend/config_utils/index
   /autoapi/xlstm_jax/models/xlstm_pytorch/blocks/mlstm/backend/fwbw/index
   /autoapi/xlstm_jax/models/xlstm_pytorch/blocks/mlstm/backend/layer_factory/index
   /autoapi/xlstm_jax/models/xlstm_pytorch/blocks/mlstm/backend/simple/index
   /autoapi/xlstm_jax/models/xlstm_pytorch/blocks/mlstm/backend/tl_utils/index
   /autoapi/xlstm_jax/models/xlstm_pytorch/blocks/mlstm/backend/triton_chunk/index


Attributes
----------

.. autoapisummary::

   xlstm_jax.models.xlstm_pytorch.blocks.mlstm.backend._mlstm_backend_registry
   xlstm_jax.models.xlstm_pytorch.blocks.mlstm.backend.create_mlstm_backend


Classes
-------

.. autoapisummary::

   xlstm_jax.models.xlstm_pytorch.blocks.mlstm.backend.mLSTMBackendNameAndKwargs


Package Contents
----------------

.. py:data:: _mlstm_backend_registry

.. py:class:: mLSTMBackendNameAndKwargs

   Bases: :py:obj:`config_utils.NameAndKwargs`


   .. py:attribute:: _registry
      :type:  dict[str, type]


.. py:data:: create_mlstm_backend

