xlstm_jax.models.xlstm_pytorch.blocks.slstm.src.vanilla
=======================================================

.. py:module:: xlstm_jax.models.xlstm_pytorch.blocks.slstm.src.vanilla


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/xlstm_jax/models/xlstm_pytorch/blocks/slstm/src/vanilla/lstm/index
   /autoapi/xlstm_jax/models/xlstm_pytorch/blocks/slstm/src/vanilla/slstm/index


Attributes
----------

.. autoapisummary::

   xlstm_jax.models.xlstm_pytorch.blocks.slstm.src.vanilla.slstm_pointwise_function_registry


Functions
---------

.. autoapisummary::

   xlstm_jax.models.xlstm_pytorch.blocks.slstm.src.vanilla.slstm_forward
   xlstm_jax.models.xlstm_pytorch.blocks.slstm.src.vanilla.slstm_forward_step


Package Contents
----------------

.. py:data:: slstm_pointwise_function_registry
   :type:  dict[str, collections.abc.Callable]

.. py:function:: slstm_forward(x, states, R, b, pointwise_forward, constants = None)

.. py:function:: slstm_forward_step(x, states, R, b, pointwise_forward, constants = None)

