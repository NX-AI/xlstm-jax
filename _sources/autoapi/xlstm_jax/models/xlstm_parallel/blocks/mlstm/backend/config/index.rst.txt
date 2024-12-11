xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.config
===========================================================

.. py:module:: xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.config


Classes
-------

.. autoapisummary::

   xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.config.mLSTMBackend


Module Contents
---------------

.. py:class:: mLSTMBackend

   Bases: :py:obj:`flax.linen.Module`


   .. py:attribute:: config
      :type:  Any


   .. py:property:: can_vmap_over_heads
      :type: bool

      :abstractmethod:


      Whether the backend can be vmaped over the heads dimension.

      Should be False if the backend requires manual transposition of the input tensors to work over heads.


