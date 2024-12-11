xlstm_jax.models.xlstm_pytorch.blocks.mlstm.backend.tl_utils
============================================================

.. py:module:: xlstm_jax.models.xlstm_pytorch.blocks.mlstm.backend.tl_utils


Classes
-------

.. autoapisummary::

   xlstm_jax.models.xlstm_pytorch.blocks.mlstm.backend.tl_utils.CumsumFunction
   xlstm_jax.models.xlstm_pytorch.blocks.mlstm.backend.tl_utils.ReversedCumsumFunction


Functions
---------

.. autoapisummary::

   xlstm_jax.models.xlstm_pytorch.blocks.mlstm.backend.tl_utils.contiguous
   xlstm_jax.models.xlstm_pytorch.blocks.mlstm.backend.tl_utils.checkpoint
   xlstm_jax.models.xlstm_pytorch.blocks.mlstm.backend.tl_utils.logcumsumexp_fwd_kernel
   xlstm_jax.models.xlstm_pytorch.blocks.mlstm.backend.tl_utils.softmax_fwd_kernel
   xlstm_jax.models.xlstm_pytorch.blocks.mlstm.backend.tl_utils.softmax_bwd_kernel
   xlstm_jax.models.xlstm_pytorch.blocks.mlstm.backend.tl_utils.recurrent_cumsum_fwd_kernel
   xlstm_jax.models.xlstm_pytorch.blocks.mlstm.backend.tl_utils.recurrent_cumsum_bwd_kernel
   xlstm_jax.models.xlstm_pytorch.blocks.mlstm.backend.tl_utils.chunk_cumsum_fwd_kernel
   xlstm_jax.models.xlstm_pytorch.blocks.mlstm.backend.tl_utils.chunk_cumsum_bwd_kernel
   xlstm_jax.models.xlstm_pytorch.blocks.mlstm.backend.tl_utils.chunk_cumsum_fwd
   xlstm_jax.models.xlstm_pytorch.blocks.mlstm.backend.tl_utils.chunk_cumsum_bwd
   xlstm_jax.models.xlstm_pytorch.blocks.mlstm.backend.tl_utils.cumsum
   xlstm_jax.models.xlstm_pytorch.blocks.mlstm.backend.tl_utils.recurrent_reversed_cumsum_fwd_kernel
   xlstm_jax.models.xlstm_pytorch.blocks.mlstm.backend.tl_utils.recurrent_reversed_cumsum_bwd_kernel
   xlstm_jax.models.xlstm_pytorch.blocks.mlstm.backend.tl_utils.chunk_reversed_cumsum_fwd_kernel
   xlstm_jax.models.xlstm_pytorch.blocks.mlstm.backend.tl_utils.chunk_reversed_cumsum_bwd_kernel
   xlstm_jax.models.xlstm_pytorch.blocks.mlstm.backend.tl_utils.chunk_reversed_cumsum_fwd
   xlstm_jax.models.xlstm_pytorch.blocks.mlstm.backend.tl_utils.chunk_reversed_cumsum_bwd
   xlstm_jax.models.xlstm_pytorch.blocks.mlstm.backend.tl_utils.reversed_cumsum


Module Contents
---------------

.. py:function:: contiguous(fn)

.. py:function:: checkpoint(func)

.. py:function:: logcumsumexp_fwd_kernel(s, z, s_s_h, s_s_t, s_s_d, T, S, BT)

.. py:function:: softmax_fwd_kernel(s, p, s_s_h, s_s_t, s_s_d, T, S, BT)

.. py:function:: softmax_bwd_kernel(p, dp, ds, s_s_h, s_s_t, s_s_d, T, S, BT)

.. py:function:: recurrent_cumsum_fwd_kernel(s, z, s_s_h, s_s_t, T, S, BS)

.. py:function:: recurrent_cumsum_bwd_kernel(ds, dz, s_s_h, s_s_t, T, S, BS)

.. py:function:: chunk_cumsum_fwd_kernel(s, z, s_s_h, s_s_t, s_s_d, T, S, BT, BS)

.. py:function:: chunk_cumsum_bwd_kernel(ds, dz, s_s_h, s_s_t, s_s_d, T, S, BT, BS)

.. py:function:: chunk_cumsum_fwd(s, dtype = None)

.. py:function:: chunk_cumsum_bwd(dz, dtype = None)

.. py:class:: CumsumFunction

   Bases: :py:obj:`torch.autograd.Function`


   .. py:method:: forward(ctx, s, dtype)
      :staticmethod:



   .. py:method:: backward(ctx, dz)
      :staticmethod:



.. py:function:: cumsum(s, dtype = None)

.. py:function:: recurrent_reversed_cumsum_fwd_kernel(s, z, s_s_h, s_s_t, T, S, BS)

.. py:function:: recurrent_reversed_cumsum_bwd_kernel(ds, dz, s_s_h, s_s_t, T, S, BS)

.. py:function:: chunk_reversed_cumsum_fwd_kernel(s, z, s_s_h, s_s_t, s_s_d, T, S, BT, BS)

.. py:function:: chunk_reversed_cumsum_bwd_kernel(ds, dz, s_s_h, s_s_t, s_s_d, T, S, BT, BS)

.. py:function:: chunk_reversed_cumsum_fwd(s, dtype = None)

.. py:function:: chunk_reversed_cumsum_bwd(dz, dtype = None)

.. py:class:: ReversedCumsumFunction

   Bases: :py:obj:`torch.autograd.Function`


   .. py:method:: forward(ctx, s, dtype)
      :staticmethod:



   .. py:method:: backward(ctx, dz)
      :staticmethod:



.. py:function:: reversed_cumsum(s, dtype = None)

