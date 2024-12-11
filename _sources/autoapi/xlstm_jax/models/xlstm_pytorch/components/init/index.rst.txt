xlstm_jax.models.xlstm_pytorch.components.init
==============================================

.. py:module:: xlstm_jax.models.xlstm_pytorch.components.init


Functions
---------

.. autoapisummary::

   xlstm_jax.models.xlstm_pytorch.components.init.bias_linspace_init_
   xlstm_jax.models.xlstm_pytorch.components.init.small_init_init_
   xlstm_jax.models.xlstm_pytorch.components.init.wang_init_


Module Contents
---------------

.. py:function:: bias_linspace_init_(param, start = 3.4, end = 6.0)

   Linearly spaced bias init across dimensions.


.. py:function:: small_init_init_(param, dim)

   Fills the input Tensor with values according to the method described in Transformers without Tears: Improving
   the Normalization of Self-Attention - Nguyen, T. & Salazar, J. (2019), using a normal distribution.
   Adopted from https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/init_functions.py.


.. py:function:: wang_init_(param, dim, num_blocks)

   Adopted from https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/init_functions.py.


