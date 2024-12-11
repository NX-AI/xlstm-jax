xlstm_jax.models.xlstm_clean.components.init
============================================

.. py:module:: xlstm_jax.models.xlstm_clean.components.init


Functions
---------

.. autoapisummary::

   xlstm_jax.models.xlstm_clean.components.init.bias_linspace_init
   xlstm_jax.models.xlstm_clean.components.init.small_init
   xlstm_jax.models.xlstm_clean.components.init.wang_init
   xlstm_jax.models.xlstm_clean.components.init.uniform_init


Module Contents
---------------

.. py:function:: bias_linspace_init(start, end)

   Linearly spaced bias init across dimensions.


.. py:function:: small_init(dim)

   Return initializer that creates a tensor with values according to the method described in:
   "Transformers without Tears: Improving the Normalization of Self-Attention", Nguyen, T. & Salazar, J. (2019).

   Adopted from https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/init_functions.py.


.. py:function:: wang_init(dim, num_blocks)

   Adopted from https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/init_functions.py.


.. py:function:: uniform_init(min_val, max_val)

   Uniform initializer.


