xlstm_jax.models.xlstm_pytorch.components.ln
============================================

.. py:module:: xlstm_jax.models.xlstm_pytorch.components.ln


Classes
-------

.. autoapisummary::

   xlstm_jax.models.xlstm_pytorch.components.ln.LayerNorm
   xlstm_jax.models.xlstm_pytorch.components.ln.MultiHeadLayerNorm


Module Contents
---------------

.. py:class:: LayerNorm(ndim = -1, weight = True, bias = False, eps = 1e-05, residual_weight = True)

   Bases: :py:obj:`torch.nn.Module`


   LayerNorm but with an optional bias.

   PyTorch doesn't support simply bias=False.


   .. py:attribute:: weight


   .. py:attribute:: bias
      :value: None



   .. py:attribute:: eps
      :value: 1e-05



   .. py:attribute:: residual_weight
      :value: True



   .. py:attribute:: ndim
      :value: -1



   .. py:property:: weight_proxy
      :type: torch.Tensor



   .. py:method:: forward(input)


   .. py:method:: reset_parameters()


.. py:class:: MultiHeadLayerNorm(ndim = -1, weight = True, bias = False, eps = 1e-05, residual_weight = True)

   Bases: :py:obj:`LayerNorm`


   LayerNorm but with an optional bias.

   PyTorch doesn't support simply bias=False.


   .. py:method:: forward(input)


   .. py:attribute:: weight


   .. py:attribute:: bias
      :value: None



   .. py:attribute:: eps
      :value: 1e-05



   .. py:attribute:: residual_weight
      :value: True



   .. py:attribute:: ndim
      :value: -1



   .. py:property:: weight_proxy
      :type: torch.Tensor



   .. py:method:: reset_parameters()


