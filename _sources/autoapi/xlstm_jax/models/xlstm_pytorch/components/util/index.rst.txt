xlstm_jax.models.xlstm_pytorch.components.util
==============================================

.. py:module:: xlstm_jax.models.xlstm_pytorch.components.util


Classes
-------

.. autoapisummary::

   xlstm_jax.models.xlstm_pytorch.components.util.ParameterProxy


Functions
---------

.. autoapisummary::

   xlstm_jax.models.xlstm_pytorch.components.util.round_to_multiple
   xlstm_jax.models.xlstm_pytorch.components.util.conditional_decorator


Module Contents
---------------

.. py:function:: round_to_multiple(n, m=8)

.. py:function:: conditional_decorator(condition, decorator)

   A higher-order decorator that applies 'decorator' only if 'condition' is True.


.. py:class:: ParameterProxy(module, parameter_name, internal_to_external, external_to_internal)

   This class helps keeping parameters in a specialized internal structure to be optimal for computation speed, while
   having a proxied version to be called externally that is backend-agnostic.

   It takes a module and a parameter name of a parameter in that module it represents. Via __setitem__ and __getitem__
   the "external"


   .. py:attribute:: module


   .. py:attribute:: parameter_name


   .. py:attribute:: internal_to_external


   .. py:attribute:: external_to_internal


   .. py:method:: clone()


   .. py:property:: shape


   .. py:property:: ndim


   .. py:property:: grad


