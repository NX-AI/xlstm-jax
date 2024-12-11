xlstm_jax.utils.error_logging_utils
===================================

.. py:module:: xlstm_jax.utils.error_logging_utils


Attributes
----------

.. autoapisummary::

   xlstm_jax.utils.error_logging_utils.R
   xlstm_jax.utils.error_logging_utils.P


Functions
---------

.. autoapisummary::

   xlstm_jax.utils.error_logging_utils.with_error_handling


Module Contents
---------------

.. py:data:: R

.. py:data:: P

.. py:function:: with_error_handling(flush_output = True, logger = None)

   A decorator that provides consistent error handling with optional output flushing.

   This decorator is used to circumvent a bug in Hydra. See here:
   https://github.com/facebookresearch/hydra/issues/2664#issuecomment-1857695600

   :param flush_output: Whether to flush stdout and stderr in the finally block.
                        Defaults to True.
   :param logger: Optional logger instance to use for error logging.
                  If None errors will be printed to stderr.

   :returns: A decorator function that wraps the target function.


