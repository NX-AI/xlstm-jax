xlstm_jax.trainer.logger.cmd_logging
====================================

.. py:module:: xlstm_jax.trainer.logger.cmd_logging

.. autoapi-nested-parse::

   Logging setup for the command line interface.

   Ported from https://github.com/NX-AI/xlstm-dev/blob/main/xlstm/ml_utils/log_utils/log_cmd.py



Attributes
----------

.. autoapisummary::

   xlstm_jax.trainer.logger.cmd_logging.LOG_FORMAT
   xlstm_jax.trainer.logger.cmd_logging.LOGGER


Functions
---------

.. autoapisummary::

   xlstm_jax.trainer.logger.cmd_logging.setup_exception_logging
   xlstm_jax.trainer.logger.cmd_logging.get_loglevel
   xlstm_jax.trainer.logger.cmd_logging.setup_logging
   xlstm_jax.trainer.logger.cmd_logging.setup_logging_multiprocess


Module Contents
---------------

.. py:data:: LOG_FORMAT
   :value: '[%(asctime)s][%(name)s:%(lineno)d][%(levelname)s]{rank} - %(message)s'


.. py:data:: LOGGER

.. py:function:: setup_exception_logging()

   Make sure that uncaught exceptions are logged with the logger.


.. py:function:: get_loglevel()

   Get loglevel from `LOGLEVEL` environment variable.

   :returns: loglevel
   :rtype: str


.. py:function:: setup_logging(logfile = 'output.log')

   Initialize logging to `logfile` and `stdout`.

   :param logfile: Name of the log file. Defaults to "output.log".


.. py:function:: setup_logging_multiprocess(logfile = 'output.log')

   Initialize logging to `logfile` and `stdout` for JAX distributed training.

   :param logfile: Name of the log file. Defaults to "output.log".


