xlstm_jax.trainer.logger.file_logger
====================================

.. py:module:: xlstm_jax.trainer.logger.file_logger


Attributes
----------

.. autoapisummary::

   xlstm_jax.trainer.logger.file_logger.LOGGER


Classes
-------

.. autoapisummary::

   xlstm_jax.trainer.logger.file_logger.FileLoggerConfig
   xlstm_jax.trainer.logger.file_logger.FileLogger


Module Contents
---------------

.. py:data:: LOGGER

.. py:class:: FileLoggerConfig

   Bases: :py:obj:`xlstm_jax.trainer.logger.base_logger.LoggerToolsConfig`


   Configuration for the file logger tool.

   .. attribute:: log_step_key

      The key to use for the step in the logs.

   .. attribute:: log_epoch_key

      The key to use for the epoch in the logs.

   .. attribute:: config_format

      The format to use when logging the config.

   .. attribute:: log_dir

      The directory to use for the logs. Is added to the
      log_path in the logger.


   .. py:attribute:: log_step_key
      :type:  str
      :value: 'log_step'



   .. py:attribute:: log_epoch_key
      :type:  str
      :value: 'log_epoch'



   .. py:attribute:: config_format
      :type:  str
      :value: 'json'



   .. py:attribute:: log_dir
      :type:  str
      :value: 'file_logs'



   .. py:method:: create(logger)

      Creates the file logger tool.



   .. py:method:: get(key, default=None)


   .. py:method:: to_dict()

      Converts the config to a dictionary.

      Helpful for saving to disk or logging.



   .. py:method:: from_dict(config_class, data, strict_classname_parsing = False, ignore_extensive_attributes = True, none_to_zero_for_ints = False)
      :staticmethod:


      Utility for parsing dictionaries back into a nested dataclass structure, including arbitrary classes and types.

      Currently, this is tailored towards the current logging system with the "hardly" invertible to_dict.

      :param config_class: Typically a dataclass, but can be any other type as well
                           If it is another type, the parser tries to create an object via
                           config_class(**data) if data is a dictionary or config_class(data) else.
      :param data: Typically a dictionary that contains attributes of the dataclass.
                   Can be any other kind of data.
      :param strict_classname_parsing: Parse class names strictly.
      :param ignore_extensive_attributes: Ignore attributes that are not defined in the dataclass.
      :param none_to_zero_for_ints: Convert None to 0 for integer types.

      :returns: An object of type `config_class` that contains the data as attributes.



.. py:class:: FileLogger(config, logger)

   Bases: :py:obj:`xlstm_jax.trainer.logger.base_logger.LoggerTool`


   Base class for logger tools.


   .. py:attribute:: config


   .. py:attribute:: config_to_log
      :value: None



   .. py:attribute:: logger


   .. py:attribute:: log_path


   .. py:attribute:: logs


   .. py:method:: log_config(config)

      Log the config to disk.

      :param config: The config to log.



   .. py:method:: setup()

      Set up the file logger.



   .. py:method:: log_metrics(metrics, step, epoch, mode)

      Log a single metric dictionary in the file logger.

      The metrics are logged in a list and saved to disk at the end.

      :param metrics: The metrics to log.
      :param step: The current step.
      :param epoch: The current epoch.
      :param mode: The mode of logging. Commonly "train", "val", or "test".



   .. py:method:: finalize(status)

      Finalize the file logger.

      Writes out the logs to disk.

      :param status: The status of the training run (e.g. success, failure).



