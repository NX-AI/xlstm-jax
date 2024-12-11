xlstm_jax.trainer.logger.base_logger
====================================

.. py:module:: xlstm_jax.trainer.logger.base_logger


Attributes
----------

.. autoapisummary::

   xlstm_jax.trainer.logger.base_logger.LOGGER


Classes
-------

.. autoapisummary::

   xlstm_jax.trainer.logger.base_logger.LoggerConfig
   xlstm_jax.trainer.logger.base_logger.LoggerToolsConfig
   xlstm_jax.trainer.logger.base_logger.Logger
   xlstm_jax.trainer.logger.base_logger.LoggerTool


Module Contents
---------------

.. py:data:: LOGGER

.. py:class:: LoggerConfig

   Bases: :py:obj:`xlstm_jax.configs.ConfigDict`


   Configuration for the logger.

   .. attribute:: log_every_n_steps

      The frequency at which logs should be written.

   .. attribute:: log_path

      The path where the logs should be written. If None, we will not write logs to disk.

   .. attribute:: log_tools

      A list of LoggerToolsConfig objects that should be used to log the metrics. These tools will be
      created in the Logger class.

   .. attribute:: cmd_logging_name

      The name of the output file for command line logging without suffix. The suffix `.log` will
      be added automatically.


   .. py:attribute:: log_every_n_steps
      :type:  int
      :value: 1



   .. py:attribute:: log_path
      :type:  pathlib.Path | None
      :value: None



   .. py:attribute:: log_tools
      :type:  list[LoggerToolsConfig]
      :value: []



   .. py:attribute:: cmd_logging_name
      :type:  str
      :value: 'output'



   .. py:property:: log_dir
      :type: str


      Returns the log directory as a string.


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



.. py:class:: LoggerToolsConfig

   Bases: :py:obj:`xlstm_jax.configs.ConfigDict`


   Base config class for logger tools.

   These are tools that can be used to log metrics, images, etc. They are created inside the Logger class.


   .. py:method:: create(logger)
      :abstractmethod:


      Creates the logger tool.



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



.. py:class:: Logger(config, metric_postprocess_fn = None)

   Logger class to log metrics, images, etc.


   .. py:attribute:: config


   .. py:attribute:: log_path


   .. py:attribute:: metric_postprocess_fn
      :value: None



   .. py:attribute:: epoch
      :value: 0



   .. py:attribute:: step
      :value: 0



   .. py:attribute:: found_nans
      :value: False



   .. py:attribute:: last_step


   .. py:attribute:: last_step_time
      :value: None



   .. py:attribute:: epoch_start_time_stack
      :value: []



   .. py:attribute:: mode_stack
      :value: []



   .. py:property:: mode
      :type: Literal['default', 'train', 'val', 'test']


      Current logging mode. Can be "default", "train", "val", or "test".

      :returns: The current logging mode.
      :rtype: str


   .. py:method:: log_config(config)

      Logs the configuration.

      :param config: The configuration to log. Can also be a dictionary of multiple configurations.



   .. py:method:: on_training_start()

      Set up the logger for training.



   .. py:method:: start_epoch(epoch, step, mode = 'train')

      Starts a new epoch.

      To be called before starting a new training, eval or test epoch. Can also be called if one is still
      in another epoch. For instance, if the training epoch is interrupted by a validation epoch, the logger
      switches to the validation mode until a `end_epoch` is called. Then, the logger switches back to the
      training mode.

      :param epoch: The index of the epoch.
      :param step: The index of the global training step.
      :param mode: The logging mode. Should be in {"train", "val", "test"}. Defaults to "train".



   .. py:method:: log_step(metrics, step)

      Log metrics for a single step.

      :param metrics: The metrics to log. Should follow the structure of the metrics in the metrics.py file.
      :param step: The current step.

      :returns: If the metrics are logged in this step, the metrics will be updated to reset all metrics.
                If the metrics are not logged in this step, the metrics will be returned unchanged.



   .. py:method:: _check_for_nans(host_metrics, step = None)

      Check if any of the metrics contain NaNs.

      If `NaN` are found, a warning is logged and the `found_nans` attribute is set to `True`.

      :param host_metrics: The metrics to check.
      :param step: The step at which the metrics were logged. Used for logging if provided.



   .. py:method:: log_host_metrics(host_metrics, step, mode = None)

      Logs a dictionary of metrics on the host.

      Can be used by callbacks to log additional metrics.

      :param host_metrics: The metrics to log.
      :param step: The current step.
      :param mode: The mode / prefix with which to log the metrics. If None, the current mode is used.



   .. py:method:: end_epoch(metrics, step)

      Ends the current epoch and logs the epoch metrics.

      If any other epoch is still running, the logger will switch back to that epoch.

      :param metrics: The metrics that should be logged in this epoch.
      :param step: The current step.

      :returns: The originally passed metric dict and potentially any other metrics that should be passed
                to callbacks later on. Note that the metrics will not be reset.



   .. py:method:: finalize(status)

      Closes the logger.

      :param status: The status of the training run (e.g. success, failure).



.. py:class:: LoggerTool

   Base class for logger tools.


   .. py:method:: log_config(config)

      Log the configuration to the tool.

      :param config: The configuration to log.



   .. py:method:: log_metrics(metrics, step, epoch, mode)
      :abstractmethod:


      Log the metrics to the tool.

      :param metrics: The metrics to log.
      :param step: The current step.
      :param epoch: The current epoch.
      :param mode: The current mode (train, val, test).



   .. py:method:: finalize(status)

      Finalize and close the tool.



