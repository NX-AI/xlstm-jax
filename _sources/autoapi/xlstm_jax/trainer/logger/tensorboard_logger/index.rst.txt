xlstm_jax.trainer.logger.tensorboard_logger
===========================================

.. py:module:: xlstm_jax.trainer.logger.tensorboard_logger


Attributes
----------

.. autoapisummary::

   xlstm_jax.trainer.logger.tensorboard_logger.LOGGER


Classes
-------

.. autoapisummary::

   xlstm_jax.trainer.logger.tensorboard_logger.TensorBoardLoggerConfig
   xlstm_jax.trainer.logger.tensorboard_logger.TensorBoardLogger


Module Contents
---------------

.. py:data:: LOGGER

.. py:class:: TensorBoardLoggerConfig

   Bases: :py:obj:`xlstm_jax.trainer.logger.base_logger.LoggerToolsConfig`


   Configuration for the TensorBoard logger tool.

   .. attribute:: tb_flush_secs

      The frequency at which to flush the tensorboard logs.

   .. attribute:: tb_max_queue

      The maximum number of items to queue before flushing.

   .. attribute:: tb_new_style

      Whether to use the new style of logging. See PyTorch
      SummaryWriter documentation for more information.

   .. attribute:: log_dir

      The directory to use for the logs. Is added to the
      log_path in the logger


   .. py:attribute:: tb_flush_secs
      :type:  int
      :value: 120



   .. py:attribute:: tb_max_queue
      :type:  int
      :value: 10



   .. py:attribute:: tb_new_style
      :type:  bool
      :value: False



   .. py:attribute:: log_dir
      :type:  str
      :value: 'tensorboard'



   .. py:method:: create(logger)

      Creates the TensorBoard logger tool.



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



.. py:class:: TensorBoardLogger(config, logger)

   Bases: :py:obj:`xlstm_jax.trainer.logger.base_logger.LoggerTool`


   Base class for logger tools.


   .. py:attribute:: config


   .. py:attribute:: config_to_log
      :value: None



   .. py:attribute:: logger


   .. py:attribute:: writer
      :type:  torch.utils.tensorboard.SummaryWriter
      :value: None



   .. py:method:: log_config(config)

      Log the config to TensorBoard.

      If the writer is not set up, the config will be saved and logged
      when the writer is set up.

      :param config: The config to log.



   .. py:method:: _log_config()

      Logs stored config to TensorBoard if writer is set up.



   .. py:method:: setup()

      Set up the TensorBoard logger.

      If the writer is already set up, this function skips the setup.



   .. py:method:: log_metrics(metrics, step, epoch, mode)

      Log a single metric dictionary in the TensorBoard logger.

      :param metrics: The metrics to log.
      :param step: The current step.
      :param epoch: The current epoch. Currently unused.
      :param mode: The mode of logging. Commonly "train", "val", or "test". Is used as prefix for the metric keys.



   .. py:method:: finalize(status)

      Close the TensorBoard logger.

      :param status: The status of the training run (e.g. success, failure).



