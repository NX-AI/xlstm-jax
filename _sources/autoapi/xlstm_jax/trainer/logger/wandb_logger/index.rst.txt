xlstm_jax.trainer.logger.wandb_logger
=====================================

.. py:module:: xlstm_jax.trainer.logger.wandb_logger


Attributes
----------

.. autoapisummary::

   xlstm_jax.trainer.logger.wandb_logger.LOGGER


Classes
-------

.. autoapisummary::

   xlstm_jax.trainer.logger.wandb_logger.WandBLoggerConfig
   xlstm_jax.trainer.logger.wandb_logger.WandBLogger


Module Contents
---------------

.. py:data:: LOGGER

.. py:class:: WandBLoggerConfig

   Bases: :py:obj:`xlstm_jax.trainer.logger.base_logger.LoggerToolsConfig`


   Configuration for the WandB logger tool.

   .. attribute:: wb_entity

      The WandB entity to log to.

   .. attribute:: wb_project

      The WandB project to log to.

   .. attribute:: wb_host

      The WandB host to log to.

   .. attribute:: wb_key

      The WandB API key to use. If None, the key will be read from the environment.

   .. attribute:: wb_name

      The name of the run.

   .. attribute:: wb_notes

      Notes to add to the run.

   .. attribute:: wb_settings

      Settings to pass to the WandB run.

   .. attribute:: wb_tags

      Tags to add to the run.

   .. attribute:: log_dir

      The directory to log to.


   .. py:attribute:: wb_entity
      :type:  str
      :value: 'xlstm'



   .. py:attribute:: wb_project
      :type:  str
      :value: 'xlstm_nxai'



   .. py:attribute:: wb_host
      :type:  str
      :value: 'https://api.wandb.ai'



   .. py:attribute:: wb_key
      :type:  str | None
      :value: None



   .. py:attribute:: wb_name
      :type:  str | None
      :value: None



   .. py:attribute:: wb_notes
      :type:  str | None
      :value: None



   .. py:attribute:: wb_settings
      :type:  dict[str, Any]


   .. py:attribute:: wb_tags
      :type:  list[str]
      :value: []



   .. py:attribute:: wb_resume_id
      :type:  str | None
      :value: None



   .. py:attribute:: log_dir
      :type:  str
      :value: 'wandb'



   .. py:method:: create(logger)

      Create a WandB logger tool.



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



.. py:class:: WandBLogger(config, logger)

   Bases: :py:obj:`xlstm_jax.trainer.logger.base_logger.LoggerTool`


   Base class for logger tools.


   .. py:attribute:: config


   .. py:attribute:: config_to_log
      :value: None



   .. py:attribute:: logger


   .. py:attribute:: log_path


   .. py:attribute:: wandb_run
      :value: None



   .. py:method:: log_config(config)

      Log the config to WandB.

      If the run is not set up, the config will be saved and logged
      when the run is set up.

      :param config: The config to log.



   .. py:method:: setup()

      Set up the WandB logger.

      If the run is already set up, this function skips the setup.



   .. py:method:: log_metrics(metrics, step, epoch, mode)

      Log a single metric dictionary in the WandB logger.

      :param metrics: The metrics to log.
      :param step: The current step.
      :param epoch: The current epoch. Currently unused.
      :param mode: The current mode. Will be used as a prefix for the metrics.



   .. py:method:: finalize(status)

      Closes the WandB logger.

      :param status: The status of the training run (e.g. success, failure).



