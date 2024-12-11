xlstm_jax.trainer.callbacks.lr_monitor
======================================

.. py:module:: xlstm_jax.trainer.callbacks.lr_monitor


Attributes
----------

.. autoapisummary::

   xlstm_jax.trainer.callbacks.lr_monitor.LOGGER


Classes
-------

.. autoapisummary::

   xlstm_jax.trainer.callbacks.lr_monitor.LearningRateMonitorConfig
   xlstm_jax.trainer.callbacks.lr_monitor.LearningRateMonitor


Module Contents
---------------

.. py:data:: LOGGER

.. py:class:: LearningRateMonitorConfig

   Bases: :py:obj:`xlstm_jax.trainer.callbacks.callback.CallbackConfig`


   Configuration for the LearningRateMonitor callback.


   .. py:attribute:: every_n_epochs
      :type:  int
      :value: -1


      Log the learning rate every n epochs. Set to -1 to disable.


   .. py:attribute:: every_n_steps
      :type:  int
      :value: 50


      Log the learning rate every n steps. By default, logs every 50 steps.


   .. py:attribute:: main_process_only
      :type:  bool
      :value: True


      Log the learning rate only in the main process.


   .. py:attribute:: log_lr_key
      :type:  str
      :value: 'optimizer/lr'


      Key to use for logging the learning rate.


   .. py:method:: create(trainer, data_module = None)

      Creates the LearningRateMonitor callback.

      :param trainer: Trainer object.
      :param data_module: Data module object.
      :type data_module: optional

      :returns: LearningRateMonitor object.



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



.. py:class:: LearningRateMonitor(config, trainer, data_module = None)

   Bases: :py:obj:`xlstm_jax.trainer.callbacks.callback.Callback`


   Callback to monitor the learning rate.


   .. py:attribute:: lr_scheduler


   .. py:method:: on_filtered_training_step(step_metrics, epoch_idx, step_idx)

      Logs the learning rate after a step.

      :param step_metrics: Metrics of the current step. Unused in this callback.
      :param epoch_idx: Index of the current epoch. Unused in this callback.
      :param step_idx: Index of the current step.



   .. py:method:: on_filtered_training_epoch_end(train_metrics, epoch_idx)

      Logs the learning rate after an epoch.

      :param train_metrics: Metrics of the current epoch. Unused in this callback.
      :param epoch_idx: Index of the current epoch. Unused in this callback.



   .. py:method:: _log_lr(step_idx)

      Logs the learning rate.

      :param step_idx: Index of the current step.



   .. py:attribute:: config


   .. py:attribute:: trainer


   .. py:attribute:: data_module
      :value: None



   .. py:attribute:: _every_n_epochs


   .. py:attribute:: _every_n_steps


   .. py:attribute:: _main_process_only


   .. py:attribute:: _active_on_epochs


   .. py:attribute:: _active_on_steps


   .. py:method:: on_training_start()

      Called at the beginning of training.



   .. py:method:: on_training_end()

      Called at the end of training.



   .. py:method:: on_training_epoch_start(epoch_idx)

      Called at the beginning of each training epoch.

      :param epoch_idx: Index of the current epoch.



   .. py:method:: on_filtered_training_epoch_start(epoch_idx)

      Called at the beginning of each `every_n_epochs` training epoch. To be implemented by subclasses.

      :param epoch_idx: Index of the current epoch.



   .. py:method:: on_training_epoch_end(train_metrics, epoch_idx)

      Called at the end of each training epoch.

      :param train_metrics: Dictionary of training metrics of the current epoch.
      :param epoch_idx: Index of the current epoch.



   .. py:method:: on_training_step(step_metrics, epoch_idx, step_idx)

      Called at the end of each training step.

      :param step_metrics: Dictionary of training metrics of the current step.
      :param epoch_idx: Index of the current epoch.
      :param step_idx: Index of the current step.



   .. py:method:: on_validation_epoch_start(epoch_idx, step_idx)

      Called at the beginning of validation.

      :param epoch_idx: Index of the current training epoch.
      :param step_idx: Index of the current training step.



   .. py:method:: on_filtered_validation_epoch_start(epoch_idx, step_idx)

      Called at the beginning of `every_n_epochs` validation. To be implemented by subclasses.

      :param epoch_idx: Index of the current training epoch.
      :param step_idx: Index of the current training step.



   .. py:method:: on_validation_epoch_end(eval_metrics, epoch_idx, step_idx)

      Called at the end of each validation epoch.

      :param eval_metrics: Dictionary of evaluation metrics of the current epoch.
      :param epoch_idx: Index of the current training epoch.
      :param step_idx: Index of the current training step.



   .. py:method:: on_filtered_validation_epoch_end(eval_metrics, epoch_idx, step_idx)

      Called at the end of each `every_n_epochs` validation epoch. To be implemented by subclasses.

      :param eval_metrics: Dictionary of evaluation metrics of the current epoch.
      :param epoch_idx: Index of the current training epoch.
      :param step_idx: Index of the current training step.



   .. py:method:: on_test_epoch_start(epoch_idx)

      Called at the beginning of testing.

      To be implemented by subclasses.

      :param epoch_idx: Index of the current epoch.



   .. py:method:: on_test_epoch_end(test_metrics, epoch_idx)

      Called at the end of each test epoch. To be implemented by subclasses.

      :param test_metrics: Dictionary of test metrics of the current epoch.
      :param epoch_idx: Index of the current epoch.



   .. py:method:: set_dataset(data_module)

      Sets the data module.

      :param data_module: Data module object.



   .. py:method:: finalize(status = None)

      Called at the end of the whole training process.

      To be implemented by subclasses.

      :param status: Status of the training process.



