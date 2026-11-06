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



