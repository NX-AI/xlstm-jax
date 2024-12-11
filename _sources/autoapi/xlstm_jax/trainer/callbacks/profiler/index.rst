xlstm_jax.trainer.callbacks.profiler
====================================

.. py:module:: xlstm_jax.trainer.callbacks.profiler


Attributes
----------

.. autoapisummary::

   xlstm_jax.trainer.callbacks.profiler.LOGGER


Classes
-------

.. autoapisummary::

   xlstm_jax.trainer.callbacks.profiler.JaxProfilerConfig
   xlstm_jax.trainer.callbacks.profiler.JaxProfiler


Module Contents
---------------

.. py:data:: LOGGER

.. py:class:: JaxProfilerConfig

   Bases: :py:obj:`xlstm_jax.trainer.callbacks.callback.CallbackConfig`


   Configuration for the JaxProfiler callback.

   .. attribute:: every_n_epochs

      Unused in this callback.

   .. attribute:: every_n_steps

      Unused in this callback.

   .. attribute:: main_process_only

      If True, the profiler is only active in the main process.
      Otherwise, one profile per process is created.

   .. attribute:: profile_every_n_minutes

      Profile every n minutes. If set below 0, the profiler
      is only done once at the beginning.

   .. attribute:: profile_first_step

      The first step to start profiling.

   .. attribute:: profile_n_steps

      Number of steps to profile.

   .. attribute:: profile_log_dir

      Directory to save the profiler logs. By default` set to
      "tensorboard", where also the TensorBoard logs are saved.


   .. py:attribute:: every_n_epochs
      :type:  int
      :value: -1



   .. py:attribute:: every_n_steps
      :type:  int
      :value: -1



   .. py:attribute:: main_process_only
      :type:  bool
      :value: True



   .. py:attribute:: profile_every_n_minutes
      :type:  int
      :value: 60



   .. py:attribute:: profile_first_step
      :type:  int
      :value: 10



   .. py:attribute:: profile_n_steps
      :type:  int
      :value: 5



   .. py:attribute:: profile_log_dir
      :type:  str
      :value: 'tensorboard'



   .. py:method:: create(trainer, data_module = None)

      Creates the JaxProfiler callback.

      :param trainer: Trainer object.
      :param data_module: Data module object.
      :type data_module: optional

      :returns: JaxProfiler object.



.. py:class:: JaxProfiler(config, trainer, data_module = None)

   Bases: :py:obj:`xlstm_jax.trainer.callbacks.callback.Callback`


   Callback to profile model training steps.


   .. py:attribute:: log_path
      :type:  pathlib.Path


   .. py:attribute:: profile_every_n_minutes


   .. py:attribute:: profile_first_step


   .. py:attribute:: profile_n_steps


   .. py:attribute:: profiler_active
      :value: False



   .. py:attribute:: profiler_last_time
      :value: None



   .. py:method:: on_training_start()

      Called at the beginning of training.

      Starts tracking the time to determine when to start the profiler.



   .. py:method:: on_training_step(step_metrics, epoch_idx, step_idx)

      Called at the end of each training step.

      Starts the profiler if the current step is the first step or if the
      time since the last profiling is greater than the specified interval.
      If the profiler is active, it stops the profiler after the specified
      number of steps.

      :param step_metrics: Dictionary of training metrics of the current step.
      :param epoch_idx: Index of the current epoch.
      :param step_idx: Index of the current step.



   .. py:method:: on_training_epoch_end(train_metrics, epoch_idx)

      Called at the end of each training epoch.

      Stop the profiler if it is still active to prevent tracing
      non-training step operations.

      :param train_metrics: Metrics of the current epoch.
      :param epoch_idx: Index of the current epoch.



   .. py:method:: on_validation_epoch_start(epoch_idx, step_idx)

      Called at the beginning of validation.

      If profiler is active, stop it to prevent tracing all validation
      steps.

      :param epoch_idx: Index of the current training epoch.
      :param step_idx: Index of the current training step.



   .. py:method:: start_trace(step_idx)

      Start the profiler trace.

      If the profiler is already active, a warning is logged.

      :param step_idx: Index of the current training step.



   .. py:method:: stop_trace()

      Stop the profiler trace.

      If the profiler is not active, nothing is done.



