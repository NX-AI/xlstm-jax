xlstm_jax.trainer.callbacks.checkpointing
=========================================

.. py:module:: xlstm_jax.trainer.callbacks.checkpointing


Attributes
----------

.. autoapisummary::

   xlstm_jax.trainer.callbacks.checkpointing.LOGGER


Classes
-------

.. autoapisummary::

   xlstm_jax.trainer.callbacks.checkpointing.ModelCheckpointConfig
   xlstm_jax.trainer.callbacks.checkpointing.ModelCheckpoint


Functions
---------

.. autoapisummary::

   xlstm_jax.trainer.callbacks.checkpointing.load_pretrained_model


Module Contents
---------------

.. py:data:: LOGGER

.. py:class:: ModelCheckpointConfig

   Bases: :py:obj:`xlstm_jax.trainer.callbacks.callback.CallbackConfig`


   Configuration for the ModelCheckpoint callback.

   By default, the checkpoint saves the model parameters, training step, random number generator state, and metadata
   to the logging directory. The metadata includes the trainer, model, and optimizer configurations.

   .. attribute:: max_to_keep

      Number of checkpoints to keep. If None, keeps all checkpoints. Otherwise, keeps the most recent
      `max_to_keep` checkpoints. If `monitor` is set, keeps the best `max_to_keep` checkpoints instead of the
      most recent.

   .. attribute:: monitor

      Metric to monitor for saving the model. Should be a key of the evaluation metrics. If None, checkpoints
      are sorted by recency.

   .. attribute:: mode

      One of {"min", "max"}. If "min", saves the model with the smallest value of the monitored metric. If
      "max", saves the model with the largest value of the monitored metric.

   .. attribute:: save_optimizer_state

      Whether to save the optimizer state.

   .. attribute:: save_dataloader_state

      Whether to save the dataloader state.

   .. attribute:: enable_async_checkpointing

      Whether to enable asynchronous checkpointing. See orbax documentation for more
      information.

   .. attribute:: log_path

      Path to save the checkpoints as subfolder to. If None, saves to the logging directory of the trainer.


   .. py:attribute:: max_to_keep
      :type:  int | None
      :value: 1



   .. py:attribute:: monitor
      :type:  str | None
      :value: None



   .. py:attribute:: mode
      :type:  str
      :value: 'min'



   .. py:attribute:: save_optimizer_state
      :type:  bool
      :value: True



   .. py:attribute:: save_dataloader_state
      :type:  bool
      :value: True



   .. py:attribute:: enable_async_checkpointing
      :type:  bool
      :value: True



   .. py:attribute:: log_path
      :type:  pathlib.Path | None
      :value: None



   .. py:method:: create(trainer, data_module = None)

      Creates the ModelCheckpoint callback.

      :param trainer: Trainer object.
      :param data_module: Data module object.
      :type data_module: optional

      :returns: ModelCheckpoint object.



.. py:class:: ModelCheckpoint(config, trainer, data_module = None)

   Bases: :py:obj:`xlstm_jax.trainer.callbacks.callback.Callback`


   Callback to save model parameters and mutable variables to the logging directory.

   Sets up an orbax checkpoint manager to save model parameters, training step, random number generator state, and
   metadata to the logging directory.

   :param config: The configuration for the ModelCheckpoint callback.
   :param trainer: The trainer object. If the trainer has no optimizer attribute, the optimizer part will not be loaded.
   :param data_module: The data module object.


   .. py:attribute:: checkpoint_path


   .. py:attribute:: dataloader_path


   .. py:attribute:: metadata


   .. py:attribute:: manager


   .. py:method:: on_filtered_validation_epoch_end(eval_metrics, epoch_idx, step_idx)

      Saves the model at the end of the validation epoch.

      :param eval_metrics: Dictionary of evaluation metrics. If a monitored metric is set, the model is saved based on
                           the monitored metrics in this dictionary. If the monitored metric is not found, an error is raised.
                           The metrics are saved along with the model.
      :param epoch_idx: Index of the current epoch.
      :param step_idx: Index of the current step.



   .. py:method:: save_model(eval_metrics, step_idx)

      Saves model state dict to the logging directory.

      :param eval_metrics: Dictionary of evaluation metrics. If a monitored metric is set, the model is saved based on
                           the monitored metrics in this dictionary. If the monitored metric is not found, an error is raised.
                           The metrics are saved along with the model.
      :param step_idx: Index of the current step.



   .. py:method:: save_dataloader(step_idx)

      Saves the dataloader state to the logging directory.

      :param step_idx: Index of the current step.



   .. py:method:: load_model(step_idx = -1, load_best = False, delete_params_before_loading = False)

      Loads model parameters and variables from the logging directory.

      :param step_idx: Index of the step to load. If -1, loads the latest step by default.
      :param load_best: If True and step_idx is -1, loads the best checkpoint
                        based on the monitored metric instead of the latest checkpoint.
      :param delete_params_before_loading: If True, deletes the current parameters in the
                                           trainer state before loading the new parameters.

      :returns: Dictionary of loaded model parameters and additional variables.



   .. py:method:: load_dataloader(step_idx = -1, load_best = False)

      Loads the dataloader state from the logging directory.

      :param step_idx: Index of the step to load. If -1, loads the latest step by default.
      :param load_best: If True and step_idx is -1, loads the best checkpoint
                        based on the monitored metric instead of the latest checkpoint.

      :returns: Dictionary of loaded dataloader states.



   .. py:method:: resolve_step_idx(step_idx, load_best)

      Resolves the step index to load.

      :param step_idx: Index of the step to load. If -1, loads the latest step by default.
      :param load_best: If True and step_idx is -1, loads the best checkpoint
                        based on the monitored metric instead of the latest checkpoint.

      :returns: The resolved step index.



   .. py:method:: finalize(status = None)

      Closes the checkpoint manager.

      :param status: The status of the training run (e.g. success, failure).



.. py:function:: load_pretrained_model(checkpoint_path, trainer, step_idx = -1, load_optimizer = True, load_best = False, delete_params_before_loading = False)

   Loads a pretrained model from a checkpoint.

   :param checkpoint_path: Path to the checkpoint directory.
   :param trainer: Trainer object.
   :param step_idx: Index of the step to load. If -1, loads the latest step by default.
   :param load_optimizer: If True the optimizer state is loaded from the checkpoint.
   :param load_best: If True and step_idx is -1, loads the best checkpoint
                     based on the monitored metric instead of the latest checkpoint.
   :param delete_params_before_loading: If True, deletes the current parameters in the
                                        trainer state before loading the new parameters.

   :returns: Dictionary of loaded model parameters and additional variables, as well as the dataloader state
             and the resolved step index that was loaded.


