xlstm_jax.trainer.base.trainer
==============================

.. py:module:: xlstm_jax.trainer.base.trainer


Attributes
----------

.. autoapisummary::

   xlstm_jax.trainer.base.trainer.LOGGER


Classes
-------

.. autoapisummary::

   xlstm_jax.trainer.base.trainer.TrainerConfig
   xlstm_jax.trainer.base.trainer.TrainerModule


Module Contents
---------------

.. py:data:: LOGGER

.. py:class:: TrainerConfig

   Bases: :py:obj:`xlstm_jax.configs.ConfigDict`


   Configuration for the Trainer module.


   .. py:attribute:: seed
      :type:  int
      :value: 0


      Random seed for reproducibility. To be used in the model init and training step.


   .. py:attribute:: debug
      :type:  bool
      :value: False


      Whether to run in debug mode. This disables jitting of the training and evaluation functions, which will slow
      down the training significantly but makes debugging easier.


   .. py:attribute:: donate_train_state
      :type:  bool
      :value: True


      Whether to donate the train state in the training step. This can reduce memory usage as the parameters and
      optimizer states are in-place updated in the training step. However, this prevents using the previous train state
      after calling the training step (not used in Trainer, but keep in mind for custom training loops and callbacks).


   .. py:attribute:: enable_progress_bar
      :type:  bool
      :value: True


      Whether to enable the progress bar. For multiprocess training, only the main process will show the progress bar.


   .. py:attribute:: gradient_accumulate_steps
      :type:  int
      :value: 1


      Number of steps to accumulate gradients before updating the parameters.


   .. py:attribute:: gradient_accumulate_scan
      :type:  bool
      :value: False


      Whether to use scan for gradient accumulation. This can be more memory efficient and significantly faster to
      compile for large models, but can be slighlty slower due to memory slicing.


   .. py:attribute:: check_val_every_n_epoch
      :type:  int
      :value: 1


      Check validation every N training epochs. If -1, no validation is performed after an epoch. Note that this is
      not mutually exclusive with check_val_every_n_steps, and both can be used.


   .. py:attribute:: check_val_every_n_steps
      :type:  int
      :value: -1


      Check validation every N training steps. If -1, no validation is performed on a per-step basis. Note that this
      is not mutually exclusive with check_val_every_n_epoch, and both can be used.


   .. py:attribute:: check_for_nan
      :type:  bool
      :value: True


      Whether to check for NaN values in the loss during training. If NaNs are found, training will be stopped.


   .. py:attribute:: log_grad_norm
      :type:  bool
      :value: True


      Whether to log the gradient norm.


   .. py:attribute:: log_grad_norm_per_param
      :type:  bool
      :value: False


      Whether to log the gradient norm per parameter. If the model has many parameters, this can lead to a large log
      file.


   .. py:attribute:: log_param_norm
      :type:  bool
      :value: True


      Whether to log the parameter norm.


   .. py:attribute:: log_param_norm_per_param
      :type:  bool
      :value: False


      Whether to log the parameter norm per parameter. If the model has many parameters, this can lead to a large log
      file.


   .. py:attribute:: log_intermediates
      :type:  bool
      :value: False


      Whether to log intermediate values during training. This is useful for debugging, but can lead to a large log
      file and a bit of overhead during training, if intermediates are complex to compute. Intermediates can be recorded
      by using the ``self.sow("intermediates", "KEY", VALUE)`` method in the model. The intermediate values are
      automatically registered and logged. Note that the values should be scalars.


   .. py:attribute:: default_train_log_modes
      :type:  list[str]
      :value: ['mean']


      Default logging modes for training metrics. Can be `mean`, `mean_nopostfix`, `single`, `max`, or `std`. See
      metrics for more information. Each selected mode will be logged with the corresponding postfix. During validation,
      we only log the `mean` of the metrics.


   .. py:attribute:: intermediates_log_modes
      :type:  list[str]
      :value: ['mean']


      Logging modes for intermediate values. See `default_train_log_modes` for more information.


   .. py:attribute:: logger
      :type:  xlstm_jax.trainer.logger.LoggerConfig | None

      Configuration for the logger.


   .. py:attribute:: callbacks
      :type:  list[xlstm_jax.trainer.callbacks.CallbackConfig]
      :value: []


      List of callbacks to apply.


   .. py:attribute:: seed_eval
      :type:  int
      :value: 0


      Random seed for evaluation, if the model uses randomness during evaluation. This is useful to ensure
      reproducibility of evaluation metrics.


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



.. py:class:: TrainerModule(trainer_config, model_config, optimizer_config, batch, mesh = None)

   A basic Trainer module summarizing most common training functionalities like logging, model initialization, training
   loop, etc.

   :param trainer_config: A dictionary containing the trainer configuration.
   :param model_config: A dictionary containing the model configuration.
   :param optimizer_config: A dictionary containing the optimizer configuration.
   :param batch: An input to the model with which the shapes are inferred. Can be a :class:`jax.ShapeDtypeStruct` instead
                 of actual full arrays for efficiency. Must NOT be a jax.ShapeDtypeStruct if jax.debug.* statements are used
                 inside the model code.
   :param mesh: A mesh object to use for parallel training. If `None`, a new mesh will be created.


   .. py:attribute:: trainer_config


   .. py:attribute:: model_config


   .. py:attribute:: optimizer_config


   .. py:attribute:: exmp_batch


   .. py:attribute:: _train_metric_shapes
      :value: None



   .. py:attribute:: _eval_metric_shapes
      :value: None



   .. py:attribute:: mesh
      :value: None



   .. py:attribute:: batch_partition_specs


   .. py:attribute:: state
      :value: None



   .. py:attribute:: callbacks
      :value: None



   .. py:attribute:: first_step
      :value: True



   .. py:attribute:: global_step
      :value: 0



   .. py:attribute:: dataset
      :value: None



   .. py:method:: batch_to_input(batch)
      :staticmethod:


      Convert a batch to the input format expected by the model.

      Needs to be implemented by the subclass if `batch.inputs` is not sufficient.

      :param batch: A batch of data.

      :returns: The input to the model.



   .. py:method:: init_mesh(model_config, mesh = None)

      Initialize the mesh for parallel training if no mesh is supplied.

      :param model_config: A dictionary containing the model configuration, including the parallelization parameters.
      :param mesh: A mesh object to use for parallel training. If `None`, a new mesh is created.



   .. py:method:: build_model(model_config)

      Create the model class from the model_config.

      :param model_config: A dictionary containing the model configuration.



   .. py:method:: init_logger(logger_config)

      Initialize a logger and creates a logging directory.

      :param logger_config:



   .. py:method:: get_metric_postprocess_fn()

      Get function to post-process metrics with on host.

      Will be passed to logger. Default implementation returns the identity function.
      Can be overwritten by subclasses.

      :returns: The postprocess metric function.
      :rtype: Callable[[HostMetrics], HostMetrics]



   .. py:method:: init_callbacks(callback_configs)

      Initialize the callbacks defined in the trainer config.



   .. py:method:: init_optimizer(optimizer_config)

      Initialize the optimizer.

      :param optimizer_config: A dictionary containing the optimizer configuration.



   .. py:method:: init_model(exmp_input)

      Create an initial training state with newly generated network parameters.

      This function is parallelized over the mesh to initialize the per-device parameters. It also initializes the
      optimizer parameters. As a result, it sets the training state of the trainer with the initialized parameters.

      :param exmp_input: An input to the model with which the shapes are inferred.



   .. py:method:: init_train_metrics(batch = None)

      Initialize the training metrics with zeros.

      We infer the training metric shape from the train_step function. This is done to prevent a double-compilation of
      the train_step function, where the first step has to be done with metrics None, and the next one with the
      metrics shape.

      :param batch: An input to the model with which the shapes are inferred. If None, the :attr:`exmp_batch` is used.

      :returns: A dictionary of metrics with the same shape as the train metrics.



   .. py:method:: init_eval_metrics(batch = None)

      Initialize the evaluation metrics with zeros.

      See init_train_metrics for more details.

      :param batch: An input to the model with which the shapes are inferred. If None, the :attr:`exmp_batch` is used.

      :returns: A dictionary of metrics with the same shape as the eval metrics.



   .. py:method:: set_dataset(dataset)

      Set the dataset for the trainer and the callbacks.

      :param dataset: The dataset to set.



   .. py:method:: get_model_rng(rng)
      :staticmethod:


      Return a dictionary of PRNGKey for init and tabulate.

      By default, adds a key for the parameters and one for dropout. If more keys are needed, this function should be
      overwritten.

      :param rng: The current PRNGKey.

      :returns: Dict of PRNG Keys.



   .. py:method:: run_model_init(exmp_input, init_rng)

      The model initialization call.

      :param exmp_input: An input to the model with which the shapes are inferred.
      :param init_rng: A jax.random.PRNGKey.

      :returns: The initialized variable dictionary.



   .. py:method:: tabulate_params()

      Return a string summary of the parameters represented as table.

      :returns: A string representation of the parameters.



   .. py:method:: get_num_params()

      Return the number of parameters in the model.

      :returns: The number of parameters.



   .. py:method:: create_jitted_functions()

      Create jitted versions of the training and evaluation functions.

      If self.trainer_config.debug is True, not jitting is applied.



   .. py:method:: loss_function(params, apply_fn, batch, rng, train = True)

      The loss function that is used for training.

      This function needs to be overwritten by a subclass.

      :param params: The model parameters.
      :param apply_fn: The apply function of the state.
      :param batch: The current batch.
      :param rng: The random number generator.
      :param train: Whether the model is in training mode.

      :returns: The loss and a tuple of metrics and mutable variables.



   .. py:method:: create_training_step_function()

      Create and return a function for the training step.

      The function takes as input the training state and a batch from the train loader. The function is expected to
      return a dictionary of logging metrics, and a new train state.



   .. py:method:: create_evaluation_step_function()

      Create and return a function for the evaluation step.

      The function takes as input the training state and a batch from the val/test loader. The function is expected to
      return a dictionary of logging metrics, and a new train state.



   .. py:method:: train_model(train_loader, val_loader, test_loader = None, num_epochs = None, num_train_steps = None, steps_per_epoch = None)

      Start a training loop for the given number of epochs.

      Inside the training loop, we use an epoch index and a global step index. Both indices are starting to count
      at 1 (i.e. first epoch is "epoch 1", not "epoch 0").

      :param train_loader: Data loader of the training set.
      :param val_loader: Data loader of the validation set. If a dictionary is given, the model is evaluated on all
                         datasets in the dictionary, and the key of the dataset is used as a prefix for the metrics
                         (`DATAKEY_METRICKEY`). Note that these naming differences also need to be considered for the callbacks,
                         such as the Model Checkpoint with tracking the best metric if used.
      :param test_loader: If given, best model will be evaluated on the test set. Similar to val_loader, if a dictionary
                          is given, the model is evaluated on all datasets in the dictionary, and the key of the dataset is used
                          as a prefix for the metrics.
      :param num_epochs: Number of epochs for which to train the model. If None, will use num_train_steps.
      :param num_train_steps: Number of training steps for which to train the model. If None, will use num_epochs.
      :param steps_per_epoch: Number of steps per epoch. If None, will use the length of the train_loader.

      :returns: A dictionary of the train, validation and evt. test metrics for the
                best model on the validation set.



   .. py:method:: _eval_model_in_train_loop(val_loader, epoch_idx)

      Evaluate the model on the validation set during the training loop.

      :param val_loader: Data loader of the validation set. If a dictionary is given, the model is evaluated on all
                         datasets in the dictionary, and the key of the dataset is used as a prefix for the metrics.
      :param epoch_idx: Current epoch index.

      :returns: A dictionary of the evaluation metrics.



   .. py:method:: test_model(test_loader, apply_callbacks = False, epoch_idx = 0)

      Tests the model on the given test set.

      :param test_loader: Data loader of the test set. If a dictionary is given, the model is evaluated on all datasets
                          in the dictionary, and the key of the dataset is used as a prefix for the metrics.
      :param apply_callbacks: If True, the callbacks will be applied.
      :param epoch_idx: The epoch index to use for the callbacks and logging.

      :returns: A dictionary of the evaluation metrics.



   .. py:method:: test_eval_function(val_loader)

      Test the evaluation function on a single batch.

      This is useful to check if the functions have the correct signature and return the correct values. This prevents
      annoying errors that occur at the first evaluation step.

      This function does not test the training function anymore. This is because the training function is already
      executed in the first epoch, and we change its jit signature to donate the train state and metrics. Thus,
      executing a training step requires updating the train state, which we would not want to do here. The compilation
      time is logged during the very first training step.

      :param val_loader: Data loader of the validation set.



   .. py:method:: eval_model(data_loader, mode, epoch_idx)

      Evaluate the model on a dataset.

      If multiple datasets are given, the evaluation is performed on all datasets and the metrics are prefixed with
      the dataset key (i.e. `DATAKEY_METRICKEY`). The evaluation metrics are logged and returned as host metrics.

      :param data_loader: Data loader of the dataset to evaluate on. If a dictionary is given, the model is evaluated on
                          all datasets in the dictionary, and the key of the dataset is used as a prefix for the metrics.
      :param mode: The mode to use for logging, commonly "val" or "test".
      :param epoch_idx: Current epoch index.

      :returns: A dictionary of the evaluation metrics on the host, averaged over data points in the dataset.



   .. py:method:: _run_model_eval(data_loader, mode = '', epoch_idx = -1)

      Evaluate the model on a single dataset.

      In contrast to eval_model, this function does not log the metrics and returns the on-device metrics. It also
      does not support evaluation on multiple datasets. For this, use eval_model.

      :param data_loader: Data loader of the dataset to evaluate on.
      :param mode: Mode to show in the progress bar and logging. Default is empty string.
      :param epoch_idx: Current epoch index. Only used for logging, default is -1.

      :returns: The on-device metrics after the full evaluation epoch.



   .. py:method:: tracker(iterator, **kwargs)

      Wrap an iterator in a progress bar tracker (tqdm) if the progress bar is enabled.

      :param iterator: Iterator to wrap in tqdm.
      :param kwargs: Additional arguments to tqdm.

      :returns: Wrapped iterator if progress bar is enabled, otherwise same iterator as input.



   .. py:method:: log_training_info(num_epochs, num_train_steps, steps_per_epoch, train_loader, val_loader, test_loader)

      Log the general training information.

      :param num_epochs: Number of epochs for which to train the model.
      :param num_train_steps: Number of training steps for which to train the model.
      :param steps_per_epoch: Number of steps per epoch.
      :param train_loader: Data loader of the training set.
      :param val_loader: Data loader of the validation set.
      :param test_loader: Data loader of the test set.



   .. py:method:: on_training_start()

      Method called before training is started.

      Can be used for additional initialization operations etc.



   .. py:method:: on_training_end()

      Method called after training has finished.

      Can be used for additional logging or similar.



   .. py:method:: on_training_epoch_start(epoch_idx)

      Method called at the start of each training epoch. Can be used for additional logging or similar.

      :param epoch_idx: Index of the training epoch that has started.



   .. py:method:: on_training_epoch_end(train_metrics, epoch_idx)

      Method called at the end of each training epoch. Can be used for additional logging or similar.

      :param train_metrics: A dictionary with training metrics. Newly added metrics will be logged as well.
      :param epoch_idx: Index of the training epoch that has finished.



   .. py:method:: on_validation_epoch_start(epoch_idx, step_idx)

      Method called at the start of each validation epoch. Can be used for additional logging or similar.

      :param epoch_idx: Index of the training epoch at which validation was started.
      :param step_idx: Index of the training step at which validation was started.



   .. py:method:: on_validation_epoch_end(eval_metrics, epoch_idx, step_idx)

      Method called at the end of each validation epoch. Can be used for additional logging and evaluation.

      :param eval_metrics: A dictionary with validation metrics. Newly added metrics will be logged as well.
      :param epoch_idx: Index of the training epoch at which validation was performed.
      :param step_idx: Index of the training step at which validation was performed.



   .. py:method:: on_test_epoch_start(epoch_idx)

      Method called at the start of each test epoch. Can be used for additional logging or similar.

      :param epoch_idx: Index of the training epoch at which testing was started.



   .. py:method:: on_test_epoch_end(test_metrics, epoch_idx)

      Method called at the end of each test epoch. Can be used for additional logging and evaluation.

      :param test_metrics: A dictionary with test metrics. Newly added metrics will be logged as well.
      :param epoch_idx: Index of the training epoch at which testing was performed.



   .. py:method:: load_model(step_idx = -1, raise_if_not_found = True)

      Load model parameters and batch statistics from the logging directory.

      :param step_idx: Step index to load the model from. If -1, the latest model is loaded.
      :param raise_if_not_found: If True, raises an error if no model is found. If False, logs a warning instead.



   .. py:method:: load_data_loaders(step_idx = -1, train_loader = None, val_loader = None, test_loader = None)

      Load states of the data loaders from the logging directory.

      :param step_idx: Step index to load the data loaders from. If -1, uses the global train step.
      :param train_loader: If given, the training data loader is set to this value.
      :param val_loader: If given, the validation data loader is set to this value.
      :param test_loader: If given, the test data loader is set to this value.



   .. py:method:: restore_model(state_dict)

      Restore the state of the trainer from a state dictionary.
      Only if the current trainer state has a tx and opt_state attribute, update these.
      Re-use the class of the current trainer state to allow such a pruned one.

      :param state_dict: State dictionary to restore from. Must contain the key "params" with the model parameters.
                         Optional keys that overwrite the trainer state are "step", "opt_state", "mutable_variables", "rng".



   .. py:method:: restore_data_loaders(state_dict, train_loader = None, val_loader = None, test_loader = None)
      :staticmethod:


      Restore the state of the data loaders from a state dictionary.

      :param state_dict: State dictionary to restore from. Should contain the keys "train", "val" and "test" with
                         the data loader states.
      :param train_loader: If given, the training data loader is set to this value.
      :param val_loader: If given, the validation data loader is set to this value.
      :param test_loader: If given, the test data loader is set to this value.



   .. py:method:: load_pretrained_model(checkpoint_path, step_idx = -1, load_best = False, load_optimizer = True, train_loader = None, val_loader = None, test_loader = None, delete_params_before_loading = True)

      Load a pretrained model from a checkpoint directory.

      :param checkpoint_path: Path to the checkpoint directory.
      :param step_idx: Step index to load the model from. If -1, the latest model is loaded.
      :param load_best: If True, loads the best model instead of the latest model.
      :param load_optimizer: If True, load the optimizer state with the pretrained model.
      :param train_loader: If given, the training data loader is set to the state of the pretrained model.
      :param val_loader: If given, the validation data loader is set to the state of the pretrained model.
      :param test_loader: If given, the test data loader is set to the state of the pretrained model.
      :param delete_params_before_loading: If True, delete the current model parameters before loading the pretrained
                                           model. Saves memory on the device, but original model parameters cannot be used anymore.

      :returns: The step index of the loaded model.



   .. py:method:: load_from_checkpoint(checkpoint, exmp_input = None, batch_size = -1)
      :classmethod:


      Create a Trainer object with same hyperparameters and loaded model from a checkpoint directory.

      :param checkpoint: Folder in which the checkpoint and hyperparameter file is stored.
      :param exmp_input: An input to the model for shape inference.
      :param batch_size: Batch size to use for shape inference. If -1, the full exmp_input is used.

      :returns: A Trainer object with model loaded from the checkpoint folder.



