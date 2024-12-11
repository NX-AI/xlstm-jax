xlstm_jax.train_init_fns
========================

.. py:module:: xlstm_jax.train_init_fns


Attributes
----------

.. autoapisummary::

   xlstm_jax.train_init_fns.LOGGER


Functions
---------

.. autoapisummary::

   xlstm_jax.train_init_fns.init_parallel
   xlstm_jax.train_init_fns.init_data_iterator
   xlstm_jax.train_init_fns.init_single_data_iterator
   xlstm_jax.train_init_fns.init_mixed_data_iterator
   xlstm_jax.train_init_fns.get_tokenizer_vocab_size
   xlstm_jax.train_init_fns.init_model_config
   xlstm_jax.train_init_fns.init_logger_config
   xlstm_jax.train_init_fns.init_scheduler_config
   xlstm_jax.train_init_fns.init_optimizer_config
   xlstm_jax.train_init_fns.init_model_checkpointing
   xlstm_jax.train_init_fns.init_lr_monitor_config
   xlstm_jax.train_init_fns.init_profiler_config
   xlstm_jax.train_init_fns.init_trainer
   xlstm_jax.train_init_fns.log_info


Module Contents
---------------

.. py:data:: LOGGER

.. py:function:: init_parallel(cfg)

   Initialize configuration for parallelism.

   :param cfg  Config assembled by Hydra.:

   :returns: Initialized parallel configuration.


.. py:function:: init_data_iterator(cfg, mesh)

   Initialize data iterators.

   :param cfg: Config assembled by Hydra.
   :param mesh: The jax device mesh.

   :returns: Training and evaluation data iterators.


.. py:function:: init_single_data_iterator(cfg, mesh, create_split = None)

   Initialize a single data iterator.

   :param cfg: Data configuration.
   :param mesh: The jax device mesh.
   :param create_split: Whether to create a train or eval config from the config class, using the
                        `create_train_eval_configs` method. If None, the config is used as is.

   :returns: Data iterator.


.. py:function:: init_mixed_data_iterator(cfg, mesh)

   Initialize a data iterator with mixed data sources.

   :param cfg: Data configuration.
   :param mesh: The jax device mesh.

   :returns: Data iterator.


.. py:function:: get_tokenizer_vocab_size(cfg, next_multiple_of = 1)

   Get the vocabulary size from the tokenizer.

   :param cfg: Config assembled by Hydra.
   :param next_multiple_of: The vocabulary size will be increased to the next multiple of this number.

   :returns: The vocabulary size, increased to the next multiple of `next_multiple_of`.


.. py:function:: init_model_config(cfg, parallel)

   Instantiate the model configuration.

   :param cfg: Config assembled by Hydra.
   :param parallel: Parallel configuration.

   :returns: Initialized model configuration.


.. py:function:: init_logger_config(cfg)

   Instantiate logger configuration.

   :param cfg: Config assembled by Hydra.

   :returns: Instance of LoggerConfig.


.. py:function:: init_scheduler_config(cfg, data_iterator)

   Instantiate scheduler configuration.

   :param data_iterator:
   :param cfg: Config assembled by Hydra.

   :returns: Instance of SchedulerConfig following the provided config.


.. py:function:: init_optimizer_config(cfg)

   Instantiate optimizer configuration.

   :param cfg: Full Hydra config.

   :returns: Instance of OptimizerConfig.


.. py:function:: init_model_checkpointing(cfg)

   Instantiate model checkpointing configuration.

   :param cfg: Full Hydra config.

   :returns: Instance of ModelCheckpointConfig.


.. py:function:: init_lr_monitor_config(cfg)

   Instantiate learning rate monitor configuration.

   :param cfg: Full Hydra config.

   :returns: Instance of LearningRateMonitorConfig.


.. py:function:: init_profiler_config(cfg)

   Instantiate profiler configuration.

   :param cfg: Full Hydra config.

   :returns: Instance of JaxProfilerConfig.


.. py:function:: init_trainer(cfg, data_iterator, model_config, mesh)

   Initializes the LLMTrainer with all sub-configs.

   :param cfg: Full Hydra config.
   :param data_iterator: A data iterator.
   :param model_config: A model config.
   :param mesh: A device mesh.

   :returns: Instance of LLM trainer.


.. py:function:: log_info(msg)

