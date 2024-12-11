xlstm_jax.define_hydra_schemas
==============================

.. py:module:: xlstm_jax.define_hydra_schemas

.. autoapi-nested-parse::

   Register all config dataclasses in the project to Hydra's ConfigStore and define hydra config schemas.



Classes
-------

.. autoapisummary::

   xlstm_jax.define_hydra_schemas.CombinedModelConfig
   xlstm_jax.define_hydra_schemas.BaseLoggerConfig
   xlstm_jax.define_hydra_schemas.DataEvalConfig
   xlstm_jax.define_hydra_schemas.DataTrainConfig
   xlstm_jax.define_hydra_schemas.Config


Functions
---------

.. autoapisummary::

   xlstm_jax.define_hydra_schemas.register_configs


Module Contents
---------------

.. py:class:: CombinedModelConfig

   This class is a flat config that combines several sub-configs.


   .. py:attribute:: name
      :type:  str


   .. py:attribute:: vocab_size
      :type:  int


   .. py:attribute:: embedding_dim
      :type:  int


   .. py:attribute:: num_blocks
      :type:  int


   .. py:attribute:: context_length
      :type:  int


   .. py:attribute:: tie_weights
      :type:  bool


   .. py:attribute:: add_embedding_dropout
      :type:  bool


   .. py:attribute:: add_post_blocks_norm
      :type:  bool


   .. py:attribute:: scan_blocks
      :type:  bool


   .. py:attribute:: norm_eps
      :type:  float


   .. py:attribute:: norm_type
      :type:  str


   .. py:attribute:: init_distribution_embed
      :type:  str


   .. py:attribute:: logits_soft_cap
      :type:  float


   .. py:attribute:: lm_head_dtype
      :type:  str


   .. py:attribute:: dtype
      :type:  str


   .. py:attribute:: add_post_norm
      :type:  bool


   .. py:attribute:: layer_type
      :type:  str


   .. py:attribute:: num_heads
      :type:  int


   .. py:attribute:: output_init_fn
      :type:  str


   .. py:attribute:: init_distribution
      :type:  str


   .. py:attribute:: qk_dim_factor
      :type:  float


   .. py:attribute:: v_dim_factor
      :type:  float


   .. py:attribute:: gate_dtype
      :type:  str


   .. py:attribute:: backend
      :type:  str


   .. py:attribute:: backend_name
      :type:  str


   .. py:attribute:: igate_bias_init_range
      :type:  float


   .. py:attribute:: add_qk_norm
      :type:  bool


   .. py:attribute:: cell_norm_type
      :type:  str


   .. py:attribute:: cell_norm_type_v1
      :type:  str


   .. py:attribute:: cell_norm_eps
      :type:  float


   .. py:attribute:: gate_soft_cap
      :type:  float


   .. py:attribute:: reset_at_document_boundaries
      :type:  bool


   .. py:attribute:: proj_factor
      :type:  float


   .. py:attribute:: act_fn
      :type:  str


   .. py:attribute:: ff_type
      :type:  str


   .. py:attribute:: ff_dtype
      :type:  str


   .. py:attribute:: head_dim
      :type:  int


   .. py:attribute:: attention_backend
      :type:  str


   .. py:attribute:: theta
      :type:  float


.. py:class:: BaseLoggerConfig

   Bases: :py:obj:`xlstm_jax.trainer.logger.base_logger.LoggerConfig`


   .. py:attribute:: loggers_to_use
      :type:  list[str]


   .. py:attribute:: file_logger_log_dir
      :type:  str


   .. py:attribute:: file_logger_config_format
      :type:  str


   .. py:attribute:: tb_log_dir
      :type:  str


   .. py:attribute:: tb_flush_secs
      :type:  int


   .. py:attribute:: wb_project
      :type:  str


   .. py:attribute:: wb_entity
      :type:  str


   .. py:attribute:: wb_name
      :type:  str


   .. py:attribute:: wb_tags
      :type:  list[str]


.. py:class:: DataEvalConfig

   Supports maximum of 5 data evaluation configurations.


   .. py:attribute:: ds1
      :type:  xlstm_jax.dataset.configs.DataConfig | None
      :value: None



   .. py:attribute:: ds2
      :type:  xlstm_jax.dataset.configs.DataConfig | None
      :value: None



   .. py:attribute:: ds3
      :type:  xlstm_jax.dataset.configs.DataConfig | None
      :value: None



   .. py:attribute:: ds4
      :type:  xlstm_jax.dataset.configs.DataConfig | None
      :value: None



   .. py:attribute:: ds5
      :type:  xlstm_jax.dataset.configs.DataConfig | None
      :value: None



.. py:class:: DataTrainConfig

   Supports maximum of 10 data training configurations.


   .. py:attribute:: ds1
      :type:  xlstm_jax.dataset.configs.DataConfig | None
      :value: None



   .. py:attribute:: weight1
      :type:  float
      :value: 1.0



   .. py:attribute:: ds2
      :type:  xlstm_jax.dataset.configs.DataConfig | None
      :value: None



   .. py:attribute:: weight2
      :type:  float
      :value: 1.0



   .. py:attribute:: ds3
      :type:  xlstm_jax.dataset.configs.DataConfig | None
      :value: None



   .. py:attribute:: weight3
      :type:  float
      :value: 1.0



   .. py:attribute:: ds4
      :type:  xlstm_jax.dataset.configs.DataConfig | None
      :value: None



   .. py:attribute:: weight4
      :type:  float
      :value: 1.0



   .. py:attribute:: ds5
      :type:  xlstm_jax.dataset.configs.DataConfig | None
      :value: None



   .. py:attribute:: weight5
      :type:  float
      :value: 1.0



   .. py:attribute:: ds6
      :type:  xlstm_jax.dataset.configs.DataConfig | None
      :value: None



   .. py:attribute:: weight6
      :type:  float
      :value: 1.0



   .. py:attribute:: ds7
      :type:  xlstm_jax.dataset.configs.DataConfig | None
      :value: None



   .. py:attribute:: weight7
      :type:  float
      :value: 1.0



   .. py:attribute:: ds8
      :type:  xlstm_jax.dataset.configs.DataConfig | None
      :value: None



   .. py:attribute:: weight8
      :type:  float
      :value: 1.0



   .. py:attribute:: ds9
      :type:  xlstm_jax.dataset.configs.DataConfig | None
      :value: None



   .. py:attribute:: weight9
      :type:  float
      :value: 1.0



   .. py:attribute:: ds10
      :type:  xlstm_jax.dataset.configs.DataConfig | None
      :value: None



   .. py:attribute:: weight10
      :type:  float
      :value: 1.0



.. py:class:: Config

   The base config class.


   .. py:attribute:: parallel
      :type:  xlstm_jax.models.configs.ParallelConfig


   .. py:attribute:: model
      :type:  CombinedModelConfig


   .. py:attribute:: scheduler
      :type:  xlstm_jax.trainer.optimizer.scheduler.SchedulerConfig


   .. py:attribute:: optimizer
      :type:  xlstm_jax.trainer.optimizer.optimizer.OptimizerConfig


   .. py:attribute:: checkpointing
      :type:  xlstm_jax.trainer.callbacks.checkpointing.ModelCheckpointConfig


   .. py:attribute:: lr_monitor
      :type:  xlstm_jax.trainer.callbacks.lr_monitor.LearningRateMonitorConfig


   .. py:attribute:: profiling
      :type:  xlstm_jax.trainer.callbacks.profiler.JaxProfilerConfig


   .. py:attribute:: logger
      :type:  xlstm_jax.trainer.logger.base_logger.LoggerConfig


   .. py:attribute:: trainer
      :type:  xlstm_jax.trainer.base.trainer.TrainerConfig


   .. py:attribute:: device
      :type:  str


   .. py:attribute:: device_count
      :type:  int


   .. py:attribute:: n_gpus
      :type:  int


   .. py:attribute:: batch_size_per_device
      :type:  int


   .. py:attribute:: global_batch_size
      :type:  int


   .. py:attribute:: lr
      :type:  float


   .. py:attribute:: context_length
      :type:  int


   .. py:attribute:: num_epochs
      :type:  int


   .. py:attribute:: num_train_steps
      :type:  int


   .. py:attribute:: log_path
      :type:  pathlib.Path


   .. py:attribute:: base_dir
      :type:  pathlib.Path


   .. py:attribute:: task_name
      :type:  str


   .. py:attribute:: logging_name
      :type:  str


   .. py:attribute:: data_train
      :type:  DataTrainConfig | None
      :value: None



   .. py:attribute:: data_eval
      :type:  DataEvalConfig | None
      :value: None



.. py:function:: register_configs()

