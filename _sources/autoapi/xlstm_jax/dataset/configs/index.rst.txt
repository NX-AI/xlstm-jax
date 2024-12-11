xlstm_jax.dataset.configs
=========================

.. py:module:: xlstm_jax.dataset.configs


Attributes
----------

.. autoapisummary::

   xlstm_jax.dataset.configs.T


Classes
-------

.. autoapisummary::

   xlstm_jax.dataset.configs.DataConfig
   xlstm_jax.dataset.configs.HFHubDataConfig
   xlstm_jax.dataset.configs.SyntheticDataConfig
   xlstm_jax.dataset.configs.GrainArrayRecordsDataConfig


Module Contents
---------------

.. py:data:: T

.. py:class:: DataConfig

   Base data configuration.


   .. py:attribute:: shuffle_data
      :type:  bool
      :value: False


      Whether to shuffle the data. Usually True for training and False for validation.


   .. py:attribute:: name
      :type:  str | None
      :value: None


      Name of the dataset. Helpful for logging.


   .. py:attribute:: data_config_type
      :type:  str | None
      :value: None


      Type of data configuration. Used for initialization via Hydra. Can be 'synthetic',
      'huggingface_hub', 'huggingface_local', or 'grain_arrayrecord'.


   .. py:attribute:: global_batch_size
      :type:  int

      Global batch size for training.


   .. py:attribute:: max_target_length
      :type:  int | None
      :value: None


      Maximum length of the target sequence.


   .. py:attribute:: data_shuffle_seed
      :type:  int
      :value: 42


      Seed for data shuffling.


   .. py:method:: create_train_eval_configs(train_kwargs = None, eval_kwargs = None, **kwargs)
      :classmethod:


      Create training and evaluation configurations.

      :param train_kwargs: Training-exclusive keyword arguments.
      :param eval_kwargs: Evaluation-exclusive keyword arguments.
      :param \*\*kwargs: Shared keyword arguments.

      :returns: Training and evaluation configurations.
      :rtype: Tuple[DataConfig, DataConfig]



.. py:class:: HFHubDataConfig

   Bases: :py:obj:`DataConfig`


   HuggingFace dataset configuration for datasets on HuggingFace.


   .. py:attribute:: hf_path
      :type:  pathlib.Path | str

      Path to the dataset on HuggingFace.


   .. py:attribute:: hf_cache_dir
      :type:  pathlib.Path | str | None
      :value: None


      Directory to cache the dataset.


   .. py:attribute:: hf_access_token
      :type:  str | None
      :value: None


      Access token for HuggingFace


   .. py:attribute:: hf_data_dir
      :type:  pathlib.Path | str | None
      :value: None


      Directory for additional data files.


   .. py:attribute:: hf_data_files
      :type:  str | None
      :value: None


      Specific (training or evaluation) files to use


   .. py:attribute:: split
      :type:  str | None
      :value: 'train'


      Split to use (for training or evaluation).


   .. py:attribute:: hf_num_data_processes
      :type:  int | None
      :value: None


      Number of processes to use for downloading the dataset.


   .. py:attribute:: data_column
      :type:  str
      :value: 'text'


      Column name for (training or evaluation) data.


   .. py:attribute:: max_steps_per_epoch
      :type:  int | None
      :value: None


      Maximum number of steps per epoch (for training or evaluation).


   .. py:attribute:: tokenizer_path
      :type:  str
      :value: 'gpt2'


      Path to the tokenizer.


   .. py:attribute:: add_bos
      :type:  bool
      :value: False


      Whether to add `beginning of sequence` token.


   .. py:attribute:: add_eos
      :type:  bool
      :value: False


      Whether to add `end of sequence` token.


   .. py:attribute:: add_eod
      :type:  bool
      :value: True


      Whether to add an end of document token.


   .. py:attribute:: grain_packing
      :type:  bool
      :value: False


      Whether to perform packing via grain FirstFitPackIterDataset.


   .. py:attribute:: grain_packing_bin_count
      :type:  int | None
      :value: None


      Number of bins for grain packing. If None, use the local batch size. Higher values may improve packing
      efficiency but may also increase memory usage and pre-processing times.


   .. py:attribute:: worker_count
      :type:  int
      :value: 1


      Number of workers for data processing.


   .. py:attribute:: worker_buffer_size
      :type:  int
      :value: 1


      Buffer size for workers.


   .. py:attribute:: drop_remainder
      :type:  bool
      :value: False


      Whether to drop the remainder of the dataset when it does not divide evenly by the global batch size.


   .. py:attribute:: batch_rampup_factors
      :type:  dict[int, float] | None
      :value: None


      Ramp up the batch size if provided. The dictionary maps the step count to the scaling factor. See
      the `boundaries_and_scales` doc in `:func:grain_batch_rampup.create_batch_rampup_schedule` for more details.


   .. py:attribute:: shuffle_data
      :type:  bool
      :value: False


      Whether to shuffle the data. Usually True for training and False for validation.


   .. py:attribute:: name
      :type:  str | None
      :value: None


      Name of the dataset. Helpful for logging.


   .. py:attribute:: data_config_type
      :type:  str | None
      :value: None


      Type of data configuration. Used for initialization via Hydra. Can be 'synthetic',
      'huggingface_hub', 'huggingface_local', or 'grain_arrayrecord'.


   .. py:attribute:: global_batch_size
      :type:  int

      Global batch size for training.


   .. py:attribute:: max_target_length
      :type:  int | None
      :value: None


      Maximum length of the target sequence.


   .. py:attribute:: data_shuffle_seed
      :type:  int
      :value: 42


      Seed for data shuffling.


   .. py:method:: create_train_eval_configs(train_kwargs = None, eval_kwargs = None, **kwargs)
      :classmethod:


      Create training and evaluation configurations.

      :param train_kwargs: Training-exclusive keyword arguments.
      :param eval_kwargs: Evaluation-exclusive keyword arguments.
      :param \*\*kwargs: Shared keyword arguments.

      :returns: Training and evaluation configurations.
      :rtype: Tuple[DataConfig, DataConfig]



.. py:class:: SyntheticDataConfig

   Bases: :py:obj:`DataConfig`


   Synthetic dataset configuration.


   .. py:attribute:: num_batches
      :type:  int
      :value: 100


      Number of samples to generate for synthetic (training or evaluation) data.


   .. py:attribute:: shuffle_data
      :type:  bool
      :value: False


      Whether to shuffle the data. Usually True for training and False for validation.


   .. py:attribute:: name
      :type:  str | None
      :value: None


      Name of the dataset. Helpful for logging.


   .. py:attribute:: data_config_type
      :type:  str | None
      :value: None


      Type of data configuration. Used for initialization via Hydra. Can be 'synthetic',
      'huggingface_hub', 'huggingface_local', or 'grain_arrayrecord'.


   .. py:attribute:: global_batch_size
      :type:  int

      Global batch size for training.


   .. py:attribute:: max_target_length
      :type:  int | None
      :value: None


      Maximum length of the target sequence.


   .. py:attribute:: data_shuffle_seed
      :type:  int
      :value: 42


      Seed for data shuffling.


   .. py:method:: create_train_eval_configs(train_kwargs = None, eval_kwargs = None, **kwargs)
      :classmethod:


      Create training and evaluation configurations.

      :param train_kwargs: Training-exclusive keyword arguments.
      :param eval_kwargs: Evaluation-exclusive keyword arguments.
      :param \*\*kwargs: Shared keyword arguments.

      :returns: Training and evaluation configurations.
      :rtype: Tuple[DataConfig, DataConfig]



.. py:class:: GrainArrayRecordsDataConfig

   Bases: :py:obj:`DataConfig`


   Grain dataset configuration for ArrayRecords datasets.


   .. py:attribute:: data_path
      :type:  pathlib.Path

      Path to the dataset directory.


   .. py:attribute:: data_column
      :type:  str
      :value: 'text'


      Column name for (training or evaluation) data.


   .. py:attribute:: split
      :type:  str
      :value: 'train'


      Dataset split to use, e.g. 'train' or 'validation'. Should be a subdirectory of data_dir.


   .. py:attribute:: drop_remainder
      :type:  bool
      :value: False


      Whether to drop the remainder of the dataset when it does not divide evenly by the global batch size.


   .. py:attribute:: max_steps_per_epoch
      :type:  int | None
      :value: None


      Maximum number of steps per epoch.


   .. py:attribute:: tokenize_data
      :type:  bool
      :value: True


      Whether to tokenize the data data. If False, the data is assumed to be already tokenized.


   .. py:attribute:: tokenizer_path
      :type:  str
      :value: 'gpt2'


      Path to the tokenizer.


   .. py:attribute:: add_bos
      :type:  bool
      :value: False


      Whether to add `beginning of sequence` token.


   .. py:attribute:: add_eos
      :type:  bool
      :value: False


      Whether to add `end of sequence` token.


   .. py:attribute:: add_eod
      :type:  bool
      :value: True


      Whether to add an end of document token.


   .. py:attribute:: grain_packing
      :type:  bool
      :value: False


      Whether to perform packing via grain FirstFitPackIterDataset.


   .. py:attribute:: grain_packing_bin_count
      :type:  int | None
      :value: None


      Number of bins for grain packing. If None, use the local batch size. Higher values may improve packing
      efficiency but may also increase memory usage and pre-processing times.


   .. py:attribute:: worker_count
      :type:  int
      :value: 1


      Number of workers for data processing.


   .. py:attribute:: worker_buffer_size
      :type:  int
      :value: 1


      Buffer size for workers.


   .. py:attribute:: hf_cache_dir
      :type:  pathlib.Path | None
      :value: None


      Directory to cache the dataset. Used to get the HF tokenizer.


   .. py:attribute:: hf_access_token
      :type:  str | None
      :value: None


      Access token for HuggingFace. Used to get the HF tokenizer.


   .. py:attribute:: batch_rampup_factors
      :type:  dict[int, float] | None
      :value: None


      Ramp up the batch size if provided. The dictionary maps the step count to the scaling factor. See
      the `boundaries_and_scales` doc in `:func:grain_batch_rampup.create_batch_rampup_schedule` for more details.


   .. py:attribute:: shuffle_data
      :type:  bool
      :value: False


      Whether to shuffle the data. Usually True for training and False for validation.


   .. py:attribute:: name
      :type:  str | None
      :value: None


      Name of the dataset. Helpful for logging.


   .. py:attribute:: data_config_type
      :type:  str | None
      :value: None


      Type of data configuration. Used for initialization via Hydra. Can be 'synthetic',
      'huggingface_hub', 'huggingface_local', or 'grain_arrayrecord'.


   .. py:attribute:: global_batch_size
      :type:  int

      Global batch size for training.


   .. py:attribute:: max_target_length
      :type:  int | None
      :value: None


      Maximum length of the target sequence.


   .. py:attribute:: data_shuffle_seed
      :type:  int
      :value: 42


      Seed for data shuffling.


   .. py:method:: create_train_eval_configs(train_kwargs = None, eval_kwargs = None, **kwargs)
      :classmethod:


      Create training and evaluation configurations.

      :param train_kwargs: Training-exclusive keyword arguments.
      :param eval_kwargs: Evaluation-exclusive keyword arguments.
      :param \*\*kwargs: Shared keyword arguments.

      :returns: Training and evaluation configurations.
      :rtype: Tuple[DataConfig, DataConfig]



