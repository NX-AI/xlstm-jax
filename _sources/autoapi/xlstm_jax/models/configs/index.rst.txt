xlstm_jax.models.configs
========================

.. py:module:: xlstm_jax.models.configs


Classes
-------

.. autoapisummary::

   xlstm_jax.models.configs.ParallelConfig
   xlstm_jax.models.configs.ModelConfig
   xlstm_jax.models.configs.SubModelConfig


Module Contents
---------------

.. py:class:: ParallelConfig

   Configuration for parallelism.


   .. py:attribute:: data_axis_size
      :type:  int
      :value: -1


      Size of the data axis. If -1, it will be inferred by the number of available devices.


   .. py:attribute:: fsdp_axis_size
      :type:  int
      :value: 1


      Size of the FSDP axis. If -1, it will be inferred by the number of available devices.


   .. py:attribute:: pipeline_axis_size
      :type:  int
      :value: 1


      Size of the pipeline axis. If -1, it will be inferred by the number of available devices.


   .. py:attribute:: model_axis_size
      :type:  int
      :value: 1


      Size of the model axis. If -1, it will be inferred by the number of available devices.


   .. py:attribute:: data_axis_name
      :type:  str
      :value: 'dp'


      Name of the data axis.


   .. py:attribute:: fsdp_axis_name
      :type:  str
      :value: 'fsdp'


      Name of the FSDP axis.


   .. py:attribute:: pipeline_axis_name
      :type:  str
      :value: 'pp'


      Name of the pipeline axis.


   .. py:attribute:: model_axis_name
      :type:  str
      :value: 'tp'


      Name of the model axis.


   .. py:attribute:: remat
      :type:  list[str]
      :value: []


      Module names on which we apply activation checkpointing / rematerialization.


   .. py:attribute:: fsdp_modules
      :type:  list[str]
      :value: []


      Module names on which we apply FSDP sharding.


   .. py:attribute:: fsdp_min_weight_size
      :type:  int
      :value: 262144


      Minimum size of a parameter to be sharded with FSDP.


   .. py:attribute:: fsdp_gather_dtype
      :type:  str | None
      :value: None


      The dtype to cast the parameters to before gathering with FSDP. If `None`, no casting is performed and parameters
      are gathered in original precision (e.g. `float32`).


   .. py:attribute:: fsdp_grad_scatter_dtype
      :type:  str | None
      :value: None


      The dtype to cast the gradients to before scattering. If `None`, the dtype of the parameters is used.


   .. py:attribute:: tp_async_dense
      :type:  bool
      :value: False


      Whether to use asynchronous tensor parallelism for dense layers. Default to `False`, as on local hardware,
      ppermute communication introduces large overhead.


.. py:class:: ModelConfig

   Bases: :py:obj:`xlstm_jax.configs.ConfigDict`


   Base class for model configurations.


   .. py:attribute:: model_class
      :type:  callable

      Model class.


   .. py:attribute:: parallel
      :type:  ParallelConfig

      Parallelism configuration.


   .. py:attribute:: model_config
      :type:  xlstm_jax.configs.ConfigDict | None
      :value: None


      Model configuration.


   .. py:method:: from_metadata(metadata_content)
      :staticmethod:


      Creates a model config from a metadata file content.

      :param metadata_content: Content of the metadata file, currently in JSON format.

      :returns: Tuple of the model_class and the model configuration parsed into a nested ModelConfig format.



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



.. py:class:: SubModelConfig

   Sub-model configuration.

   This class is currently a quick fix to allow for post-init style model configs, like the xlstm-clean we ported from
   the original xlstm codebase. Once the config system is more mature, we should remove this and all becomes a subclass
   of ModelConfig.


   .. py:method:: to_dict()

      Converts the config to a dictionary.

      Helpful for saving to disk or logging.



