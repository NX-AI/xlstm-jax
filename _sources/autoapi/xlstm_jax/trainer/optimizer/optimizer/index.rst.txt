xlstm_jax.trainer.optimizer.optimizer
=====================================

.. py:module:: xlstm_jax.trainer.optimizer.optimizer


Attributes
----------

.. autoapisummary::

   xlstm_jax.trainer.optimizer.optimizer.LOGGER


Classes
-------

.. autoapisummary::

   xlstm_jax.trainer.optimizer.optimizer.OptimizerConfig


Functions
---------

.. autoapisummary::

   xlstm_jax.trainer.optimizer.optimizer.build_optimizer
   xlstm_jax.trainer.optimizer.optimizer.build_optimizer_function
   xlstm_jax.trainer.optimizer.optimizer.build_gradient_transformations
   xlstm_jax.trainer.optimizer.optimizer.clip_by_global_norm_sharded


Module Contents
---------------

.. py:data:: LOGGER

.. py:class:: OptimizerConfig

   Bases: :py:obj:`xlstm_jax.configs.ConfigDict`


   Configuration for optimizer.

   .. attribute:: name

      Name of the optimizer. The supported optimizers are "adam", "adamw", "sgd", "nadam", "adamax",
      "radam", "nadamw", "adamax", and "lamb". If "none" is provided, the optimizer will be set to an SGD
      optimizer with learning rate 0.0, effectively skipping the optimizer.

      :type: str

   .. attribute:: scheduler

      Configuration for learning rate scheduler. Defaults to a constant learning rate.
      If the optimizer is "none", the scheduler is not used.

      :type: SchedulerConfig

   .. attribute:: beta1

      Exponential decay rate for the first moment estimates. This includes momentum in SGD, which
      can be set to None for no momentum.

      :type: float

   .. attribute:: beta2

      Exponential decay rate for the second moment estimates.

      :type: float

   .. attribute:: beta3

      For AdEMAMix, exponential decay rate for the slow EMA.

      :type: float

   .. attribute:: alpha

      For AdEMAMix, mixing coefficient for the linear combination of the fast and slow EMAs.
      Commonly in the range 5-10, with Mamba models performing best at 8. TODO: Update with xLSTM results.

      :type: float

   .. attribute:: eps

      Epsilon value for numerical stability in Adam-like optimizers.

      :type: float

   .. attribute:: weight_decay

      Weight decay coefficient.

      :type: float

   .. attribute:: weight_decay_exclude

      List of regex patterns of `re.Pattern` to exclude from weight decay.
      Parameter names are flattened and joined with ".". Mutually exclusive with weight_decay_include.

      :type: list[str] | None

   .. attribute:: weight_decay_include

      List of regex patterns of `re.Pattern` to include in weight decay.
      Parameter names are flattened and joined with ".". Mutually exclusive with weight_decay_exclude.
      If neither exclude nor include is set, all parameters are included.

      :type: list[str] | None

   .. attribute:: grad_clip_norm

      Global norm to clip gradients.

      :type: float | None

   .. attribute:: use_sharded_clip_norm

      Whether to calculate the global norm for clipping over all shards of the
      parameter (True), or only calculate the grad norm for local shards (False). If True, may introduce a small
      communication overhead, but reproduces the behavior of the original implementation for sharded parameters.

      :type: bool

   .. attribute:: grad_clip_value

      Value to clip gradients element-wise.

      :type: float | None

   .. attribute:: nesterov

      Whether to use Nesterov momentum in SGD.

      :type: bool


   .. py:attribute:: name
      :type:  str


   .. py:attribute:: scheduler
      :type:  xlstm_jax.trainer.optimizer.scheduler.SchedulerConfig


   .. py:attribute:: beta1
      :type:  float | None
      :value: 0.9



   .. py:attribute:: beta2
      :type:  float
      :value: 0.999



   .. py:attribute:: beta3
      :type:  float
      :value: 0.9999



   .. py:attribute:: alpha
      :type:  float
      :value: 8.0



   .. py:attribute:: eps
      :type:  float
      :value: 1e-08



   .. py:attribute:: weight_decay
      :type:  float
      :value: 0.0



   .. py:attribute:: weight_decay_exclude
      :type:  list[str] | None
      :value: None



   .. py:attribute:: weight_decay_include
      :type:  list[str] | None
      :value: None



   .. py:attribute:: grad_clip_norm
      :type:  float | None
      :value: None



   .. py:attribute:: use_sharded_clip_norm
      :type:  bool
      :value: True



   .. py:attribute:: grad_clip_value
      :type:  float | None
      :value: None



   .. py:attribute:: nesterov
      :type:  bool
      :value: False



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



.. py:function:: build_optimizer(optimizer_config)

   Build optimizer from config.

   :param optimizer_config: ConfigDict for optimizer.
   :type optimizer_config: OptimizerConfig

   :returns: Optimizer.
   :rtype: optax.GradientTransformation


.. py:function:: build_optimizer_function(optimizer_config, learning_rate)

   Build optimizer class function from config.

   By default, it supports Adam, AdamW, and SGD. To add custom optimizers, overwrite the
   function build_extra_optimizer_function.

   :param optimizer_config: ConfigDict for optimizer.
   :type optimizer_config: OptimizerConfig
   :param learning_rate: Learning rate schedule.
   :type learning_rate: float | optax.Schedule

   :returns: Optimizer class function.
   :rtype: Callable


.. py:function:: build_gradient_transformations(optimizer_config)

   Build gradient transformations from config.

   By default, it supports gradient clipping by norm and value, and weight decay. We distinguish
   between pre- and post-optimizer gradient transformations. Pre-optimizer
   gradient transformations are applied before the optimizer, e.g. gradient clipping. Post-optimizer
   gradient transformations are applied after the optimizer.

   :param optimizer_config: ConfigDict for optimizer
   :type optimizer_config: ConfigDict

   :returns:

             Tuple of pre-optimizer and
                 post-optimizer gradient transformations.
   :rtype: Tuple[List[optax.GradientTransformation], List[optax.GradientTransformation]]


.. py:function:: clip_by_global_norm_sharded(max_norm)

   Clip gradients by global norm.

   This extends optax.clip_by_global_norm to work with sharded gradients.

   :param max_norm: Maximum norm.
   :type max_norm: float

   :returns: Gradient transformation.
   :rtype: optax.GradientTransformation


