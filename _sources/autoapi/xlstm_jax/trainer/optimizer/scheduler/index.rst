xlstm_jax.trainer.optimizer.scheduler
=====================================

.. py:module:: xlstm_jax.trainer.optimizer.scheduler


Classes
-------

.. autoapisummary::

   xlstm_jax.trainer.optimizer.scheduler.SchedulerConfig


Functions
---------

.. autoapisummary::

   xlstm_jax.trainer.optimizer.scheduler.build_lr_scheduler


Module Contents
---------------

.. py:class:: SchedulerConfig

   Bases: :py:obj:`xlstm_jax.configs.ConfigDict`


   Configuration for learning rate scheduler.

   .. attribute:: lr

      Initial/peak learning rate of the main scheduler.

      :type: float

   .. attribute:: name

      Name of the learning rate schedule. The supported schedules are "constant", "cosine_decay",
      "exponential_decay", and "linear".

      :type: Literal

   .. attribute:: decay_steps

      Number of steps for the learning rate schedule, including warmup and cooldown. If not
      provided, it is defined at runtime in the start script.

      :type: int | None

   .. attribute:: end_lr

      Final learning rate before the cooldown. This is mutually exclusive with end_lr_factor.

      :type: float | None

   .. attribute:: end_lr_factor

      Factor to multiply initial learning rate to get final learning rate before the
      cooldown. This is mutually exclusive with end_lr.

      :type: float | None

   .. attribute:: cooldown_steps

      Number of steps for cooldown.

      :type: int

   .. attribute:: warmup_steps

      Number of steps for warmup.

      :type: int

   .. attribute:: cooldown_lr

      Final learning rate for cooldown.

      :type: float


   .. py:attribute:: lr
      :type:  float


   .. py:attribute:: name
      :type:  str
      :value: 'constant'



   .. py:attribute:: decay_steps
      :type:  int | None
      :value: 0



   .. py:attribute:: end_lr
      :type:  float | None
      :value: None



   .. py:attribute:: end_lr_factor
      :type:  float | None
      :value: None



   .. py:attribute:: cooldown_steps
      :type:  int
      :value: 0



   .. py:attribute:: warmup_steps
      :type:  int
      :value: 0



   .. py:attribute:: cooldown_lr
      :type:  float
      :value: 0.0



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



.. py:function:: build_lr_scheduler(scheduler_config)

   Build learning rate schedule from config.

   By default, it supports constant, linear, cosine decay, and exponential decay,
   all with warmup and cooldown.

   :param scheduler_config: ConfigDict for learning rate schedule.
   :type scheduler_config: ConfigDict

   :returns: Learning rate schedule function.
   :rtype: Callable


