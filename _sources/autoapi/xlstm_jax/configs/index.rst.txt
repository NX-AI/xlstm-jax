xlstm_jax.configs
=================

.. py:module:: xlstm_jax.configs


Classes
-------

.. autoapisummary::

   xlstm_jax.configs.ConfigDict


Functions
---------

.. autoapisummary::

   xlstm_jax.configs._instantiate_class_empty
   xlstm_jax.configs._get_annotations_full_dataclass
   xlstm_jax.configs._parse_python_arguments_to_json
   xlstm_jax.configs._parsed_string


Module Contents
---------------

.. py:function:: _instantiate_class_empty(cls)

   Instantiates a class with no keywords.

   :param cls: The class to be instantiated. If it is a Union containing None, None will be returned

   :returns: The object of type cls.


.. py:function:: _get_annotations_full_dataclass(cls)

   Get annotations for all attributes of a dataclass via reverse getmro resolution order.

   :param cls: Dataclass

   :returns: Annotations of all attributes of the dataclass.


.. py:function:: _parse_python_arguments_to_json(arguments)

   Parse badly serialized dataclasses via json into dictionary.

   E.g. "mLSTMBackendTritonConfig(autocast_kernel_dtype='float32')"
   Unfortunately our previous configs are not serialized in a good format.

   :param arguments: A string of the form `SomeClass(argument1=..., argument2=...)`

   :returns: {"argument1": ..., "argument2": ...}
   :rtype: Dictionary of the keyword arguments


.. py:function:: _parsed_string(data, cfg_class, strict_classname_parsing = False)

.. py:class:: ConfigDict

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



