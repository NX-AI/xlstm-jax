xlstm_jax.models.xlstm_pytorch.blocks.mlstm.backend.layer_factory
=================================================================

.. py:module:: xlstm_jax.models.xlstm_pytorch.blocks.mlstm.backend.layer_factory


Attributes
----------

.. autoapisummary::

   xlstm_jax.models.xlstm_pytorch.blocks.mlstm.backend.layer_factory.LOGGER


Functions
---------

.. autoapisummary::

   xlstm_jax.models.xlstm_pytorch.blocks.mlstm.backend.layer_factory.create_layer


Module Contents
---------------

.. py:data:: LOGGER

.. py:function:: create_layer(config, registry, layer_cfg_key, nameandkwargs = None, set_cfg_kwargs = True, **kwargs)

   Create a layer from a config dataclass object, a layer class registry and a layer config key.

   The layer config
   key is the name of the attribute in the config object that contains the layer config.

   This function assumes that the config object is a hierarchical dataclass object, i.e. it contains configurable
   layers of type `NameAndKwargs` (which is a dataclass with keys `name` and `kwargs`).
   The `name` attribute is used to get the layer class from the registry and the `kwargs` attribute is used to
   instantiate the layer class.

   If `nameandkwargs` is not None, it is used instead of the `NameAndKwargs` attribute in the config object.

   If the layer class has a `config_class` attribute, the `kwargs` attribute is used to instantiate the layer config.

   :param config: config object created
   :type config: dataclass
   :param registry: layer class registry
   :type registry: dict[str, Type]
   :param layer_cfg_key: layer config key
   :type layer_cfg_key: str
   :param nameandkwargs: layer name and kwargs. Defaults to None.
   :type nameandkwargs: NameAndKwargs, optional
   :param set_cfg_kwargs: whether to set the kwargs attribute to the instantiated config dataclass object
                          in the config object. Defaults to True.
   :type set_cfg_kwargs: bool, optional

   :returns: layer instance
   :rtype: LayerInterface


