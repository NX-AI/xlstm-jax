# Copyright JKU Linz 2023
# Maximilian Beck
import logging
from typing import Any

import dacite
from dacite import Config, from_dict

from .config_utils import NameAndKwargs

LOGGER = logging.getLogger(__name__)


# TODO what would be a typehint for a generic dataclass?
def create_layer(
    config,
    registry: dict[str, type],
    layer_cfg_key: str,
    nameandkwargs: NameAndKwargs = None,
    set_cfg_kwargs: bool = True,
    **kwargs,
) -> Any:
    """Create a layer from a config dataclass object, a layer class registry and a layer config key.
    The layer config key is the name of the attribute in the config object that contains the layer config.

    This function assumes that the config object is a hierarchical dataclass object, i.e. it contains configurable
    layers of type `NameAndKwargs` (which is a dataclass with keys `name` and `kwargs`).
    The `name` attribute is used to get the layer class from the registry and the `kwargs` attribute is used to
    instantiate the layer class.

    If `nameandkwargs` is not None, it is used instead of the `NameAndKwargs` attribute in the config object.

    If the layer class has a `config_class` attribute, the `kwargs` attribute is used to instantiate the layer config.

    Args:
        config (dataclass): config object created
        registry (dict[str, Type]): layer class registry
        layer_cfg_key (str): layer config key
        nameandkwargs (NameAndKwargs, optional): layer name and kwargs. Defaults to None.
        set_cfg_kwargs (bool, optional): whether to set the kwargs attribute to the instantiated config dataclass object
                                         in the config object. Defaults to True.

    Returns:
        LayerInterface: layer instance
    """

    def get_layer(name: str) -> type:
        if name in registry:
            return registry[name]
        else:
            raise ValueError(
                f"Unknown {layer_cfg_key} layer: {name}. Available {layer_cfg_key} layers: {list(registry.keys())}"
            )

    # clear registry in every case, since it is not needed anymore after it has
    # been used to create the layer
    if nameandkwargs is not None:
        cfg_name = nameandkwargs.name
        cfg_kwargs = nameandkwargs.kwargs
        nameandkwargs._registry = {}
    else:
        if hasattr(config, layer_cfg_key):
            layer_nameandkwargs = getattr(config, layer_cfg_key)
            cfg_name = layer_nameandkwargs.name
            cfg_kwargs = layer_nameandkwargs.kwargs
            if hasattr(layer_nameandkwargs, "_registry"):
                layer_nameandkwargs._registry = {}
        elif layer_cfg_key in config:
            cfg_name = config[layer_cfg_key]["name"]
            cfg_kwargs = config[layer_cfg_key]["kwargs"]
            if "_registry" in config[layer_cfg_key]:
                config[layer_cfg_key]["_registry"] = {}
        else:
            raise ValueError(
                f"Config {config} has no {layer_cfg_key} attribute or key. "
                "Edit config or check if layer is correctly named."
            )

    layer_class = get_layer(cfg_name)
    layer_config_class = getattr(layer_class, "config_class", None)
    if layer_config_class is None:
        assert not kwargs, (
            f"Layer {cfg_name} does not have a config class ",
            "but other kwargs is not None.",
        )
        return layer_class(**cfg_kwargs) if cfg_kwargs is not None else layer_class()
    else:
        # * create the layer config dataclass object
        # if we create multiple layers with the same config
        # (e.g. multiple transformer blocks), we need to make sure that the config
        # is created only once.
        if isinstance(cfg_kwargs, dict) or cfg_kwargs is None:
            try:
                layer_config = from_dict(
                    data_class=layer_config_class,
                    data=(cfg_kwargs if cfg_kwargs is not None else {}),
                    config=Config(strict=True, strict_unions_match=True),
                )
            except dacite.exceptions.UnexpectedDataError as e:
                LOGGER.error(
                    f"Dacite error: class: {layer_config_class}, kwargs: {cfg_kwargs}"
                )
                raise e
            layer_config.assign_model_config_params(model_config=config)
        else:
            layer_config = cfg_kwargs

        # * assign the config dataclass to the kwargs
        # this is used for printing the full config
        if set_cfg_kwargs:
            if nameandkwargs is not None:
                nameandkwargs.kwargs = layer_config
            else:
                setattr(
                    config,
                    layer_cfg_key,
                    NameAndKwargs(name=cfg_name, kwargs=layer_config),
                )

        if kwargs:
            layer = layer_class(layer_config, **kwargs)
        else:
            layer = layer_class(layer_config)

        return layer
