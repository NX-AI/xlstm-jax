import importlib
import inspect
import json
import re
from dataclasses import dataclass, is_dataclass
from pathlib import Path
from types import NoneType, UnionType
from typing import Any, Literal, get_args, get_origin

from xlstm_jax.import_utils import class_to_name


def _instantiate_class_empty(cls: Any) -> Any:
    """
    Instantiates a class with no keywords.

    Args:
        cls: The class to be instantiated. If it is a Union containing None, None will be returned

    Returns:
        The object of type cls.
    """
    # checks for a union class
    if get_origin(cls) is UnionType:
        if None in get_args(cls):
            return None
        else:
            return get_args(cls)[0]()
    else:
        return cls()


def _get_annotations_full_dataclass(cls: dataclass) -> dict[str, type]:
    """
    Get annotations for all attributes of a dataclass via reverse getmro resolution order.

    Args:
        cls: Dataclass

    Returns:
        Annotations of all attributes of the dataclass.
    """
    annotations = {}
    for supercls in reversed(inspect.getmro(cls)):
        if is_dataclass(supercls):
            annotations.update(inspect.get_annotations(supercls))
    return annotations


def _parse_python_arguments_to_json(arguments: str) -> dict | None:
    """
    Parse badly serialized dataclasses via json into dictionary.
    E.g. "mLSTMBackendTritonConfig(autocast_kernel_dtype='float32')"
    Unfortunately our previous configs are not serialized in a good format.

    Args:
        arguments: A string of the form SomeClass(argument1=..., argument2=...)

    Returns:
        Dictionary of the keyword arguments: {"argument1": ..., "argument2": ...}
    """
    if arguments[0] == "{" and arguments[-1] == "}":
        arguments = re.sub(r": ([^:},]+)", r': "\1"', re.sub("'([a-zA-Z_]+)':", r'"\1":', arguments))
        try:
            args = json.loads(arguments)
        except json.JSONDecodeError:
            return None
        return args
    if "(" in arguments and ")" in arguments:
        args = arguments[arguments.index("(") + 1 : len(arguments) - arguments[::-1].index(")") - 1]
        args_json = (
            "{"
            + re.sub(
                r"([a-zA-Z_]+)\=",
                r'"\1": ',
                args.replace("'", '"')
                .replace("(", "[")
                .replace(")", "]")
                .replace("None", "null")
                .replace("True", "true")
                .replace("False", "false"),
            )
            + "}"
        )
        try:
            args_dict = json.loads(args_json)
            return args_dict
        except json.JSONDecodeError:
            return None
    else:
        return arguments


def _parsed_string(data: str, cfg_class: type, strict_classname_parsing: bool = False) -> Any:
    mat = re.match("<class '(.*)'>", data)
    if mat and len(mat.groups()) > 0:
        class_path = mat.group(1)
    else:
        if strict_classname_parsing:
            raise ValueError(f"Bad class path {data}")
        if not any(a in data for a in "( ,)}{"):
            class_path = data
        else:
            # Try to parse calling arguments
            kwargs = _parse_python_arguments_to_json(data)
            if kwargs is not None:
                return ConfigDict.from_dict(config_class=cfg_class, data=kwargs)
            return _instantiate_class_empty(cfg_class)
    if "." in class_path:
        class_module = importlib.import_module(".".join(class_path.split(".")[:-1]))
        resolved_class = getattr(class_module, class_path.split(".")[-1])
        return resolved_class
    elif class_path == "None":
        return None
    raise ValueError(f"Could not parse {data} to {cfg_class}")


@dataclass(kw_only=True, frozen=False)
class ConfigDict:
    def get(self, key: str, default=None):
        return getattr(self, key, default)

    def to_dict(self):
        """
        Converts the config to a dictionary.

        Helpful for saving to disk or logging.
        """
        d = {}

        for k, v in self.__dict__.items():
            if isinstance(v, ConfigDict) or hasattr(v, "to_dict"):
                d[k] = v.to_dict()
            elif isinstance(v, (tuple, list)):
                d[k] = tuple([x.to_dict() if isinstance(x, ConfigDict) or hasattr(v, "to_dict") else x for x in v])
            elif isinstance(v, Path):
                d[k] = v.as_posix()
            elif inspect.isclass(v):
                d[k] = class_to_name(v)
            elif isinstance(v, (int, float, str, bool)):
                d[k] = v
            elif v is None:
                d[k] = v
            else:
                d[k] = str(v)
        return d

    @staticmethod
    def from_dict(
        config_class: type,
        data: Any,
        strict_classname_parsing: bool = False,
        ignore_extensive_attributes: bool = True,
        none_to_zero_for_ints: bool = False,
    ) -> Any:
        """
        Utility for parsing dictionaries back into a nested dataclass structure, including
        arbitrary classes (ends nesting) and types (dtypes).
        Currently this is taylored towards the current logging system with the "hardly" invertable
        to_dict.

        Args:
            config_class: Typically a dataclass, but can be any other type as well
                If it is another type, the parser tries to create an object via
                config_class(**data) if data is a dictionary or config_class(data) else.
            data: Typically a dictionary that contains attributes of the dataclass.
                Can be any other kind of data.

        Returns:
            An object of type config_class that contains the data as attributes.

        """
        if is_dataclass(config_class) or (
            get_origin(config_class) == UnionType and any(is_dataclass(c) for c in get_args(config_class))
        ):
            if isinstance(data, dict):
                if get_origin(config_class) == UnionType:
                    config_class = get_args(config_class)[0]
                annotations = _get_annotations_full_dataclass(config_class)
                cfg_dict = {}
                for attr in data:
                    if attr not in annotations:
                        if ignore_extensive_attributes:
                            continue
                        raise ValueError(f"Undefined attribute {attr} for dataclass {config_class}")
                    val_parsed = ConfigDict.from_dict(annotations[attr], data[attr])
                    cfg_dict[attr] = val_parsed
                return config_class(**cfg_dict)
            # the "None" check is a leftover from bad serialization
            elif data is None or data == "None":
                if NoneType in get_args(config_class):
                    return None
                else:
                    raise ValueError(f"Could not parse {data} into {config_class}")
            elif isinstance(data, str):
                if get_origin(config_class) == UnionType:
                    config_class = get_args(config_class)[0]
                cfg_dict = _parse_python_arguments_to_json(data)
                if cfg_dict is not None:
                    return config_class(**cfg_dict)
            raise ValueError(f"Could not parse {data} into {config_class}")
        elif get_origin(config_class) is Literal:
            if data in get_args(config_class):
                return data
            raise ValueError(f"Bad literal value {data} of {config_class}: {get_args(config_class)}")
        else:
            if get_origin(config_class) == UnionType:
                config_class_opts = get_args(config_class)
            else:
                config_class_opts = [config_class]
            for cfg_class in config_class_opts:
                try:
                    if inspect.getmro(cfg_class)[0] == dict:
                        if isinstance(data, dict):
                            return {
                                key: _parsed_string(val, str, strict_classname_parsing=strict_classname_parsing)
                                if isinstance(val, str)
                                else val
                                for key, val in data.items()
                            }
                        if isinstance(data, str):
                            return _parsed_string(data, dict, strict_classname_parsing=strict_classname_parsing)
                        raise ValueError(f"Parsing error of {data} into {cfg_class}")
                    if issubclass(cfg_class, str) and not data.startswith("<class "):
                        if None in config_class_opts and data == "None":
                            return None
                        return str(data)
                    if issubclass(cfg_class, bool):
                        return bool(data)
                    if issubclass(cfg_class, int):
                        if none_to_zero_for_ints and (data is None or data == "None"):
                            return 0
                        return int(data)
                    if issubclass(cfg_class, float):
                        return float(data)
                    if cfg_class is NoneType:
                        if data == "None" or data is None:
                            return None
                        raise ValueError(f"Bad Value {data} for NoneType")
                    if isinstance(data, str):
                        return _parsed_string(
                            data, cfg_class=cfg_class, strict_classname_parsing=strict_classname_parsing
                        )
                    if inspect.getmro(cfg_class)[0] == list:
                        return list(data)
                    if inspect.getmro(cfg_class)[0] == tuple:
                        return tuple(data)
                    if isinstance(data, dict):
                        return cfg_class(**data)
                    return cfg_class(data)
                # TODO: how to improve this
                except (ValueError, TypeError, KeyError, AttributeError):
                    continue
            raise ValueError("Could not parse")
