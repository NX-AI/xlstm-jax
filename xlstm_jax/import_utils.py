#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import importlib
import inspect
import sys
from typing import Any


def resolve_import(import_path: str | Any) -> Any:
    """
    Resolves an import from a string or returns the input.

    Args:
        import_path (str | Any): The import path or the object itself.

    Returns:
        Any: The resolved object.
    """
    if isinstance(import_path, str):
        import_path = resolve_import_from_string(import_path)
    return import_path


def resolve_import_from_string(import_string: str) -> Any:
    """
    Resolves an import from a string.

    Args:
        import_string (str): The import string.

    Returns:
        Any: The resolved object.
    """
    if "." in import_string:
        module_path, class_name = import_string.rsplit(".", 1)
        module = importlib.import_module(module_path)
        resolved_class = getattr(module, class_name)
    else:
        resolved_class = getattr(sys.modules[__name__], import_string)
    return resolved_class


def class_to_name(x: Any) -> str | Any:
    """
    Converts a class to a string representation.

    Useful for logging/saving the class name.

    Args:
        x (Any): The input object.

    Returns:
        str | Any: The string representation of the object.
    """
    return (inspect.getmodule(x).__name__ + "." + x.__name__) if inspect.isclass(x) else x
