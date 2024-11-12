import logging
import sys
import traceback
from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

R = TypeVar("R")
P = ParamSpec("P")


def with_error_handling(
    flush_output: bool = True, logger: logging.Logger | None = None
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    A decorator that provides consistent error handling with optional output flushing.

    This decorator is used to circumvent a bug in Hydra. See here:
    https://github.com/facebookresearch/hydra/issues/2664#issuecomment-1857695600

    Args:
        flush_output: Whether to flush stdout and stderr in the finally block.
                      Defaults to True.
        logger: Optional logger instance to use for error logging.
                If None errors will be printed to stderr.

    Returns:
        A decorator function that wraps the target function.
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            try:
                return func(*args, **kwargs)
            except BaseException as e:
                error_msg = f"Error in function {func.__name__}: {str(e)}"
                stack_trace = "".join(traceback.format_exc())

                if logger is not None:
                    logger.error(error_msg)
                    logger.error(stack_trace)
                else:
                    traceback.print_exc(file=sys.stderr)
                raise
            finally:
                if flush_output:
                    sys.stdout.flush()
                    sys.stderr.flush()

        return wrapper

    return decorator
