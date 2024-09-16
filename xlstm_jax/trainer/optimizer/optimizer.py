import importlib
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from xlstm_jax.configs import ConfigDict

import jax
import optax

from .scheduler import SchedulerConfig, build_lr_scheduler

PyTree = Any


@dataclass(kw_only=True, frozen=True)
class OptimizerConfig(ConfigDict):
    """Configuration for optimizer.

    Attributes:
        name (str): Name of the optimizer. The supported optimizers are "adam", "adamw", "sgd", "nadam", "adamax", "radam", "nadamw", "adamax", and "lamb".
        scheduler (SchedulerConfig): Configuration for learning rate scheduler.
        beta1 (float): Exponential decay rate for the first moment estimates. This includes momentum in SGD.
        beta2 (float): Exponential decay rate for the second moment estimates.
        eps (float): Epsilon value for numerical stability in Adam-like optimizers.
        weight_decay (float): Weight decay coefficient.
        weight_decay_exclude (Sequence[re.Pattern] | None): List of regex patterns to exclude from weight decay. Parameter names are flattened and joined with ".". Mutually exclusive with weight_decay_include.
        weight_decay_include (Sequence[re.Pattern] | None): List of regex patterns to include in weight decay. Parameter names are flattened and joined with ".". Mutually exclusive with weight_decay_exclude. If neither exclude nor include is set, all parameters are included.
        grad_clip_norm (float | None): Global norm to clip gradients. Warning: if parameters are explicitly sharded (i.e. with shard_map), the global norm is computed only across parameters on the device. If sharded via logical constraints and pjit, the global norm is computed across all parameters, but may introduce communication overhead.
        grad_clip_value (float | None): Value to clip gradients element-wise.
        nesterov (bool): Whether to use Nesterov momentum in SGD.
    """

    name: Literal["adam", "adamw", "sgd", "nadam", "adamax", "radam", "nadamw", "adamaxw", "lamb"]
    scheduler: SchedulerConfig
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0
    weight_decay_exclude: Sequence[re.Pattern] | None = None
    weight_decay_include: Sequence[re.Pattern] | None = None
    grad_clip_norm: float | None = None
    grad_clip_value: float | None = None
    nesterov: bool = False


def build_optimizer(optimizer_config: ConfigDict) -> tuple[optax.GradientTransformation, optax.Schedule]:
    """Build optimizer from config.

    Args:
        optimizer_config (ConfigDict): ConfigDict for optimizer.

    Returns:
        optax.GradientTransformation: Optimizer.
    """
    # Build elements of optimizer
    lr_schedule = build_lr_scheduler(optimizer_config.scheduler)
    optimizer = build_optimizer_function(optimizer_config, lr_schedule)
    pre_grad_trans, post_grad_trans = build_gradient_transformations(optimizer_config)
    # Put everything together
    optimizer = optax.chain(*pre_grad_trans, optimizer, *post_grad_trans)
    return optimizer, lr_schedule


def build_optimizer_function(
    optimizer_config: ConfigDict, learning_rate: float | optax.Schedule
) -> optax.GradientTransformation:
    """Build optimizer class function from config.

    By default, it supports Adam, AdamW, and SGD. To add custom optimizers, overwrite the
    function build_extra_optimizer_function.

    Args:
        optimizer_config (ConfigDict): ConfigDict for optimizer.
        learning_rate (float | optax.Schedule): Learning rate schedule.

    Returns:
        Callable: Optimizer class function.
    """
    # Build optimizer class
    optimizer_name = optimizer_config.name
    optimizer_name = optimizer_name.lower()
    opt_class = None
    if optimizer_name in ["adam", "nadam", "adamax", "radam"]:
        opt_class = getattr(optax, optimizer_name)(
            learning_rate,
            b1=optimizer_config.beta1,
            b2=optimizer_config.beta2,
            eps=optimizer_config.eps,
        )
    elif optimizer_name in ["adamw", "nadamw", "adamaxw", "lamb"]:
        opt_class = getattr(optax, optimizer_name)(
            learning_rate,
            b1=optimizer_config.beta1,
            b2=optimizer_config.beta2,
            eps=optimizer_config.eps,
            weight_decay=optimizer_config.weight_decay,
            mask=_get_param_mask_fn(
                exclude=optimizer_config.weight_decay_exclude, include=optimizer_config.weight_decay_include
            ),
        )
    elif optimizer_name == "sgd":
        opt_class = optax.sgd(
            learning_rate,
            momentum=optimizer_config.beta1,
            nesterov=optimizer_config.nesterov,
        )
    else:
        raise ValueError(f"Unknown optimizer {optimizer_name}.")

    return opt_class


def _key_path_to_str(path: jax.tree_util.KeyPath) -> str:
    """Converts a path to a string.

    An adjusted version of jax.tree_util.keystr to be more intuitive
    and fitting to our flatten_dict method.

    Args:
        path (jax.tree_util.KeyPath): Path.

    Returns:
        str: Path as string.
    """
    cleaned_keys = []
    for key in path:
        if isinstance(key, jax.tree_util.DictKey):
            cleaned_keys.append(f"{key.key}")
        elif isinstance(key, jax.tree_util.SequenceKey):
            cleaned_keys.append(f"{key.idx}")
        elif isinstance(key, jax.tree_util.GetAttrKey):
            cleaned_keys.append(key.name)
        else:
            cleaned_keys.append(str(key))
    return ".".join(cleaned_keys)


def _get_param_mask_fn(
    exclude: Sequence[str] | None, include: Sequence[str] | None = None
) -> Callable[[PyTree], PyTree]:
    """Returns a function that generates a mask, which can for instance be used for weight decay.

    Args:
        exclude (Sequence[str]): List of strings to exclude.
        include (Sequence[str]): List of strings to include. If None, all parameters except those in exclude are included.

    Returns:
        Callable[[PyTree], PyTree]: Function that generates a mask.
    """
    assert exclude is None or include is None, "Only one of exclude or include can be set."

    def is_param_included(path, _):
        param_name = _key_path_to_str(path)
        if exclude is not None:
            if any(re.search(excl, param_name) for excl in exclude):
                return False
            return True
        elif include is not None:
            if any(re.search(incl, param_name) for incl in include):
                return True
            return False
        else:
            return True

    def mask_fn(params: PyTree):
        mask_tree = jax.tree_util.tree_map_with_path(
            is_param_included,
            params,
        )
        print(mask_tree)
        return mask_tree

    return mask_fn


def build_gradient_transformations(
    optimizer_config: ConfigDict,
) -> tuple[list[optax.GradientTransformation], list[optax.GradientTransformation]]:
    """Build gradient transformations from config.

    By default, it supports gradient clipping by norm and value, and weight decay. We distinguish
    between pre- and post-optimizer gradient transformations. Pre-optimizer
    gradient transformations are applied before the optimizer, e.g. gradient clipping. Post-optimizer
    gradient transformations are applied after the optimizer.

    Args:
        optimizer_config (ConfigDict): ConfigDict for optimizer

    Returns:
        Tuple[List[optax.GradientTransformation], List[optax.GradientTransformation]]: Tuple of pre-optimizer and post-optimizer gradient transformations.
    """
    # Gradient transformation
    optimizer_name = optimizer_config.name
    optimizer_name = optimizer_name.lower()
    pre_trans, post_trans = [], []

    if optimizer_config.grad_clip_norm is not None:
        pre_trans.append(optax.clip_by_global_norm(optimizer_config.grad_clip_norm))
    if optimizer_config.grad_clip_value is not None:
        pre_trans.append(optax.clip(optimizer_config.grad_clip_value))
    if optimizer_config.weight_decay > 0.0 and optimizer_name not in ["adamw"]:
        post_trans.append(
            optax.add_decayed_weights(
                optimizer_config.weight_decay,
                mask=_get_param_mask_fn(
                    exclude=optimizer_config.weight_decay_exclude, include=optimizer_config.weight_decay_include
                ),
            )
        )

    return pre_trans, post_trans
