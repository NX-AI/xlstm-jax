import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import jax
import jax.numpy as jnp
import optax

from xlstm_jax.configs import ConfigDict
from xlstm_jax.trainer.base.param_utils import get_param_mask_fn, get_sharded_global_norm

from .scheduler import SchedulerConfig, build_lr_scheduler

PyTree = Any


@dataclass(kw_only=True, frozen=True)
class OptimizerConfig(ConfigDict):
    """
    Configuration for optimizer.

    Attributes:
        name (str): Name of the optimizer. The supported optimizers are "adam", "adamw", "sgd", "nadam", "adamax",
            "radam", "nadamw", "adamax", and "lamb".
        scheduler (SchedulerConfig): Configuration for learning rate scheduler.
        beta1 (float): Exponential decay rate for the first moment estimates. This includes momentum in SGD.
        beta2 (float): Exponential decay rate for the second moment estimates.
        eps (float): Epsilon value for numerical stability in Adam-like optimizers.
        weight_decay (float): Weight decay coefficient.
        weight_decay_exclude (Sequence[re.Pattern] | None): List of regex patterns to exclude from weight decay.
            Parameter names are flattened and joined with ".". Mutually exclusive with weight_decay_include.
        weight_decay_include (Sequence[re.Pattern] | None): List of regex patterns to include in weight decay.
            Parameter names are flattened and joined with ".". Mutually exclusive with weight_decay_exclude.
            If neither exclude nor include is set, all parameters are included.
        grad_clip_norm (float | None): Global norm to clip gradients.
        use_sharded_clip_norm (bool): Whether to calculate the global norm for clipping over all shards of the
            parameter (True), or only calculate the grad norm for local shards (False). If True, may introduce a small
            communication overhead, but reproduces the behavior of the original implementation for sharded parameters.
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
    use_sharded_clip_norm: bool = True
    grad_clip_value: float | None = None
    nesterov: bool = False


def build_optimizer(optimizer_config: OptimizerConfig) -> tuple[optax.GradientTransformation, optax.Schedule]:
    """
    Build optimizer from config.

    Args:
        optimizer_config (OptimizerConfig): ConfigDict for optimizer.

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
    optimizer_config: OptimizerConfig, learning_rate: float | optax.Schedule
) -> optax.GradientTransformation:
    """
    Build optimizer class function from config.

    By default, it supports Adam, AdamW, and SGD. To add custom optimizers, overwrite the
    function build_extra_optimizer_function.

    Args:
        optimizer_config (OptimizerConfig): ConfigDict for optimizer.
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
            mask=get_param_mask_fn(
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


def build_gradient_transformations(
    optimizer_config: OptimizerConfig,
) -> tuple[list[optax.GradientTransformation], list[optax.GradientTransformation]]:
    """
    Build gradient transformations from config.

    By default, it supports gradient clipping by norm and value, and weight decay. We distinguish
    between pre- and post-optimizer gradient transformations. Pre-optimizer
    gradient transformations are applied before the optimizer, e.g. gradient clipping. Post-optimizer
    gradient transformations are applied after the optimizer.

    Args:
        optimizer_config (ConfigDict): ConfigDict for optimizer

    Returns:
        Tuple[List[optax.GradientTransformation], List[optax.GradientTransformation]]: Tuple of pre-optimizer and
            post-optimizer gradient transformations.
    """
    optimizer_name = optimizer_config.name
    optimizer_name = optimizer_name.lower()
    pre_trans, post_trans = [], []

    # Gradient clipping by norm.
    if optimizer_config.grad_clip_norm is not None:
        if optimizer_config.use_sharded_clip_norm:
            pre_trans.append(clip_by_global_norm_sharded(optimizer_config.grad_clip_norm))
        else:
            pre_trans.append(optax.clip_by_global_norm(optimizer_config.grad_clip_norm))

    # Gradient clipping by value.
    if optimizer_config.grad_clip_value is not None:
        pre_trans.append(optax.clip(optimizer_config.grad_clip_value))

    # Weight decay.
    if optimizer_config.weight_decay > 0.0 and optimizer_name not in ["adamw"]:
        post_trans.append(
            optax.add_decayed_weights(
                optimizer_config.weight_decay,
                mask=get_param_mask_fn(
                    exclude=optimizer_config.weight_decay_exclude, include=optimizer_config.weight_decay_include
                ),
            )
        )

    return pre_trans, post_trans


def clip_by_global_norm_sharded(max_norm: float) -> optax.GradientTransformation:
    """
    Clip gradients by global norm.

    This extends optax.clip_by_global_norm to work with sharded gradients.

    Args:
        max_norm (float): Maximum norm.
        parallel (ParallelConfig): Parallel configuration.

    Returns:
        optax.GradientTransformation: Gradient transformation.
    """

    def update_fn(updates, state, params=None) -> tuple[PyTree, PyTree]:
        # Clip gradients by global norm.
        # - updates: gradients
        # - state: optimizer state of transformation (empty in this case)
        # - params: (optional) model params, unused here
        del params
        g_norm, _ = get_sharded_global_norm(updates)
        g_norm = jnp.maximum(max_norm, g_norm)
        updates = jax.tree_util.tree_map(lambda x: x * (max_norm / g_norm), updates)
        return updates, state

    return optax.GradientTransformation(lambda _: optax.EmptyState(), update_fn)
