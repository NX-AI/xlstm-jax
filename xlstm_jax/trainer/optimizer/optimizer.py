import logging
import re
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax

from xlstm_jax.common_types import PyTree
from xlstm_jax.configs import ConfigDict
from xlstm_jax.trainer.base.param_utils import get_param_mask_fn, get_sharded_global_norm

from .ademamix import ademamix, alpha_scheduler, beta3_scheduler
from .scheduler import SchedulerConfig, build_lr_scheduler

LOGGER = logging.getLogger(__name__)


@dataclass(kw_only=True, frozen=False)
class OptimizerConfig(ConfigDict):
    """
    Configuration for optimizer.

    Attributes:
        name (str): Name of the optimizer. The supported optimizers are "adam", "adamw", "sgd", "nadam", "adamax",
            "radam", "nadamw", "adamax", and "lamb".
        scheduler (SchedulerConfig): Configuration for learning rate scheduler.
        beta1 (float): Exponential decay rate for the first moment estimates. This includes momentum in SGD.
        beta2 (float): Exponential decay rate for the second moment estimates.
        beta3 (float): For AdEMAMix, exponential decay rate for the slow EMA.
        alpha (float): For AdEMAMix, mixing coefficient for the linear combination of the fast and slow EMAs.
            Commonly in the range 5-10, with Mamba models performing best at 8. TODO: Update with xLSTM results.
        eps (float): Epsilon value for numerical stability in Adam-like optimizers.
        weight_decay (float): Weight decay coefficient.
        weight_decay_exclude (list[str] | None): List of regex patterns of `re.Pattern` to exclude from weight decay.
            Parameter names are flattened and joined with ".". Mutually exclusive with weight_decay_include.
        weight_decay_include (list[str] | None): List of regex patterns of `re.Pattern` to include in weight decay.
            Parameter names are flattened and joined with ".". Mutually exclusive with weight_decay_exclude.
            If neither exclude nor include is set, all parameters are included.
        grad_clip_norm (float | None): Global norm to clip gradients.
        use_sharded_clip_norm (bool): Whether to calculate the global norm for clipping over all shards of the
            parameter (True), or only calculate the grad norm for local shards (False). If True, may introduce a small
            communication overhead, but reproduces the behavior of the original implementation for sharded parameters.
        grad_clip_value (float | None): Value to clip gradients element-wise.
        nesterov (bool): Whether to use Nesterov momentum in SGD.
    """

    name: str
    scheduler: SchedulerConfig
    beta1: float = 0.9
    beta2: float = 0.999
    beta3: float = 0.9999
    alpha: float = 8.0
    eps: float = 1e-8
    weight_decay: float = 0.0
    weight_decay_exclude: list[str] | None = None
    weight_decay_include: list[str] | None = None
    grad_clip_norm: float | None = None
    use_sharded_clip_norm: bool = True
    grad_clip_value: float | None = None
    nesterov: bool = False

    def __post_init__(self):
        optimizers = [
            "adam",
            "adamw",
            "sgd",
            "nadam",
            "adamax",
            "radam",
            "nadamw",
            "adamaxw",
            "lamb",
            "ademamix",
        ]
        assert (
            self.name in optimizers
        ), f"Unknown optimizer {self.name} provided in config, supported are: {optimizers}."


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
    # Compile incoming strings into regex patterns.  TODO: check if this works in a unit test
    weight_decay_exclude_re = (
        [re.compile(pattern) for pattern in optimizer_config.weight_decay_exclude]
        if optimizer_config.weight_decay_exclude
        else None
    )
    weight_decay_include_re = (
        [re.compile(pattern) for pattern in optimizer_config.weight_decay_include]
        if optimizer_config.weight_decay_include
        else None
    )

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
            mask=get_param_mask_fn(exclude=weight_decay_exclude_re, include=weight_decay_include_re),
        )
    elif optimizer_name == "sgd":
        opt_class = optax.sgd(
            learning_rate,
            momentum=optimizer_config.beta1,
            nesterov=optimizer_config.nesterov,
        )
    elif optimizer_name == "ademamix":
        total_train_steps = optimizer_config.scheduler.decay_steps
        if total_train_steps <= 0:
            LOGGER.warning(
                "AdEMAMix uses the decay steps from scheduler to adjust its alpha and beta scheduler. "
                "However, none was given. Will be setting it by default to 100k steps."
            )
            total_train_steps = 100_000
        opt_class = ademamix(
            learning_rate,
            b1=optimizer_config.beta1,
            b2=optimizer_config.beta2,
            b3=optimizer_config.beta3,
            alpha=optimizer_config.alpha,
            alpha_scheduler=alpha_scheduler(optimizer_config.alpha, 0.0, warmup=total_train_steps),
            b3_scheduler=beta3_scheduler(optimizer_config.beta3, optimizer_config.beta1, warmup=total_train_steps),
            eps=optimizer_config.eps,
            weight_decay=optimizer_config.weight_decay,
            mask=get_param_mask_fn(
                exclude=optimizer_config.weight_decay_exclude, include=optimizer_config.weight_decay_include
            ),
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

    # Compile incoming strings into regex patterns.
    weight_decay_exclude_re = (
        [re.compile(pattern) for pattern in optimizer_config.weight_decay_exclude]
        if optimizer_config.weight_decay_exclude
        else None
    )
    weight_decay_include_re = (
        [re.compile(pattern) for pattern in optimizer_config.weight_decay_include]
        if optimizer_config.weight_decay_include
        else None
    )

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
    if optimizer_config.weight_decay > 0.0 and optimizer_name not in ["adamw", "nadamw", "adamaxw", "lamb", "ademamix"]:
        post_trans.append(
            optax.add_decayed_weights(
                optimizer_config.weight_decay,
                mask=get_param_mask_fn(exclude=weight_decay_exclude_re, include=weight_decay_include_re),
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
