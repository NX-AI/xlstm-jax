import importlib
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import optax

from xlstm_jax.configs import ConfigDict
from xlstm_jax.import_utils import resolve_import_from_string


@dataclass(kw_only=True, frozen=True)
class SchedulerConfig(ConfigDict):
    """Configuration for learning rate scheduler.

    Attributes:
        lr (float): Initial/peak learning rate of the main scheduler.
        name (Literal): Name of the learning rate schedule. The supported schedules are "constant", "cosine_decay", "exponential_decay", and "linear".
        decay_steps (int): Number of steps for the learning rate schedule, including warmup and cooldown.
        end_lr (float | None): Final learning rate before the cooldown. This is mutually exclusive with end_lr_factor.
        end_lr_factor (float | None): Factor to multiply initial learning rate to get final learning rate before the cooldown. This is mutually exclusive with end_lr.
        cooldown_steps (int): Number of steps for cooldown.
        warmup_steps (int): Number of steps for warmup.
        cooldown_lr (float): Final learning rate for cooldown.
    """

    lr: float
    name: Literal["constant", "cosine_decay", "exponential_decay", "linear"] = "constant"
    decay_steps: int = 0
    end_lr: float | None = None
    end_lr_factor: float | None = None
    cooldown_steps: int = 0
    warmup_steps: int = 0
    cooldown_lr: float = 0.0


def build_lr_scheduler(scheduler_config: ConfigDict) -> optax.Schedule:
    """Build learning rate schedule from config.

    By default, it supports constant, linear, cosine decay, and exponential decay,
    all with warmup and cooldown.

    Args:
        scheduler_config (ConfigDict): ConfigDict for learning rate schedule.

    Returns:
        Callable: Learning rate schedule function.
    """
    lr = scheduler_config.lr
    scheduler_name = scheduler_config.name
    decay_steps = scheduler_config.decay_steps
    warmup_steps = scheduler_config.warmup_steps
    cooldown_steps = scheduler_config.cooldown_steps
    cooldown_lr = scheduler_config.cooldown_lr
    # Verify dependencies between config attributes.
    main_scheduler_steps = decay_steps - warmup_steps - cooldown_steps
    assert (
        main_scheduler_steps >= 0
    ), f"Decay steps includes warmup and cooldown steps, and must be at least of this size. Instead got {decay_steps} decay steps, {warmup_steps} warmup steps, and {cooldown_steps} cooldown steps."
    end_lr, end_lr_factor = None, None
    if scheduler_config.end_lr is not None:
        end_lr = scheduler_config.end_lr
        end_lr_factor = end_lr / lr
    if scheduler_config.end_lr_factor is not None:
        if end_lr_factor is not None:
            assert (
                end_lr_factor == scheduler_config.end_lr_factor
            ), f"end_lr and end_lr_factor are mutually exclusive and must be consistent. Instead got end_lr={end_lr} and end_lr_factor={scheduler_config.end_lr_factor} with learning rate {lr}."
        end_lr = scheduler_config.end_lr_factor * lr
        end_lr_factor = scheduler_config.end_lr_factor
    if end_lr is None and end_lr_factor is None:
        end_lr = 0.0
        end_lr_factor = 0.0
    # Build main learning rate schedule.
    lr_schedule = None
    if scheduler_name is None or scheduler_name == "constant":
        lr_schedule = optax.constant_schedule(lr)
        end_lr, end_lr_factor = lr, 1.0
    elif scheduler_name == "cosine_decay":
        lr_schedule = optax.cosine_decay_schedule(
            init_value=lr,
            decay_steps=main_scheduler_steps,
            alpha=end_lr_factor,
        )
    elif scheduler_name == "exponential_decay":
        lr_schedule = optax.exponential_decay(
            init_value=lr,
            decay_rate=end_lr_factor,
            transition_steps=main_scheduler_steps,
            staircase=False,
        )
    elif scheduler_name == "linear":
        lr_schedule = optax.linear_schedule(
            init_value=lr,
            end_value=end_lr,
            transition_steps=main_scheduler_steps,
        )
    else:
        raise ValueError(f"Unknown learning rate schedule {scheduler_name}.")
    # Add warmup and cooldown.
    schedules = [lr_schedule]
    boundaries = []
    if warmup_steps > 0:
        schedules.insert(
            0,
            optax.linear_schedule(
                init_value=0.0,
                end_value=lr,
                transition_steps=warmup_steps,
            ),
        )
        boundaries.insert(0, warmup_steps)
    if cooldown_steps > 0:
        schedules.append(
            optax.linear_schedule(
                init_value=end_lr,
                end_value=cooldown_lr,
                transition_steps=cooldown_steps,
            )
        )
        boundaries.append(decay_steps - cooldown_steps)
    lr_schedule = optax.join_schedules(
        schedules=schedules,
        boundaries=boundaries,
    )
    return lr_schedule
