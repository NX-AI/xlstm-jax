"""
Adapted from Apple's official AdeMAMix implementation:
https://github.com/apple/ml-ademamix/blob/main/optax/ademamix.py

TODO: Check license and add it here.
"""

import functools
from collections.abc import Callable
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import optax
from jax import tree_util as jtu
from optax._src import base, combine, numerics, transform, utils

from xlstm_jax.common_types import PyTree


def alpha_scheduler(alpha: float, alpha_start: float = 0.0, warmup: int = 0) -> optax.Schedule:
    """
    Linear scheduler for the mixing coefficient alpha in AdEMAMix.

    Args:
        alpha: Final value of alpha.
        alpha_start: Initial value of alpha.
        warmup: Number of steps for the warmup phase. Often set equal to the number of training steps.

    Returns:
        A scheduler function that takes a step and returns the value of alpha.
    """

    def schedule(step: int | jax.Array) -> jax.Array:
        is_warmup = (step < warmup).astype(jnp.float32)
        a = step / float(warmup)
        return is_warmup * ((1.0 - a) * alpha_start + a * alpha) + alpha * (1.0 - is_warmup)

    return schedule


def beta3_scheduler(beta_end: float, beta_start: float = 0.0, warmup: int = 0) -> optax.Schedule:
    """
    Linear scheduler for the EMA parameter beta3 in AdEMAMix.

    Args:
        beta_end: Final value of beta3.
        beta_start: Initial value of beta3. Often set equal to beta1.
        warmup: Number of steps for the warmup phase. Often set equal to the number of training steps.

    Returns:
        A scheduler function that takes a step and returns the value of beta3.
    """

    def f(beta):
        return jnp.log(0.5) / jnp.log(beta) - 1

    def f_inv(t):
        return jnp.power(0.5, 1 / (t + 1))

    def schedule(step: int | jax.Array) -> jax.Array:
        is_warmup = (step < warmup).astype(jnp.float32)
        alpha = step / float(warmup)
        return is_warmup * f_inv((1.0 - alpha) * f(beta_start) + alpha * f(beta_end)) + beta_end * (1.0 - is_warmup)

    return schedule


class ScaleByAdemamixState(NamedTuple):
    """State for the AdEMAMix algorithm."""

    count: chex.Array
    """Step counter for the first momentum and adaptive learning rate."""
    count_m2: chex.Array
    """Step counter for the slower momentum."""
    m1: base.Updates
    """Fast EMA."""
    m2: base.Updates
    """Slow EMA."""
    nu: base.Updates
    """Second moment estimate."""


def ademamix(
    lr: float | optax.Schedule,
    b1: float = 0.9,
    b2: float = 0.999,
    b3: float = 0.9999,
    alpha: float = 5.0,
    b3_scheduler: optax.Schedule | None = None,
    alpha_scheduler: optax.Schedule | None = None,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    weight_decay: float = 0.0,
    mu_dtype: jnp.dtype | None = None,
    mask: Callable[[PyTree], PyTree] | None = None,
) -> optax.GradientTransformation:
    """AdEMAMix.

    Args:
        lr: A global scaling factor, either fixed or evolving along
            iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
        b1: Exponential decay rate to track the fast EMA.
        b2: Exponential decay rate to track the second moment of past gradients.
        b3: Exponential decay rate to track the slow EMA.
        alpha: Mixing coeficient use for the linear combination of the fast and slow EMAs.
        b3_scheduler: an optional scheduler function, given a timestep, returns the
            value of b3. Use `beta3_scheduler(b3,b1,T_b3)` to follow the AdEMAMix paper.
        alpha_scheduler: an optional scheduler function, given a timestep, returns the
            value of alpha. Use `alpha_scheduler(alpha,0,T_alpha)` to follow the
            AdEMAMix paper.
        eps: A small constant applied to denominator outside the square root
            (as in the Adam paper) to avoid dividing by zero when rescaling.
        eps_root: A small constant applied to denominator inside the square root (as
            in RMSProp), to avoid dividing by zero when rescaling. This is needed for
            instance when computing (meta-)gradients through Adam.
        mu_dtype: Optional `dtype` to be used for the first order accumulator; if
            `None` then the `dtype` is inferred from `params` and `updates`.
        weight_decay: Strength of the weight decay regularization. Note that this
            weight decay is multiplied with the learning rate. This is consistent
            with other frameworks such as PyTorch, but different from
            (Loshchilov et al., 2019) where the weight decay is only multiplied with
            the "schedule multiplier", but not the base learning rate.
        mask: A tree with same structure as (or a prefix of) the params PyTree,
            or a Callable that returns such a pytree given the params/updates.
            The leaves should be booleans, `True` for leaves/subtrees you want to
            apply the weight decay to, and `False` for those you want to skip. Note
            that the Adam gradient transformations are applied to all parameters.

    Returns:
        The corresponding `GradientTransformation`.
    """
    return combine.chain(
        scale_by_ademamix(b1, b2, b3, alpha, b3_scheduler, alpha_scheduler, eps, eps_root, mu_dtype),
        transform.add_decayed_weights(weight_decay, mask),
        transform.scale_by_learning_rate(lr),
    )


def scale_by_ademamix(
    b1: float,
    b2: float,
    b3: float,
    alpha: float,
    b3_scheduler: optax.Schedule | None,
    alpha_scheduler: optax.Schedule | None,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: jnp.dtype | None = None,
) -> optax.GradientTransformation:
    """
    Scales updates by the AdEMAMix algorithm.

    Args:
        b1: Exponential decay rate to track the fast EMA.
        b2: Exponential decay rate to track the second moment of past gradients.
        b3: Exponential decay rate to track the slow EMA.
        alpha: Mixing coeficient use for the linear combination of the fast and slow EMAs.
        b3_scheduler: an optional scheduler function, given a timestep, returns the
            value of b3. Use `beta3_scheduler(b3,b1,T_b3)` to follow the AdEMAMix paper.
        alpha_scheduler: an optional scheduler function, given a timestep, returns the
            value of alpha. Use `alpha_scheduler(alpha,0,T_alpha)` to follow the
            AdEMAMix paper.
        eps: A small constant applied to denominator outside the square root
            (as in the Adam paper) to avoid dividing by zero when rescaling.
        eps_root: A small constant applied to denominator inside the square root (as
            in RMSProp), to avoid dividing by zero when rescaling. This is needed for
            instance when computing (meta-)gradients through Adam.
        mu_dtype: Optional `dtype` to be used for the first order accumulator; if
            `None` then the `dtype` is inferred from `params` and `updates`.

    Returns:
        The corresponding `GradientTransformation`.
    """
    mu_dtype = utils.canonicalize_dtype(mu_dtype)

    def init_fn(params):
        m1 = tree_zeros_like(params, dtype=mu_dtype)  # fast EMA
        m2 = tree_zeros_like(params, dtype=mu_dtype)  # slow EMA
        nu = tree_zeros_like(params, dtype=mu_dtype)  # second moment estimate
        return ScaleByAdemamixState(
            count=jnp.zeros([], jnp.int32), count_m2=jnp.zeros([], jnp.int32), m1=m1, m2=m2, nu=nu
        )

    def update_fn(updates, state, params=None):
        del params
        c_b3 = b3_scheduler(state.count_m2) if b3_scheduler is not None else b3
        c_alpha = alpha_scheduler(state.count_m2) if alpha_scheduler is not None else alpha
        m1 = tree_update_moment(updates, state.m1, b1, 1)  # m1 = b1 * m1 + (1-b1) * updates
        m2 = tree_update_moment(updates, state.m2, c_b3, 1)
        nu = tree_update_moment_per_elem_norm(updates, state.nu, b2, 2)
        count_inc = numerics.safe_int32_increment(state.count)
        count_m2_inc = numerics.safe_int32_increment(state.count_m2)
        m1_hat = tree_bias_correction(m1, b1, count_inc)
        nu_hat = tree_bias_correction(nu, b2, count_inc)
        updates = jtu.tree_map(
            lambda m1_, m2_, v_: (m1_ + c_alpha * m2_) / (jnp.sqrt(v_ + eps_root) + eps), m1_hat, m2, nu_hat
        )
        m1 = tree_cast(m1, mu_dtype)
        m2 = tree_cast(m2, mu_dtype)
        return updates, ScaleByAdemamixState(count=count_inc, count_m2=count_m2_inc, m1=m1, m2=m2, nu=nu)

    return base.GradientTransformation(init_fn, update_fn)


def tree_cast(tree: PyTree, dtype: jnp.dtype) -> PyTree:
    """Cast tree to given dtype, skip if None."""
    if dtype is not None:
        return jtu.tree_map(lambda t: t.astype(dtype), tree)
    return tree


def tree_zeros_like(
    tree: PyTree,
    dtype: jnp.dtype | None = None,
):
    """Creates an all-zeros tree with the same structure.

    Args:
        tree: pytree.
        dtype: optional dtype to use for the tree of zeros.

    Returns:
        an all-zeros tree with the same structure as ``tree``.
    """
    return jtu.tree_map(lambda x: jnp.zeros_like(x, dtype=dtype), tree)


def tree_update_moment(updates: PyTree, moments: PyTree, decay: float | jax.Array, order: float | jax.Array) -> PyTree:
    """Compute the exponential moving average of the `order`-th moment.

    Args:
        updates: Gradients.
        moments: Moments.
        decay: Decay rate.
        order: Order of the moment.

    Returns:
        The updated moments.
    """
    return jtu.tree_map(lambda g, t: (1 - decay) * (g**order) + decay * t, updates, moments)


def tree_update_moment_per_elem_norm(
    updates: PyTree, moments: PyTree, decay: float | jax.Array, order: float | jax.Array
) -> PyTree:
    """Compute the EMA of the `order`-th moment of the element-wise norm.

    Args:
        updates: Gradients.
        moments: Moments.
        decay: Decay rate.
        order: Order of the moment.

    Returns:
        The updated moments.
    """

    def orderth_norm(g):
        if jnp.isrealobj(g):
            return g**order
        half_order = order / 2
        # JAX generates different HLO for int and float `order`
        if half_order.is_integer():
            half_order = int(half_order)
        return numerics.abs_sq(g) ** half_order

    return jtu.tree_map(lambda g, t: (1 - decay) * orderth_norm(g) + decay * t, updates, moments)


@functools.partial(jax.jit, inline=True)
def tree_bias_correction(moment: PyTree, decay: float | jax.Array, count: float | int | jax.Array) -> PyTree:
    """Performs bias correction. It becomes a no-op as count goes to infinity.

    Args:
        moment: Moments.
        decay: Decay rate.
        count: Step count.

    Returns:
        The bias-corrected moments.
    """
    # The conversion to the data type of the moment ensures that bfloat16 remains
    # bfloat16 in the optimizer state. This conversion has to be done after
    # `bias_correction_` is calculated as calculating `decay**count` in low
    # precision can result in it being rounded to 1 and subsequently a
    # "division by zero" error.
    bias_correction_ = 1 - decay**count

    # Perform division in the original precision.
    return jax.tree_util.tree_map(lambda t: t / bias_correction_.astype(t.dtype), moment)
