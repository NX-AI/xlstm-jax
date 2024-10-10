import logging
from typing import Any, Literal

import jax.numpy as jnp
from flax import linen as nn

LOGGER = logging.getLogger(__name__)


def NormLayer(
    weight: bool = True,
    bias: bool = False,
    eps: float = 1e-5,
    dtype: jnp.dtype = jnp.float32,
    norm_type: Literal["layernorm", "rmsnorm"] = "layernorm",
    **kwargs,
) -> nn.Module:
    """
    Create a norm layer.

    Args:
        weight: Whether to use a learnable scaling weight or not.
        bias: Whether to use a learnable bias or not.
        eps: Epsilon value for numerical stability.
        dtype: Data type of the norm. Note that the statistic reductions in the norms are forced to be float32.
        norm_type: Type of the norm layer. Currently supported types are "layernorm" and "rmsnorm".
        **kwargs: Additional keyword arguments.

    Returns:
        Norm layer.
    """
    norm_class, norm_kwargs = resolve_norm(norm_type, weight, bias, eps, dtype, **kwargs)
    return norm_class(**norm_kwargs)


def MultiHeadNormLayer(
    weight: bool = True,
    bias: bool = False,
    eps: float = 1e-5,
    dtype: jnp.dtype = jnp.float32,
    axis: int = 1,
    norm_type: Literal["layernorm", "rmsnorm"] = "layernorm",
    **kwargs,
) -> nn.Module:
    """
    Create a multi-head norm layer.

    Effectively vmaps a norm layer over the specified axis.

    Args:
        weight: Whether to use a learnable scaling weight or not.
        bias: Whether to use a learnable bias or not.
        eps: Epsilon value for numerical stability.
        dtype: Data type of the norm. Note that the statistic reductions in the norms are forced to be float32.
        axis: Axis to vmap the norm layer over, i.e. the head axis. The normalization is always performed over
            the last axis.
        norm_type: Type of the norm layer. Currently supported types are "layernorm" and "rmsnorm".
        **kwargs: Additional keyword arguments.

    Returns:
        Multi-head norm layer.
    """
    norm_class, norm_kwargs = resolve_norm(norm_type, weight, bias, eps, dtype, **kwargs)
    return nn.vmap(
        norm_class,
        variable_axes={"params": 0},
        in_axes=axis,
        out_axes=axis,
        split_rngs={"params": True},
    )(**norm_kwargs)


def resolve_norm(
    norm_type: Literal["layernorm", "rmsnorm"],
    weight: bool = True,
    bias: bool = False,
    eps: float = 1e-5,
    dtype: jnp.dtype = jnp.float32,
    **kwargs,
) -> tuple[Any, dict[str, Any]]:
    """
    Resolve the norm layer based on the norm type.

    Args:
        norm_type: Type of the norm layer. Currently supported types are "layernorm" and "rmsnorm".
        weight: Whether to use a learnable scaling weight or not.
        bias: Whether to use a learnable bias or not.
        eps: Epsilon value for numerical stability.
        dtype: Data type of the norm. Note that the statistic reductions in the norms are forced to be float32.
        **kwargs: Additional keyword arguments.

    Returns:
        Tuple of the norm class and the keyword arguments.
    """
    if norm_type == "layernorm":
        return nn.LayerNorm, {"epsilon": eps, "use_bias": bias, "use_scale": weight, "dtype": dtype, **kwargs}
    elif norm_type == "rmsnorm":
        return nn.RMSNorm, {"epsilon": eps, "use_scale": weight, "dtype": dtype, **kwargs}
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")
