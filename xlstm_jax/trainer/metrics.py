from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict, freeze, unfreeze

# Immutable metrics for compilation.
ImmutableMetricElement = FrozenDict[str, jax.Array | int | float]
ImmutableMetrics = FrozenDict[str, ImmutableMetricElement]
# Mutable metrics for updating/editing.
MutableMetricElement = dict[str, jax.Array | int | float]
MutableMetrics = dict[str, MutableMetricElement]
# Metrics forwarded per step.
StepMetrics = dict[
    str,
    jax.Array | int | float | dict[str, jax.Array | int | float],
]
# Combined types.
MetricElement = ImmutableMetricElement | MutableMetricElement
Metrics = ImmutableMetrics | MutableMetrics
# Metrics on host (for logging).
HostMetricElement = float | int | np.ndarray
HostMetrics = dict[str, HostMetricElement]


def update_metrics(
    global_metrics: Metrics | None,
    step_metrics: StepMetrics,
    train: bool = True,
) -> ImmutableMetrics:
    """Update metrics with new values.

    Args:
        global_metrics: Global metrics to update. If None, a new dictionary is created.
        step_metrics: Metrics to update with.
        train: Whether the metrics are logged during training or evaluation. If training,
            we add a step-wise metric, otherwise we only add a mean over all steps.

    Returns:
        Updated global metrics.
    """
    if global_metrics is None:
        global_metrics = {}
    if isinstance(global_metrics, FrozenDict):
        global_metrics = unfreeze(global_metrics)
    for key in step_metrics:
        # Prepare input metric
        metric_in = step_metrics[key]
        if not isinstance(metric_in, dict):
            metric_in = {"value": metric_in}
        val = metric_in["value"]
        count = metric_in.get("count", 1)
        global_metrics = _update_single_metric(
            global_metrics,
            key,
            val,
            count,
            train,
        )
    global_metrics = freeze(global_metrics)
    return global_metrics


def _update_single_metric(
    global_metrics: MutableMetrics,
    key: str,
    value: Any,
    count: Any,
    train: bool,
) -> MutableMetrics:
    """Update a single metric.

    For training, we create a key both for tracking the mean over N steps and the last of N steps.
    This helps identify instabilities better while having a smoother curve. For evaluation, we only
    add a mean over all steps.

    Args:
        global_metrics: Global metrics to update.
        key: Key of the metric to update.
        value: Value of the metric to update.
        count: Count of the metric to update.
        train: Whether the metrics are logged during training or evaluation. If training,
            we add a step-wise metric, otherwise we only add a mean over all steps.

    Returns:
        Updated global metrics.
    """
    if key not in global_metrics:
        metrics_dict = {"value": 0.0, "count": 0}
    else:
        metrics_dict = global_metrics[key]
    metrics_dict["count"] += count
    metrics_dict["value"] += value
    if train:
        # Key for tracking the mean over N steps.
        global_metrics[key + "_mean"] = metrics_dict
        # Key for tracking the last of N steps.
        global_metrics[key + "_single"] = {"value": value, "count": count}
    else:
        global_metrics[key] = metrics_dict
    return global_metrics


def get_metrics(
    global_metrics: Metrics,
    reset_metrics: bool = True,
) -> tuple[ImmutableMetrics, HostMetrics]:
    """Calculates metrics to log from global metrics.

    Supports resetting the global metrics after logging. For example, if the global metrics
    are logged every epoch, the global metrics can be reset after obtaining the metrics to log
    such that the next epoch starts with empty metrics.

    Args:
        global_metrics: Global metrics to log.
        reset_metrics: Whether to reset the metrics after logging.

    Returns:
        The updated global metrics if reset_metrics is True, otherwise the original global metrics.
        Additionally, the metrics to log on the host device are returned.
    """
    if isinstance(global_metrics, FrozenDict) and reset_metrics:
        global_metrics = unfreeze(global_metrics)
    host_metrics = jax.device_get(global_metrics)
    metrics = {}
    for key in host_metrics:
        if isinstance(host_metrics[key], (dict, FrozenDict)):
            metrics[key] = host_metrics[key]["value"] / host_metrics[key]["count"]
        else:
            metrics[key] = host_metrics[key]
    if reset_metrics:
        global_metrics = jax.tree.map(jnp.zeros_like, global_metrics)
    if not isinstance(global_metrics, FrozenDict):
        global_metrics = freeze(global_metrics)
    return global_metrics, metrics
