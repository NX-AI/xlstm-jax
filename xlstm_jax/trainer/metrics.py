import logging
from collections.abc import Sequence
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict, freeze, unfreeze

LogMode = Literal["mean", "mean_nopostfix", "single", "max", "std"]
"""Mode for logging. Describes how to aggregate metrics over steps.

- `mean`: Mean of the metric.
- `mean_nopostfix`: Mean of the metric without adding a mean postfix to the key.
- `single`: Single value of the metric, i.e. only tracks the last value.
- `max`: Maximum value of the metric.
- `std`: Standard deviation of the metric.
"""
# TODO: not clear how to document module level type definitions
# TODO: Python 3.12 has the new `type` statement, check if this makes things easier
# Immutable metrics for compilation.
ImmutableMetricElement = FrozenDict[LogMode, FrozenDict[str, jax.Array | int | float]]
ImmutableMetrics = FrozenDict[str, ImmutableMetricElement]
# Mutable metrics for updating/editing.
MutableMetricElement = dict[LogMode, dict[str, jax.Array | int | float]]
MutableMetrics = dict[str, MutableMetricElement]
# Metrics forwarded per step.
StepMetricsElement = (
    jax.Array | int | float | dict[Literal["value", "count", "log_modes"], jax.Array | int | float | Sequence[LogMode]]
)
StepMetrics = dict[str, StepMetricsElement]
# Combined types.
MetricElement = ImmutableMetricElement | MutableMetricElement
Metrics = ImmutableMetrics | MutableMetrics
# Metrics on host (for logging).
HostMetricElement = float | int | np.ndarray
HostMetrics = dict[str, HostMetricElement]

LOGGER = logging.getLogger(__name__)


def update_metrics(
    global_metrics: Metrics | None,
    step_metrics: StepMetrics,
    default_log_modes: Sequence[LogMode] | None = None,
) -> ImmutableMetrics:
    """
    Update metrics with new values.

    Args:
        global_metrics: Global metrics to update. If `None`, a new dictionary is created.
        step_metrics: Metrics to update with.
        default_log_modes: The default log mode for the metrics. If `None`, only the mean will be logged. Otherwise, we
            log each of the modes specified. The metric key will be appended with the log mode.

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
        if isinstance(val, jnp.ndarray):
            val = val.astype(jnp.float32)
        count = metric_in.get("count", 1)
        log_modes = metric_in.get("log_modes", default_log_modes)
        _update_single_metric(
            global_metrics,
            key,
            val,
            count,
            log_modes,
        )
    global_metrics = freeze(global_metrics)
    return global_metrics


def _update_single_metric(
    global_metrics: MutableMetrics,
    key: str,
    value: Any,
    count: Any,
    log_modes: Sequence[LogMode] | None = None,
) -> MutableMetrics:
    """
    Update a single metric.

    Args:
        global_metrics: Global metrics to update.
        key: Key of the metric to update.
        value: Value of the metric to update.
        count: Count of the metric to update.
        log_modes: The log modes for the metric.
    """
    if log_modes is None:
        log_modes = ["mean_nopostfix"]

    # Get previous metrics from global dict.
    if key not in global_metrics:
        global_metrics[key] = {mode: {"value": 0.0, "count": 0} for mode in log_modes}
    metrics_dict = global_metrics[key]

    # Update each log mode.
    for log_mode in log_modes:
        mode_dict = metrics_dict[log_mode]
        if log_mode == "mean" or log_mode == "mean_nopostfix":
            # For mean, we store the sum of the values and the count.
            mode_dict["count"] += count
            mode_dict["value"] += value
        elif log_mode == "single":
            # For single, we store the last value.
            mode_dict["count"] = count
            mode_dict["value"] = value
        elif log_mode == "max":
            # For max, we store the maximum average value over all steps.
            mode_dict["count"] = 1
            mode_dict["value"] = jnp.maximum(mode_dict["value"], value / count)
        elif log_mode == "std":
            # For std, we store the sum of the values and the sum of the squared values.
            mode_dict["count"] += 1
            mode_dict["value"] += value / count
            mode_dict["value2"] = mode_dict.get("value2", 0.0) + (value / count) ** 2
        else:
            raise ValueError(f"Invalid log mode {log_mode}.")


def get_metrics(
    global_metrics: Metrics,
    reset_metrics: bool = True,
) -> tuple[ImmutableMetrics, HostMetrics]:
    """
    Calculates metrics to log from global metrics.

    Supports resetting the global metrics after logging. For example, if the global metrics are logged every epoch, the
    global metrics can be reset after obtaining the metrics to log such that the next epoch starts with empty metrics.

    Args:
        global_metrics: Global metrics to log.
        reset_metrics: Whether to reset the metrics after logging.

    Returns:
        The updated global metrics if reset_metrics is `True`, otherwise the original global metrics.
        Additionally, the metrics to log on the host device are returned.
    """
    if isinstance(global_metrics, FrozenDict) and reset_metrics:
        global_metrics = unfreeze(global_metrics)
    host_metrics = jax.device_get(global_metrics)
    metrics = {}
    for key in host_metrics:
        if isinstance(host_metrics[key], (dict, FrozenDict)):
            for log_mode in host_metrics[key]:
                out_key = key if log_mode == "mean_nopostfix" else f"{key}_{log_mode}"
                mode_metrics = host_metrics[key][log_mode]
                if mode_metrics["count"] == 0:
                    LOGGER.warning(f"Metric {key} has count 0.")
                    metrics[out_key] = 0.0
                elif log_mode == "std":
                    mean = mode_metrics["value"] / mode_metrics["count"]
                    mean2 = mode_metrics["value2"] / mode_metrics["count"]
                    std = np.sqrt(np.clip(mean2 - mean**2, a_min=0.0, a_max=None))
                    metrics[out_key] = std
                else:
                    metrics[out_key] = mode_metrics["value"] / mode_metrics["count"]
        else:
            metrics[key] = host_metrics[key]
    if reset_metrics:
        global_metrics = jax.tree.map(jnp.zeros_like, global_metrics)
    if not isinstance(global_metrics, FrozenDict):
        global_metrics = freeze(global_metrics)
    return global_metrics, metrics
