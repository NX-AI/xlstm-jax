import logging
from collections.abc import Sequence
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict, freeze, unfreeze

LogMode = Literal["mean", "mean_nopostfix", "single", "max", "std", "single_noreduce", "single_noreduce_wcount"]
"""Mode for logging. Describes how to aggregate metrics over steps.

- `mean`: Mean of the metric.
- `mean_nopostfix`: Mean of the metric without adding a mean postfix to the key.
- `single`: Single value of the metric, i.e. only tracks the last value.
- `max`: Maximum value of the metric.
- `std`: Standard deviation of the metric.
- `single_noreduce`: Concatenate the metrics of multiple values.
- `single_noreduce_wcount`: Concatenate the metrics and counts of multiple values.
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
        if isinstance(val, jnp.ndarray) and jnp.issubdtype(val.dtype, jnp.floating):
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


def aggregate_metrics(aggregated_metrics: HostMetrics, batch_metrics: ImmutableMetrics) -> HostMetrics:
    """
    This function aggregates multiple metrics for the `single_noreduce` and `single_noreduce_wcount` case.
    For `single_noreduce` batches of single values are concatenated. The count is the number of samples.
    For `single_noreduce_wcount` batches of values are concatenated, as well as counts for each sample.
    This is needed for e.g. the loglikelihood per sequence.
    Concatenation happens in CPU memory after a conversion.

    The function returns batch_metrics in all other cases and moves them to CPU memory.

    Args:
        aggregated_metrics: Previously aggregated metrics to append to potentially.
        batch_metrics: Metrics from a batch

    Returns:
        Newly aggregated metrics.
    """
    aggregated_metrics_updated = {}
    host_metrics = jax.device_get(batch_metrics)
    for key in set(aggregated_metrics).union(set(host_metrics)):
        if key in batch_metrics:
            if key not in aggregated_metrics_updated:
                aggregated_metrics_updated[key] = {}
            for aggregation in batch_metrics[key]:
                if "single_noreduce_wcount" == aggregation:
                    aggregated_metrics_updated[key][aggregation] = (
                        {
                            "value": np.concatenate(
                                [aggregated_metrics[key][aggregation]["value"], host_metrics[key][aggregation]["value"]]
                            ),
                            "count": np.concatenate(
                                [aggregated_metrics[key][aggregation]["count"], host_metrics[key][aggregation]["count"]]
                            ),
                        }
                        if key in aggregated_metrics and aggregation in aggregated_metrics[key]
                        else {
                            "value": host_metrics[key][aggregation]["value"],
                            "count": host_metrics[key][aggregation]["count"],
                        }
                    )
                elif "single_noreduce" == aggregation:
                    aggregated_metrics_updated[key][aggregation] = (
                        {
                            "value": np.concatenate(
                                [aggregated_metrics[key][aggregation]["value"], host_metrics[key][aggregation]["value"]]
                            ),
                            "count": aggregated_metrics[key][aggregation]["count"]
                            + host_metrics[key][aggregation]["count"],
                        }
                        if key in aggregated_metrics and aggregation in aggregated_metrics[key]
                        else {
                            "value": host_metrics[key][aggregation]["value"],
                            "count": host_metrics[key][aggregation]["count"],
                        }
                    )
                else:
                    aggregated_metrics_updated[key][aggregation] = host_metrics[key][aggregation]
    return aggregated_metrics_updated


def _empty_val(value: Any) -> Any:
    if isinstance(value, int):
        return 0
    elif isinstance(value, str):
        return ""
    elif isinstance(value, float):
        return 0.0
    elif isinstance(value, np.ndarray):
        return np.zeros_like(value)
    elif isinstance(value, jax.Array):
        return jnp.zeros_like(value)
    else:
        return value


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
        global_metrics[key] = {mode: {"value": _empty_val(value), "count": 0} for mode in log_modes}
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
        elif log_mode == "single_noreduce":
            mode_dict["value"] = value
            mode_dict["count"] = count
        elif log_mode == "single_noreduce_wcount":
            mode_dict["value"] = value
            mode_dict["count"] = count
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
                if (isinstance(mode_metrics["count"], int) or mode_metrics["count"].size == 1) and mode_metrics[
                    "count"
                ] == 0:
                    LOGGER.warning(f"Metric {key} has count 0.")
                    metrics[out_key] = 0.0
                elif log_mode == "std":
                    mean = mode_metrics["value"] / mode_metrics["count"]
                    mean2 = mode_metrics["value2"] / mode_metrics["count"]
                    std = np.sqrt(np.clip(mean2 - mean**2, a_min=0.0, a_max=None))
                    metrics[out_key] = std
                elif log_mode == "single_noreduce":
                    metrics[out_key] = mode_metrics["value"]
                elif log_mode == "single_noreduce_wcount":
                    metrics[out_key] = (mode_metrics["value"], mode_metrics["count"])
                else:
                    metrics[out_key] = mode_metrics["value"] / mode_metrics["count"]

        else:
            metrics[key] = host_metrics[key]
    if reset_metrics:
        global_metrics = jax.tree.map(jnp.zeros_like, global_metrics)
    if not isinstance(global_metrics, FrozenDict):
        global_metrics = freeze(global_metrics)
    return global_metrics, metrics
