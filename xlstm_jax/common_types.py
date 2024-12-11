#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from collections.abc import Sequence
from typing import Any, Literal

import flax.linen as nn
import jax
import numpy as np
from flax.core import FrozenDict
from flax.training import train_state

# General types
PyTree = Any
Parameter = jax.Array | nn.Partitioned
PRNGKeyArray = jax.Array

# Training types

# Metrics types
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


class TrainState(train_state.TrainState):
    """TrainState with additional mutable variables and RNG."""

    # A simple extension of TrainState to also include mutable variables
    # like batch statistics. If a model has no mutable vars, it is None.
    mutable_variables: Any = None
    # RNG kept for init, dropout, etc.
    rng: PRNGKeyArray | None = None
