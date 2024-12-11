#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.
"""
Batch rampup schedule for grain IterDatasets.

This module provides a BatchRampUpIterDataset that allows for a batch rampup
schedule to be provided. The batch rampup schedule is a function that takes the
current batch step and returns the batch size. It can be used to gradually
increase the batch size over time.

The implementation is based on the standard BatchIterDataset from grain, with
the addition of a batch rampup schedule. See
grain._src.python.dataset.transformations.batch for more details.

NOTE: If the grain API for batching changes, this module may need to be updated.
"""

from __future__ import annotations

import logging
import pprint
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

import numpy as np
from grain._src.core import tree
from grain._src.python.dataset import dataset, stats as dataset_stats

T = TypeVar("T")
S = TypeVar("S")

LOGGER = logging.getLogger(__name__)


def _make_batch(values: Sequence[T]) -> T:
    """Returns a batch of values with a new batch dimension at the front."""

    if not values:
        raise ValueError("Cannot batch 0 values. Please file a bug.")

    try:
        return tree.map_structure(lambda *xs: np.stack(xs), *values)

    except ValueError as e:
        # NumPy error message doesn't include actual shapes and dtypes. Provide a
        # more helpful error message.
        raise ValueError(
            "Expected all input elements to have the same structure but got:\n"
            f"{pprint.pformat(tree.spec_like(values))}"
        ) from e


class _BatchRampUpDatasetIterator(dataset.DatasetIterator[T]):
    """Iterator that batches elements with a batch rampup schedule."""

    def __init__(
        self,
        parent: dataset.DatasetIterator[S],
        batch_rampup_schedule: Callable[[int], int],
        drop_remainder: bool,
        batch_fn: Callable[[Sequence[S]], T],
        stats: dataset_stats.Stats,
    ):
        """A Dataset iterator that batches elements with a batch rampup schedule.

        Args:
            parent: The parent IterDataset whose elements are batched.
            batch_rampup_schedule: A function that takes the current batch step and
                returns the batch size.
            drop_remainder: Whether to drop the last batch if it is smaller than
                batch_size.
            batch_fn: A function that takes a list of elements and returns a batch.
                Defaults to stacking the elements along a new batch dimension.
            stats: A Stats object for recording statistics.
        """
        super().__init__(stats)
        self._parent = parent
        self._batch_rampup_schedule = batch_rampup_schedule
        self._drop_remainder = drop_remainder
        self._batch_fn = batch_fn
        self._batch_step = 0

    def __next__(self) -> T:
        """Create and return the next batch of elements with size according to schedule."""
        values = []
        batch_size = self._batch_rampup_schedule(self._batch_step)
        if batch_size <= 0:
            raise ValueError(f"Batch size must be positive but got {batch_size} at step {self._batch_step}.")
        for _ in range(batch_size):
            try:
                values.append(next(self._parent))
            except StopIteration as e:
                if self._drop_remainder:
                    raise e
                break
        if not values:
            raise StopIteration
        self._batch_step += 1
        with self._stats.record_self_time():
            return self._stats.record_output_spec(self._batch_fn(values))

    def get_state(self) -> dict[str, Any]:
        """Return the state of the iterator."""
        return {"batch_step": self._batch_step, "parent": self._parent.get_state()}

    def set_state(self, state: dict[str, Any]):
        """Set the state of the iterator."""
        self._batch_step = state["batch_step"]
        self._parent.set_state(state["parent"])

    def __str__(self) -> str:
        return (
            f"BatchDatasetIterator(parent={self._parent},"
            f" batch_rampup_schedule={self._batch_rampup_schedule},"
            f" drop_remainder={self._drop_remainder})"
        )


class BatchRampUpIterDataset(dataset.IterDataset[T]):
    """Batch transformation with ramp up for IterDatasets."""

    def __init__(
        self,
        parent: dataset.IterDataset[S],
        batch_rampup_schedule: Callable[[int], int],
        drop_remainder: bool = False,
        batch_fn: Callable[[Sequence[S]], T] | None = None,
    ):
        """A IterDataset that batches elements with a batch rampup schedule.

        In comparison to the standard BatchIterDataset, this class allows for a
        batch rampup schedule to be provided. The batch rampup schedule is a
        function that takes the current batch step and returns the batch size.
        It can be used to gradually increase the batch size over time.

        Args:
            parent: The parent IterDataset whose elements are batched.
            batch_rampup_schedule: A function that takes the current batch step and
                returns the batch size.
            drop_remainder: Whether to drop the last batch if it is smaller than
                batch_size.
            batch_fn: A function that takes a list of elements and returns a batch.
                Defaults to stacking the elements along a new batch dimension.
        """
        super().__init__(parent)
        self._batch_rampup_schedule = batch_rampup_schedule
        self._drop_remainder = drop_remainder
        self._batch_fn = _make_batch if batch_fn is None else batch_fn

    def __iter__(self) -> _BatchRampUpDatasetIterator[T]:
        """Create a new iterator for the dataset."""
        parent_iter = self._parent.__iter__()
        return _BatchRampUpDatasetIterator(
            parent_iter,
            batch_rampup_schedule=self._batch_rampup_schedule,
            drop_remainder=self._drop_remainder,
            batch_fn=self._batch_fn,
            stats=self._stats,
        )

    def __str__(self) -> str:
        return (
            f"BatchRampUpIterDataset(batch_rampup_schedule={self._batch_rampup_schedule},"
            f" drop_remainder={self._drop_remainder})"
        )


def batch_dataset_with_rampup(
    parent: dataset.IterDataset[S],
    batch_size: int,
    drop_remainder: bool = False,
    batch_fn: Callable[[Sequence[S]], T] | None = None,
    schedule_type: str = "stepwise",
    boundaries_and_scales: dict[str, float] | None = None,
) -> BatchRampUpIterDataset[T] | dataset.IterDataset[T]:
    """Creates a BatchRampUpIterDataset from an IterDataset.

    Args:
        parent: The parent IterDataset whose elements are batched.
        batch_size: The initial batch size.
        drop_remainder: Whether to drop the last batch if it is smaller than
            batch_size.
        batch_fn: A function that takes a list of elements and returns a batch.
            Defaults to stacking the elements along a new batch dimension.
        schedule_type: The type of the batch rampup schedule. Supported types are
            "constant" and "stepwise".
        boundaries_and_scales: Used only for the "stepwise" schedule type. A dictionary
            mapping the boundaries b_i to non-negative scaling factors f_i. For any
            step count s, the schedule returns batch_size scaled by the product of
            factor f_i for the largest b_i such that b_i < s.

    Returns:
        A BatchRampUpIterDataset that batches elements. If no schedule is provided,
        falls back to the standard BatchIterDataset.
    """
    if schedule_type == "constant" or (schedule_type == "stepwise" and boundaries_and_scales is None):
        # If the schedule creates a constant batch size, use the standard BatchIterDataset.
        LOGGER.info(
            f"Batch rampup schedule returns constant batch size, falling back to standard batch with "
            f"batch size {batch_size}."
        )
        return parent.batch(batch_size, drop_remainder=drop_remainder, batch_fn=batch_fn)
    LOGGER.info(f"Creating {schedule_type} batch rampup schedule with boundaries and scales: {boundaries_and_scales}.")
    batch_rampup_schedule = create_batch_rampup_schedule(batch_size, schedule_type, boundaries_and_scales)
    return BatchRampUpIterDataset(parent, batch_rampup_schedule, drop_remainder, batch_fn)


def create_batch_rampup_schedule(
    batch_size: int, schedule_type: str, boundaries_and_scales: dict[str, float] | None = None
) -> Callable[[int], int]:
    """Creates a batch rampup schedule.

    Args:
        batch_size: The initial batch size.
        schedule_type: The type of the batch rampup schedule. Supported types are
            "constant" and "stepwise".
        boundaries_and_scales: A dictionary mapping the boundaries b_i to non-negative
            scaling factors f_i. For any step count s, the schedule returns batch_size
            scaled by the product of factor f_i for the largest b_i such that b_i < s.
            Only required for the "stepwise" schedule.

    Returns:
        A function that takes the current batch step and returns the batch size.
    """
    if schedule_type == "constant":
        return constant_rampup_schedule(batch_size)
    if schedule_type == "stepwise":
        if boundaries_and_scales is None:
            raise ValueError("boundaries_and_scales must be provided for the 'stepwise' schedule.")
        return stepwise_rampup_schedule(batch_size, boundaries_and_scales)
    raise ValueError(f"Unsupported schedule type: {schedule_type}.")


def constant_rampup_schedule(batch_size: int) -> Callable[[int], int]:
    """
    Returns a constant batch rampup schedule.

    Args:
        batch_size: The constant batch size.

    Returns:
        A function that takes the current batch step and returns the batch size.
    """
    return lambda _: batch_size


def stepwise_rampup_schedule(batch_size: int, boundaries_and_scales: dict[int, float]) -> Callable[[int], int]:
    """
    Returns a stepwise batch rampup schedule.

    Args:
        batch_size: The initial batch size on which the factors are applied.
        boundaries_and_scales: A dictionary mapping the boundaries b_i to non-negative scaling factors f_i.
            For any step count s, the schedule returns batch_size scaled by the product of factor f_i for the
            largest b_i such that b_i < s.

    Returns:
        A function that takes the current batch step and returns the batch size.
    """
    boundaries = sorted(boundaries_and_scales.keys(), reverse=True)  # Sort in decreasing order.

    def schedule(step: int) -> int:
        for boundary in boundaries:
            if step >= boundary:
                return int(batch_size * boundaries_and_scales[boundary])
        return batch_size

    return schedule
