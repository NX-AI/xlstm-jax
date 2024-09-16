from collections.abc import Callable

import jax
import jax.numpy as jnp

from .common_types import Batch, Metrics, PRNGKeyArray, PyTree, TrainState


def accumulate_gradients_loop(
    state: TrainState,
    batch: Batch,
    rng: PRNGKeyArray,
    num_minibatches: int,
    loss_fn: Callable,
) -> tuple[PyTree, Metrics]:
    """Calculate gradients and metrics for a batch using gradient accumulation.

    Args:
        state: Current training state.
        batch: Full training batch.
        rng: Random number generator to use.
        num_minibatches: Number of minibatches to split the batch into. Equal to the number of gradient accumulation steps.
        loss_fn: Loss function to calculate gradients and metrics.

    Returns:
        Tuple with accumulated gradients and metrics over the minibatches.
    """
    batch_size = batch.inputs.shape[0]
    minibatch_size = batch_size // num_minibatches
    rngs = jax.random.split(rng, num_minibatches)
    # Define gradient function for single minibatch.
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    # Prepare loop variables.
    grads = None
    metrics = None
    for minibatch_idx in range(num_minibatches):
        with jax.named_scope(f"minibatch_{minibatch_idx}"):
            # Split the batch into minibatches.
            start = minibatch_idx * minibatch_size
            end = start + minibatch_size
            minibatch = jax.tree.map(lambda x: x[start:end], batch)
            # Calculate gradients and metrics for the minibatch.
            (_, step_metrics), step_grads = grad_fn(state.params, state.apply_fn, minibatch, rngs[minibatch_idx])
            # Accumulate gradients and metrics across minibatches.
            if grads is None:
                grads = step_grads
                metrics = step_metrics
            else:
                grads = jax.tree.map(jnp.add, grads, step_grads)
                metrics = jax.tree.map(jnp.add, metrics, step_metrics)
    # Average gradients over minibatches.
    grads = jax.tree.map(lambda g: g / num_minibatches, grads)
    return grads, metrics


def accumulate_gradients_scan(
    state: TrainState,
    batch: Batch,
    rng: PRNGKeyArray,
    num_minibatches: int,
    loss_fn: Callable,
) -> tuple[PyTree, Metrics]:
    """Calculate gradients and metrics for a batch using gradient accumulation.

    In this version, we use `jax.lax.scan` to loop over the minibatches. This is more efficient in terms of compilation time.

    Args:
        state: Current training state.
        batch: Full training batch.
        rng: Random number generator to use.
        num_minibatches: Number of minibatches to split the batch into. Equal to the number of gradient accumulation steps.
        loss_fn: Loss function to calculate gradients and metrics.

    Returns:
        Tuple with accumulated gradients and metrics over the minibatches.
    """
    batch_size = batch.inputs.shape[0]
    minibatch_size = batch_size // num_minibatches
    rngs = jax.random.split(rng, num_minibatches)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    def _minibatch_step(minibatch_idx: jax.Array | int) -> tuple[PyTree, Metrics]:
        """Determine gradients and metrics for a single minibatch."""
        minibatch = jax.tree.map(
            lambda x: jax.lax.dynamic_slice_in_dim(  # Slicing with variable index (jax.Array).
                x,
                start_index=minibatch_idx * minibatch_size,
                slice_size=minibatch_size,
                axis=0,
            ),
            batch,
        )
        (_, step_metrics), step_grads = grad_fn(state.params, state.apply_fn, minibatch, rngs[minibatch_idx])
        return step_grads, step_metrics

    def _scan_step(
        carry: tuple[PyTree, Metrics], minibatch_idx: jax.Array | int
    ) -> tuple[tuple[PyTree, Metrics], None]:
        """Scan step function for looping over minibatches."""
        step_grads, step_metrics = _minibatch_step(minibatch_idx)
        carry = jax.tree.map(jnp.add, carry, (step_grads, step_metrics))
        return carry, None

    # Determine initial shapes for gradients and metrics.
    grads_shapes, metrics_shape = jax.eval_shape(_minibatch_step, 0)
    grads = jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), grads_shapes)
    metrics = jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), metrics_shape)
    # Loop over minibatches to determine gradients and metrics.
    (grads, metrics), _ = jax.lax.scan(
        _scan_step,
        init=(grads, metrics),
        xs=jnp.arange(num_minibatches),
        length=num_minibatches,
    )
    # Average gradients over minibatches.
    grads = jax.tree.map(lambda g: g / num_minibatches, grads)
    return grads, metrics


def accumulate_gradients(
    state: TrainState,
    batch: Batch,
    rng: PRNGKeyArray,
    num_minibatches: int,
    loss_fn: Callable,
    use_scan: bool = False,
) -> tuple[PyTree, Metrics]:
    """Calculate gradients and metrics for a batch using gradient accumulation.

    This function supports scanning over the minibatches using `jax.lax.scan` or using a for loop.

    Args:
        state: Current training state.
        batch: Full training batch.
        rng: Random number generator to use.
        num_minibatches: Number of minibatches to split the batch into. Equal to the number of gradient accumulation steps.
        loss_fn: Loss function to calculate gradients and metrics.
        use_scan: Whether to use `jax.lax.scan` for looping over the minibatches.

    Returns:
        Tuple with accumulated gradients and metrics over the minibatches.
    """
    if use_scan:
        return accumulate_gradients_scan(
            state=state,
            batch=batch,
            rng=rng,
            num_minibatches=num_minibatches,
            loss_fn=loss_fn,
        )
    else:
        return accumulate_gradients_loop(
            state=state,
            batch=batch,
            rng=rng,
            num_minibatches=num_minibatches,
            loss_fn=loss_fn,
        )
