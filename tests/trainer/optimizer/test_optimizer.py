import operator
import re
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import linen as nn
from flax.training.train_state import TrainState

from xlstm_jax.trainer.base.param_utils import flatten_dict
from xlstm_jax.trainer.optimizer import OptimizerConfig, SchedulerConfig, build_optimizer

SCHEDULERS = [
    SchedulerConfig(name="constant", lr=0.1),
    SchedulerConfig(name="exponential_decay", lr=0.1, end_lr=0.01, decay_steps=100, warmup_steps=20, cooldown_steps=20),
    SchedulerConfig(name="cosine_decay", lr=0.1, end_lr=0.01, decay_steps=100, warmup_steps=20, cooldown_steps=0),
    SchedulerConfig(name="linear", lr=0.1, end_lr=0.01, decay_steps=100, warmup_steps=0, cooldown_steps=20),
]


class ToyModel(nn.Module):
    """Simple model for testing purposes."""

    features: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.features, name="in", bias_init=jax.nn.initializers.normal())(x)
        x = nn.LayerNorm(name="ln", bias_init=jax.nn.initializers.normal())(x)
        x = nn.relu(x)
        x = nn.Dense(1, name="out", bias_init=jax.nn.initializers.normal())(x)
        return x


def _init_model(optimizer: optax.GradientTransformation) -> TrainState:
    """Helper function to initialize the model state."""
    model = ToyModel(features=32)
    params = model.init(jax.random.PRNGKey(0), jnp.ones((8, 32)))
    return TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)


def _update_model(
    state: TrainState, loss_factor: float, return_loss: bool = False
) -> TrainState | tuple[TrainState, float]:
    """Helper function to update the model state with a simple loss function."""

    def loss_fn(params):
        inp = jax.random.normal(jax.random.PRNGKey(0), (64, 32))
        return loss_factor * jnp.mean((state.apply_fn(params, inp) - 1.0) ** 2)

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(state.params)
    new_state = state.apply_gradients(grads=grad)
    if return_loss:
        return new_state, loss
    else:
        return new_state


def _get_standard_adam(scheduler_config: SchedulerConfig, num_update_steps: int = 1, **kwargs) -> TrainState:
    """Helper function returning the state of the model after num_update_steps with Adam."""
    optimizer_config = OptimizerConfig(
        name="adam",
        scheduler=scheduler_config,
        beta1=kwargs.get("beta1", 0.9),
        beta2=kwargs.get("beta2", 0.999),
        eps=kwargs.get("eps", 1e-8),
        weight_decay=kwargs.get("weight_decay", 0.0),
        grad_clip_norm=kwargs.get("grad_clip_norm", None),
        grad_clip_value=kwargs.get("grad_clip_value", None),
        nesterov=kwargs.get("nesterov", False),
    )
    optimizer, _ = build_optimizer(optimizer_config)
    train_state = _init_model(optimizer)
    for _ in range(num_update_steps):
        train_state = _update_model(train_state, 1.0)
    return train_state


@pytest.mark.parametrize(
    "optimizer_name", ["adam", "adamw", "sgd", "nadam", "adamax", "radam", "nadamw", "adamaxw", "lamb"]
)
def test_optimizer_variants(optimizer_name: str):
    """Tests that all optimizers are running and improving loss."""
    optimizer_config = OptimizerConfig(name=optimizer_name, scheduler=SchedulerConfig(name="constant", lr=1e-2))
    optimizer, _ = build_optimizer(optimizer_config)
    assert optimizer is not None
    train_state = _init_model(optimizer)
    train_state, loss_first = _update_model(train_state, 1.0, return_loss=True)
    train_state, loss_second = _update_model(train_state, 1.0, return_loss=True)
    assert loss_second < loss_first, f"Loss did not decrease for {optimizer_name}."


@pytest.mark.parametrize("scheduler_config", SCHEDULERS)
def test_adam(scheduler_config: SchedulerConfig):
    """Tests that Adam is creating the opt state as expected."""
    optimizer_config = OptimizerConfig(name="adam", scheduler=scheduler_config)
    optimizer, _ = build_optimizer(optimizer_config)
    assert optimizer is not None
    train_state = _init_model(optimizer)
    assert train_state is not None
    assert train_state.opt_state is not None
    assert hasattr(train_state.opt_state[0][0], "count"), "Adam optimizer state does not have count."
    assert hasattr(
        train_state.opt_state[0][0], "mu"
    ), "Adam optimizer state does not have mu for the first-order momentum."
    assert hasattr(
        train_state.opt_state[0][0], "nu"
    ), "Adam optimizer state does not have nu for the second-order momentum."
    assert hasattr(train_state.opt_state[0][1], "count"), "Adam optimizer state does not have count for the scheduler."
    np.testing.assert_allclose(train_state.opt_state[0][0].count, 0, err_msg="Optimizer count is not initialized to 0.")
    for p in jax.tree.leaves(train_state.opt_state[0][0].mu):
        np.testing.assert_allclose(
            p, 0.0, atol=1e-5, rtol=1e-5, err_msg="First-order momentum is not initialized to 0.0."
        )
    for p in jax.tree.leaves(train_state.opt_state[0][0].nu):
        np.testing.assert_allclose(
            p, 0.0, atol=1e-5, rtol=1e-5, err_msg="Second-order momentum is not initialized to 0.0."
        )
    np.testing.assert_allclose(train_state.opt_state[0][1].count, 0, err_msg="Scheduler count is not initialized to 0.")
    train_state = _update_model(train_state, 1.0)
    np.testing.assert_allclose(train_state.opt_state[0][0].count, 1, err_msg="Optimizer count is not updated.")
    np.testing.assert_allclose(train_state.opt_state[0][1].count, 1, err_msg="Scheduler count is not updated.")
    for p in jax.tree.leaves(train_state.opt_state[0][0].mu):
        np.testing.assert_array_compare(operator.__ne__, p, 0.0, err_msg="First-order momentum is not updated.")
    for p in jax.tree.leaves(train_state.opt_state[0][0].nu):
        np.testing.assert_array_compare(operator.__ne__, p, 0.0, err_msg="Second-order momentum is not updated.")


@pytest.mark.parametrize("beta1,beta2,eps", [(0.5, 0.999, 1e-8), (0.9, 0.5, 1e-8), (0.9, 0.999, 1e-5)])
@pytest.mark.parametrize("optimizer_name", ["adam", "adamw"])
def test_adam_params(beta1: float, beta2: float, eps: float, optimizer_name: str):
    """Tests that the hyperparameters of Adam are all used and making in impact in the updates."""
    scheduler_config = SchedulerConfig(name="constant", lr=0.1)
    optimizer_config = OptimizerConfig(
        name=optimizer_name, scheduler=scheduler_config, beta1=beta1, beta2=beta2, eps=eps, weight_decay=0.0
    )
    optimizer, _ = build_optimizer(optimizer_config)
    train_state = _init_model(optimizer)
    for _ in range(2):
        train_state = _update_model(train_state, 1.0)
    base_state = _get_standard_adam(scheduler_config, num_update_steps=2)
    np.testing.assert_allclose(
        train_state.opt_state[0][0].count, base_state.opt_state[0][0].count, err_msg="Optimizer count is not the same."
    )
    np.testing.assert_allclose(
        train_state.opt_state[0][-1].count,
        base_state.opt_state[0][-1].count,
        err_msg="Scheduler count is not the same.",
    )
    for p1, p2 in zip(jax.tree.leaves(train_state.params), jax.tree.leaves(base_state.params)):
        np.testing.assert_array_compare(operator.__ne__, p1, p2, err_msg="Model parameters do not react to change.")


@pytest.mark.parametrize("scheduler_config", SCHEDULERS)
def test_adamw(scheduler_config: SchedulerConfig):
    """Tests that AdamW is creating the opt state as expected."""
    optimizer_config = OptimizerConfig(name="adamw", scheduler=scheduler_config, weight_decay=0.0)
    optimizer, _ = build_optimizer(optimizer_config)
    assert optimizer is not None
    train_state = _init_model(optimizer)
    assert train_state is not None
    assert train_state.opt_state is not None
    assert hasattr(train_state.opt_state[0][0], "count"), "AdamW optimizer state does not have count."
    assert hasattr(
        train_state.opt_state[0][0], "mu"
    ), "AdamW optimizer state does not have mu for the first-order momentum."
    assert hasattr(
        train_state.opt_state[0][0], "nu"
    ), "AdamW optimizer state does not have nu for the second-order momentum."
    assert hasattr(train_state.opt_state[0][2], "count"), "AdamW optimizer state does not have count for the scheduler."
    np.testing.assert_allclose(train_state.opt_state[0][0].count, 0, err_msg="Optimizer count is not initialized to 0.")
    for p in jax.tree.leaves(train_state.opt_state[0][0].mu):
        np.testing.assert_allclose(
            p, 0.0, atol=1e-5, rtol=1e-5, err_msg="First-order momentum is not initialized to 0.0."
        )
    for p in jax.tree.leaves(train_state.opt_state[0][0].nu):
        np.testing.assert_allclose(
            p, 0.0, atol=1e-5, rtol=1e-5, err_msg="Second-order momentum is not initialized to 0.0."
        )
    np.testing.assert_allclose(train_state.opt_state[0][2].count, 0, err_msg="Scheduler count is not initialized to 0.")
    train_state = _update_model(train_state, 1.0)
    np.testing.assert_allclose(train_state.opt_state[0][0].count, 1, err_msg="Optimizer count is not updated.")
    np.testing.assert_allclose(train_state.opt_state[0][2].count, 1, err_msg="Scheduler count is not updated.")
    base_adam_state = _get_standard_adam(scheduler_config, num_update_steps=1)
    np.testing.assert_allclose(
        train_state.opt_state[0][0].count,
        base_adam_state.opt_state[0][0].count,
        err_msg="Optimizer count is not the same.",
    )
    np.testing.assert_allclose(
        train_state.opt_state[0][2].count,
        base_adam_state.opt_state[0][1].count,
        err_msg="Scheduler count is not the same.",
    )
    for p1, p2 in zip(jax.tree.leaves(train_state.params), jax.tree.leaves(base_adam_state.params)):
        np.testing.assert_allclose(
            p1, p2, err_msg="Model parameters should be the same under adam and adamw without weight decay."
        )


@pytest.mark.parametrize("weight_decay", [0.1, 0.01])
def test_adamw_weight_decay(weight_decay: float):
    """Tests that weight decay is working in adamw."""
    scheduler_config = SchedulerConfig(name="constant", lr=0.1)
    optimizer_config = OptimizerConfig(name="adamw", scheduler=scheduler_config, weight_decay=weight_decay)
    optimizer, _ = build_optimizer(optimizer_config)
    train_state = _init_model(optimizer)
    for _ in range(2):
        train_state = _update_model(train_state, 1.0)
    base_state = _get_standard_adam(scheduler_config, num_update_steps=2, weight_decay=weight_decay)
    np.testing.assert_allclose(
        train_state.opt_state[0][0].count, base_state.opt_state[0][0].count, err_msg="Optimizer count is not the same."
    )
    np.testing.assert_allclose(
        train_state.opt_state[0][2].count, base_state.opt_state[0][1].count, err_msg="Scheduler count is not the same."
    )
    for p1, p2 in zip(jax.tree.leaves(train_state.params), jax.tree.leaves(base_state.params)):
        np.testing.assert_array_compare(
            operator.__ne__, p1, p2, err_msg="Model parameters are the same between adam and adamw under weight decay."
        )


@pytest.mark.parametrize("grad_clip_norm", [10.0, 1.0, 0.1, 0.01])
def test_grad_clip_norm(grad_clip_norm: float):
    """Tests that the gradient norm is clipped and stays below the threshold."""
    scheduler_config = SchedulerConfig(name="constant", lr=1.0)
    optimizer_config = OptimizerConfig(name="sgd", scheduler=scheduler_config, grad_clip_norm=grad_clip_norm)
    optimizer, _ = build_optimizer(optimizer_config)
    state = _init_model(optimizer)

    def loss_fn(params, loss_factor: float = 1.0):
        inp = jax.random.normal(jax.random.PRNGKey(0), (64, 32))
        return loss_factor * jnp.mean((state.apply_fn(params, inp) - 1.0) ** 2)

    # Make sure that the gradient norm is clipped and stays below the threshold.
    for loss_log_factor in np.linspace(-2.0, 2.0, 5):
        loss_factor = np.exp(loss_log_factor)
        grad = jax.grad(partial(loss_fn, loss_factor=loss_factor))(state.params)
        update, _ = optimizer.update(grad, state.opt_state)
        grad_norm = optax.global_norm(grad)
        update_norm = optax.global_norm(update)
        if grad_norm > grad_clip_norm:
            np.testing.assert_allclose(
                update_norm, grad_clip_norm, rtol=1e-5, atol=1e-5, err_msg="Gradient norm is not clipped."
            )
        else:
            np.testing.assert_allclose(
                update_norm, grad_norm, err_msg="Update norm should not be larger than the gradient."
            )


@pytest.mark.parametrize("grad_clip_value", [1.0, 0.1, 0.01, 0.001])
def test_grad_clip_value(grad_clip_value: float):
    """Tests that the gradient value is clipped and stays below the threshold."""
    scheduler_config = SchedulerConfig(name="constant", lr=1.0)
    optimizer_config = OptimizerConfig(name="sgd", scheduler=scheduler_config, grad_clip_value=grad_clip_value)
    optimizer, _ = build_optimizer(optimizer_config)
    state = _init_model(optimizer)

    def loss_fn(params, loss_factor: float = 1.0):
        inp = jax.random.normal(jax.random.PRNGKey(0), (64, 32))
        return loss_factor * jnp.mean((state.apply_fn(params, inp) - 1.0) ** 2)

    # Make sure that the gradient norm is clipped and stays below the threshold.
    for loss_log_factor in np.linspace(-2.0, 2.0, 5):
        loss_factor = np.exp(loss_log_factor)
        grad = jax.grad(partial(loss_fn, loss_factor=loss_factor))(state.params)
        update, _ = optimizer.update(grad, state.opt_state)
        for p in jax.tree.leaves(update):
            np.testing.assert_array_compare(
                operator.__le__, jnp.abs(p), grad_clip_value, err_msg="Gradient value is not clipped."
            )


@pytest.mark.parametrize("weight_decay", [1.0, 0.1])
@pytest.mark.parametrize("optimizer_name", ["adamw", "adam", "sgd"])
@pytest.mark.parametrize(
    "exclude,include",
    [(None, None), ([r".*bias", r".*\.scale"], None), (None, [r".*\.kernel"]), ([r".*\.ln\.*"], None)],
)
def test_weight_decay(weight_decay: float, optimizer_name: str, exclude: list[str] | None, include: list[str] | None):
    """Tests that weight decay is applied correctly to the parameters, including the mask."""
    scheduler_config = SchedulerConfig(name="constant", lr=1e-2)
    optimizer_config = OptimizerConfig(
        name=optimizer_name,
        scheduler=scheduler_config,
        weight_decay=weight_decay,
        weight_decay_exclude=exclude,
        weight_decay_include=include,
    )
    optimizer, _ = build_optimizer(optimizer_config)
    state = _init_model(optimizer)
    optimizer_wo_decay, _ = build_optimizer(
        OptimizerConfig(name=optimizer_name, scheduler=scheduler_config, weight_decay=0.0)
    )
    state_wo_decay = _init_model(optimizer_wo_decay)

    state = _update_model(state, 1.0)
    state_wo_decay = _update_model(state_wo_decay, 1.0)
    params = flatten_dict(state.params)
    params_wo_decay = flatten_dict(state_wo_decay.params)

    if exclude is None and include is None:
        include = [".*"]
    if exclude is not None:
        for key in params:
            if any(re.search(pattern, key) for pattern in exclude):
                np.testing.assert_allclose(
                    params[key],
                    params_wo_decay[key],
                    err_msg=f"Excluded parameter should not be affected by weight decay, but happened for {key}.",
                )
            else:
                np.testing.assert_array_compare(
                    operator.__ne__,
                    params[key],
                    params_wo_decay[key],
                    err_msg=f"Parameters not excluded should be affected by weight decay, but did not happen for {key}.",
                )
    if include is not None:
        for key in params:
            if any(re.search(pattern, key) for pattern in include):
                np.testing.assert_array_compare(
                    operator.__ne__,
                    params[key],
                    params_wo_decay[key],
                    err_msg=f"Included parameter should be affected by weight decay, but did not happen for {key}.",
                )
            else:
                np.testing.assert_allclose(
                    params[key],
                    params_wo_decay[key],
                    err_msg=f"Parameters not included should not be affected by weight decay, but happened for {key}.",
                )