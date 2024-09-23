import numpy as np
import pytest

from xlstm_jax.trainer.optimizer.scheduler import SchedulerConfig, build_lr_scheduler


def _check_warmup(lr: float, warmup_steps: int, warmup_lr: float):
    """Check the warmup phase of the learning rate schedule."""
    if warmup_steps > 0:
        np.testing.assert_allclose(lr[0], 0.0, err_msg="Initial warmup LR is not 0.0.")
        np.testing.assert_allclose(
            lr[: warmup_steps + 1],
            np.linspace(0.0, warmup_lr, warmup_steps + 1),
            atol=1e-5,
            rtol=1e-5,
            err_msg="Warmup LR is not correct.",
        )
    np.testing.assert_allclose(
        lr[warmup_steps], warmup_lr, atol=1e-5, rtol=1e-5, err_msg="Final warmup LR is not correct."
    )


def _check_cooldown(lr: float, cooldown_steps: int, last_lr: float):
    """Check the cooldown phase of the learning rate schedule."""
    if cooldown_steps > 0:
        np.testing.assert_allclose(lr[-1], 0.0, err_msg="Final cooldown LR is not 0.0.")
        np.testing.assert_allclose(
            lr[-cooldown_steps - 1 :],
            np.linspace(last_lr, 0.0, cooldown_steps + 1),
            atol=1e-5,
            rtol=1e-5,
            err_msg="Cooldown LR is not correct.",
        )


@pytest.mark.parametrize("end_lr,end_lr_factor", [(0.03, None), (None, 0.1)])
@pytest.mark.parametrize("decay_steps", [100, 200, 300])
@pytest.mark.parametrize("warmup_steps", [0, 20, 30])
@pytest.mark.parametrize("cooldown_steps", [0, 20, 30])
def test_scheduler_exponential_decay(
    end_lr: float | None,
    end_lr_factor: float | None,
    decay_steps: int,
    warmup_steps: int,
    cooldown_steps: int,
):
    """Test the exponential decay scheduler."""
    config = SchedulerConfig(
        name="exponential_decay",
        lr=0.1,
        end_lr=end_lr,
        end_lr_factor=end_lr_factor,
        decay_steps=decay_steps,
        warmup_steps=warmup_steps,
        cooldown_steps=cooldown_steps,
    )
    lr_schedule = build_lr_scheduler(config)
    steps = np.arange(decay_steps + 1)
    lr = lr_schedule(steps)
    # Warmup.
    _check_warmup(lr, warmup_steps, 0.1)
    # Exponential decay.
    lr_log = np.log(lr[warmup_steps : decay_steps - cooldown_steps])
    lr_fac = lr_log[1:] - lr_log[:-1]  # The difference in log space should be constant.
    np.testing.assert_allclose(
        lr_fac, np.full_like(lr_fac, lr_fac[0]), atol=1e-5, rtol=1e-5, err_msg="Exponential decay is not constant."
    )
    # Final LR.
    last_lr = end_lr if end_lr is not None else 0.1 * end_lr_factor
    np.testing.assert_allclose(
        lr[-cooldown_steps - 1], last_lr, atol=1e-5, rtol=1e-5, err_msg="Final LR is not correct."
    )
    # Cooldown.
    _check_cooldown(lr, cooldown_steps, last_lr)
    # Verify outside of cooldown.
    steps = np.arange(decay_steps * 10)
    lr = lr_schedule(steps)
    assert np.all(lr >= 0.0), "Learning rate is negative outside of cooldown phase."


@pytest.mark.parametrize("end_lr,end_lr_factor", [(0.05, None), (None, 0.01)])
@pytest.mark.parametrize("decay_steps", [100, 200, 300])
@pytest.mark.parametrize("warmup_steps", [0, 10, 30])
@pytest.mark.parametrize("cooldown_steps", [0, 10, 20])
def test_cosine_decay(
    end_lr: float | None,
    end_lr_factor: float | None,
    decay_steps: int,
    warmup_steps: int,
    cooldown_steps: int,
):
    """Test the cosine decay scheduler."""
    config = SchedulerConfig(
        name="cosine_decay",
        lr=0.1,
        decay_steps=decay_steps,
        end_lr=end_lr,
        end_lr_factor=end_lr_factor,
        warmup_steps=warmup_steps,
        cooldown_steps=cooldown_steps,
    )
    lr_schedule = build_lr_scheduler(config)
    steps = np.arange(decay_steps + 1)
    lr = lr_schedule(steps)
    # Warmup.
    _check_warmup(lr, warmup_steps, 0.1)
    # Cosine decay.
    last_lr = end_lr if end_lr is not None else 0.1 * end_lr_factor
    lr_cosine = lr[warmup_steps : decay_steps - cooldown_steps]
    cosine_steps = lr_cosine.shape[0]
    # First half of cosine decay should be above linear.
    np.testing.assert_array_less(
        np.linspace(0.1, last_lr + (0.1 - last_lr) / 2, cosine_steps // 2)[1:],
        lr_cosine[1 : cosine_steps // 2],
        err_msg="First half of cosine decay is not above linear.",
    )
    if cosine_steps % 2 == 0:  # Middle point should be midpoint between last LR and 0.1.
        np.testing.assert_allclose(
            lr_cosine[cosine_steps // 2],
            last_lr + (0.1 - last_lr) / 2,
            atol=1e-5,
            rtol=1e-5,
            err_msg="Middle LR is not correct.",
        )
    # Second half of cosine decay should be below linear.
    np.testing.assert_array_less(
        lr_cosine[cosine_steps // 2 + 1 : -1],
        np.linspace(last_lr + (0.1 - last_lr) / 2, last_lr, cosine_steps // 2)[1:-1],
        err_msg="Second half of cosine decay is not below linear.",
    )
    np.testing.assert_allclose(
        lr[decay_steps - cooldown_steps], last_lr, atol=1e-5, rtol=1e-5, err_msg="Final LR is not correct."
    )
    # Cooldown.
    _check_cooldown(lr, cooldown_steps, last_lr)
    # Verify outside of cooldown.
    steps = np.arange(decay_steps * 10)
    lr = lr_schedule(steps)
    assert np.all(lr >= 0.0), "Learning rate is negative outside of cooldown phase."


@pytest.mark.parametrize("lr_peak", [0.1, 0.02, 0.005])
@pytest.mark.parametrize("decay_steps", [100, 200, 300])
@pytest.mark.parametrize("warmup_steps", [0, 10, 30])
@pytest.mark.parametrize("cooldown_steps", [0, 10, 20])
def test_constant(
    lr_peak: float,
    decay_steps: int,
    warmup_steps: int,
    cooldown_steps: int,
):
    """Test the constant scheduler."""
    config = SchedulerConfig(
        name="constant",
        lr=lr_peak,
        decay_steps=decay_steps,
        warmup_steps=warmup_steps,
        cooldown_steps=cooldown_steps,
    )
    lr_schedule = build_lr_scheduler(config)
    steps = np.arange(decay_steps + 1)
    lr = lr_schedule(steps)
    # If only constant, this schedule does not react correctly to array inputs and returns a float.
    if isinstance(lr, float):
        lr = np.full(decay_steps + 1, lr)
    # Warmup.
    _check_warmup(lr, warmup_steps, lr_peak)
    # Constant lr.
    np.testing.assert_allclose(
        lr[warmup_steps : decay_steps - cooldown_steps],
        lr_peak,
        atol=1e-5,
        rtol=1e-5,
        err_msg="Learning rate is not constant.",
    )
    # Cooldown.
    _check_cooldown(lr, cooldown_steps, lr_peak)
    # Verify outside of cooldown.
    steps = np.arange(decay_steps * 10)
    lr = lr_schedule(steps)
    assert np.all(lr >= 0.0), "Learning rate is negative outside of cooldown phase."


@pytest.mark.parametrize("end_lr,end_lr_factor", [(0.05, None), (None, 0.01)])
@pytest.mark.parametrize("decay_steps", [100, 200, 300])
@pytest.mark.parametrize("warmup_steps", [0, 10, 30])
@pytest.mark.parametrize("cooldown_steps", [0, 10, 20])
def test_linear(
    end_lr: float | None,
    end_lr_factor: float | None,
    decay_steps: int,
    warmup_steps: int,
    cooldown_steps: int,
):
    """Test the linear decay scheduler."""
    config = SchedulerConfig(
        name="linear",
        lr=0.1,
        decay_steps=decay_steps,
        end_lr=end_lr,
        end_lr_factor=end_lr_factor,
        warmup_steps=warmup_steps,
        cooldown_steps=cooldown_steps,
    )
    lr_schedule = build_lr_scheduler(config)
    steps = np.arange(decay_steps + 1)
    lr = lr_schedule(steps)
    # Warmup.
    _check_warmup(lr, warmup_steps, 0.1)
    # Linear decay.
    last_lr = end_lr if end_lr is not None else 0.1 * end_lr_factor
    lr_linear = lr[warmup_steps : decay_steps - cooldown_steps + 1]
    np.testing.assert_allclose(
        lr_linear,
        np.linspace(0.1, last_lr, lr_linear.shape[0]),
        atol=1e-5,
        rtol=1e-5,
        err_msg="Linear decay is not correct.",
    )
    np.testing.assert_allclose(lr_linear[-1], last_lr, atol=1e-5, rtol=1e-5, err_msg="Final LR is not correct.")
    # Cooldown.
    _check_cooldown(lr, cooldown_steps, last_lr)
    # Verify outside of cooldown.
    steps = np.arange(decay_steps * 10)
    lr = lr_schedule(steps)
    assert np.all(lr >= 0.0), "Learning rate is negative outside of cooldown phase."


@pytest.mark.parametrize("end_lr,end_lr_factor", [(0.05, 0.1), (1.0, 0.01), (100.0, 0.001), (0.1, 0.1)])
def test_end_lr_conflict(end_lr: float, end_lr_factor: float):
    """Test that the end_lr and end_lr_factor cannot be set differently at the same time."""
    config = SchedulerConfig(
        name="constant",
        lr=0.1,
        end_lr=end_lr,
        end_lr_factor=end_lr_factor,
    )
    with pytest.raises(AssertionError):
        build_lr_scheduler(config)


@pytest.mark.parametrize("decay_steps", [100, 200, 300])
@pytest.mark.parametrize("cooldown_steps", [0, 10, 20, 50])
def test_no_negative_lrs(decay_steps: int, cooldown_steps: int):
    """Test that the learning rate does not go negative, even outside of the cooldown phase."""
    config = SchedulerConfig(
        name="exponential_decay",
        lr=0.1,
        end_lr_factor=0.1,
        decay_steps=decay_steps,
        cooldown_steps=cooldown_steps,
    )
    lr_schedule = build_lr_scheduler(config)
    steps = np.arange(decay_steps + 1)
    lr = lr_schedule(steps)
    # Cooldown.
    _check_cooldown(lr, cooldown_steps, 0.01)
    # Verify outside of cooldown.
    steps = np.arange(decay_steps * 10)
    lr = lr_schedule(steps)
    assert np.all(lr >= 0.0), "Learning rate is negative outside of cooldown phase."
