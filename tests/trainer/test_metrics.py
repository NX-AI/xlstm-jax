import jax
import numpy as np
import pytest

from xlstm_jax.common_types import Metrics
from xlstm_jax.trainer.metrics import get_metrics, update_metrics


@pytest.mark.parametrize("seed", list(range(10)))
def test_update_and_get_metrics_single_metric(seed: int):
    """Tests updating the all modes for random sequence of numbers."""
    rng = np.random.default_rng(seed)
    num_elements = rng.integers(5, 100)
    values = rng.normal(size=num_elements, scale=10.0)
    count = rng.integers(1, 10, size=num_elements)
    global_metrics = {}
    for i in range(num_elements):
        step_metrics = {
            "a": {
                "value": values[i],
                "count": count[i],
                "log_modes": ["mean", "mean_nopostfix", "std", "max", "single"],
            }
        }
        global_metrics = update_metrics(global_metrics, step_metrics)
    _, host_metrics = get_metrics(global_metrics)

    # Mean.
    mean_key = "a_mean"
    assert mean_key in host_metrics, f"Key {mean_key} not found in host metrics."
    np.testing.assert_allclose(
        host_metrics[mean_key], values.sum() / count.sum(), err_msg=f"Failed for key {mean_key}."
    )
    mean_nopostfix_key = "a"
    assert mean_nopostfix_key in host_metrics, f"Key {mean_nopostfix_key} not found in host metrics."
    np.testing.assert_allclose(
        host_metrics[mean_nopostfix_key], host_metrics[mean_key], err_msg=f"Failed for key {mean_nopostfix_key}."
    )

    # Std.
    std_key = "a_std"
    assert std_key in host_metrics, f"Key {std_key} not found in host metrics."
    np.testing.assert_allclose(host_metrics[std_key], np.std(values / count), err_msg=f"Failed for key {std_key}.")

    # Max.
    max_key = "a_max"
    assert max_key in host_metrics, f"Key {max_key} not found in host metrics."
    np.testing.assert_allclose(host_metrics[max_key], np.max(values / count), err_msg=f"Failed for key {max_key}.")

    # Single.
    single_key = "a_single"
    assert single_key in host_metrics, f"Key {single_key} not found in host metrics."
    np.testing.assert_allclose(
        host_metrics[single_key], values[-1] / count[-1], err_msg=f"Failed for key {single_key}."
    )


@pytest.mark.parametrize("default_log_modes", [None, ["mean_nopostfix"], ["mean", "single"]])
def test_update_and_get_metrics_multi_metric(default_log_modes: list[str]):
    """Tests updating the global metrics."""

    @jax.jit
    def _update_fn1(global_metrics: Metrics) -> Metrics:
        step_metrics = {
            "a": 5.0,
            "b": {"value": 5.0},
            "c": {"value": 15.0, "count": 3},
            "d": {"value": 5.0, "log_modes": ["mean", "std"]},
            "e": {"value": 20.0, "count": 4, "log_modes": ["mean", "mean_nopostfix", "std", "max", "single"]},
        }
        return update_metrics(global_metrics, step_metrics, default_log_modes=default_log_modes)

    @jax.jit
    def _update_fn2(global_metrics: Metrics) -> Metrics:
        step_metrics = {
            "a": 5.0,
            "b": {"value": 10.0},
            "c": {"value": 50.0, "count": 10},
            "d": {"value": 25.0, "log_modes": ["mean", "std"]},
            "e": {"value": 30.0, "count": 2, "log_modes": ["mean", "mean_nopostfix", "std", "max", "single"]},
        }
        return update_metrics(global_metrics, step_metrics, default_log_modes=default_log_modes)

    global_metrics = _update_fn1(dict())
    global_metrics = _update_fn2(global_metrics)
    _, host_metrics = get_metrics(global_metrics)

    # Verify metrics with default log modes.
    if default_log_modes is None:
        default_log_modes = ["mean_nopostfix"]
    for log_mode in default_log_modes:
        a_key, b_key, c_key = "a", "b", "c"
        if log_mode != "mean_nopostfix":
            a_key += f"_{log_mode}"
            b_key += f"_{log_mode}"
            c_key += f"_{log_mode}"
        assert a_key in host_metrics, f"Key {a_key} not found in host metrics."
        assert b_key in host_metrics, f"Key {b_key} not found in host metrics."
        assert c_key in host_metrics, f"Key {c_key} not found in host metrics."
        if log_mode in ["mean", "mean_nopostfix"]:
            np.testing.assert_allclose(
                host_metrics[a_key], 5.0, err_msg=f"Failed for key {a_key} and log_mode {log_mode}."
            )
            np.testing.assert_allclose(
                host_metrics[b_key], 7.5, err_msg=f"Failed for key {b_key} and log_mode {log_mode}."
            )
            np.testing.assert_allclose(
                host_metrics[c_key], 5.0, err_msg=f"Failed for key {c_key} and log_mode {log_mode}."
            )
        elif log_mode == "single":
            np.testing.assert_allclose(
                host_metrics[a_key], 5.0, err_msg=f"Failed for key {a_key} and log_mode {log_mode}."
            )
            np.testing.assert_allclose(
                host_metrics[b_key], 10.0, err_msg=f"Failed for key {b_key} and log_mode {log_mode}."
            )
            np.testing.assert_allclose(
                host_metrics[c_key], 5.0, err_msg=f"Failed for key {c_key} and log_mode {log_mode}."
            )

    # Verify metrics with custom log modes.
    d_mean_key = "d_mean"
    assert d_mean_key in host_metrics, f"Key {d_mean_key} not found in host metrics."
    np.testing.assert_allclose(host_metrics[d_mean_key], 15.0, err_msg=f"Failed for key {d_mean_key}.")
    d_std_key = "d_std"
    assert d_std_key in host_metrics, f"Key {d_std_key} not found in host metrics."
    np.testing.assert_allclose(
        host_metrics[d_std_key], np.std(np.array([5.0, 25.0])), err_msg=f"Failed for key {d_std_key}."
    )

    e_mean_key = "e_mean"
    assert e_mean_key in host_metrics, f"Key {e_mean_key} not found in host metrics."
    np.testing.assert_allclose(
        host_metrics[e_mean_key],
        np.mean(np.array([5.0, 5.0, 5.0, 5.0, 15.0, 15.0])),
        err_msg=f"Failed for key {e_mean_key}.",
    )
    e_mean_nopostfix_key = "e"
    assert e_mean_nopostfix_key in host_metrics, f"Key {e_mean_nopostfix_key} not found in host metrics."
    np.testing.assert_allclose(
        host_metrics[e_mean_nopostfix_key], host_metrics[e_mean_key], err_msg=f"Failed for key {e_mean_nopostfix_key}."
    )
    e_std_key = "e_std"
    assert e_std_key in host_metrics, f"Key {e_std_key} not found in host metrics."
    np.testing.assert_allclose(
        host_metrics[e_std_key], np.std(np.array([5.0, 15.0])), err_msg=f"Failed for key {e_std_key}."
    )
    e_max_key = "e_max"
    assert e_max_key in host_metrics, f"Key {e_max_key} not found in host metrics."
    np.testing.assert_allclose(host_metrics[e_max_key], 15.0, err_msg=f"Failed for key {e_max_key}.")
    e_single_key = "e_single"
    assert e_single_key in host_metrics, f"Key {e_single_key} not found in host metrics."
    np.testing.assert_allclose(host_metrics[e_single_key], 15.0, err_msg=f"Failed for key {e_single_key}.")
