import os

import pytest

from xlstm_jax.distributed.xla_utils import simulate_CPU_devices

# Select CPU or GPU devices.
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
if "JAX_PLATFORMS" not in os.environ:
    os.environ["JAX_PLATFORMS"] = "cpu"

# Select number of devices. On CPU, we simulate 8 devices.
if os.environ["JAX_PLATFORMS"] == "cpu":
    NUM_DEVICES = 8
    simulate_CPU_devices(NUM_DEVICES)
else:
    NUM_DEVICES = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))

# Check if grain is available.
try:
    import grain  # noqa: F401

    GRAIN_AVAILABLE = True
except ImportError:
    GRAIN_AVAILABLE = False

# Set W&B offline for tests.
os.environ["WANDB_MODE"] = "offline"


# Share environment variables with pytest.
def pytest_configure():
    pytest.grain_available = GRAIN_AVAILABLE
    pytest.num_devices = NUM_DEVICES
