import os

import pytest

from xlstm_jax.distributed.xla_utils import simulate_CPU_devices

# Select CPU or GPU devices.
if "CUDA_VISIBLE_DEVICES" not in os.environ or os.environ["CUDA_VISIBLE_DEVICES"] == "":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
elif "JAX_PLATFORM_NAME" not in os.environ:
    # Don't override JAX_PLATFORMS if it has been set.

    # If JAX_PLATFORM_NAME has not been set, set it to auto-congigure (i.e., to "").
    # Setting it this way will let jax automatically configure the platform, which
    # defaults to GPU when GPUs are available (which they should be when CUDA_VISIBLE_DEVICES
    # is set).
    os.environ["JAX_PLATFORMS"] = ""

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

# Set XLSTM test data environment variable if not set.
if "XLSTM_JAX_TEST_DATA" not in os.environ:
    os.environ["XLSTM_JAX_TEST_DATA"] = "/nfs-gpu/xlstm/shared_tests"


# Share environment variables with pytest.
def pytest_configure():
    pytest.grain_available = GRAIN_AVAILABLE
    pytest.num_devices = NUM_DEVICES
