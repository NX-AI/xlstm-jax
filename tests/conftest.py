import os

import pytest

# Select CPU or GPU devices.
if "CUDA_VISIBLE_DEVICES" not in os.environ or os.environ["CUDA_VISIBLE_DEVICES"] == "":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
elif "JAX_PLATFORMS" not in os.environ:
    # Don't override JAX_PLATFORMS if it has been set.

    # If JAX_PLATFORMS has not been set, set it to auto-congigure (i.e., to "").
    # Setting it this way will let jax automatically configure the platform, which
    # defaults to GPU when GPUs are available (which they should be when CUDA_VISIBLE_DEVICES
    # is set).
    os.environ["JAX_PLATFORMS"] = ""

# Select number of devices. On CPU, we simulate 8 devices.
if os.environ["JAX_PLATFORMS"] == "cpu":
    NUM_DEVICES = 8
    # The following line has to be imported here to avoid jax being initialized before
    # setting JAX_PLATFORMS to "cpu".
    from xlstm_jax.distributed.xla_utils import simulate_CPU_devices

    simulate_CPU_devices(NUM_DEVICES)
else:
    NUM_DEVICES = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    # Set XLA flags for deterministic operations on GPU. We do not use this flag for training runs
    # as it can slow down the training significantly.
    os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_gpu_deterministic_ops=true"

# Check if grain is available.
try:
    import grain  # noqa: F401 pylint: disable=unused-import

    GRAIN_AVAILABLE = True
except ImportError:
    GRAIN_AVAILABLE = False

# Check if triton is available.
try:
    import jax_triton

    jt_version = jax_triton.__version__

    # If we run on GPU environments with jax triton installed, but with JAX_PLATFORMS
    # set to CPU, we need to disable the triton tests.
    TRITON_AVAILABLE = os.environ.get("JAX_PLATFORMS", "") != "cpu"
except ImportError:
    TRITON_AVAILABLE = False


try:
    from transformers import AutoModelForCausalLM, xLSTMForCausalLM  # noqa: F401

    HUGGINGFACE_XLSTM_AVAILABLE = True
except ImportError:
    HUGGINGFACE_XLSTM_AVAILABLE = False


# Set W&B offline for tests.
os.environ["WANDB_MODE"] = "offline"

# Set XLSTM test data environment variable if not set.
if "XLSTM_JAX_TEST_DATA" not in os.environ:
    os.environ["XLSTM_JAX_TEST_DATA"] = "/nfs-gpu/xlstm/shared_tests"


# Share environment variables with pytest.
def pytest_configure():
    pytest.num_devices = NUM_DEVICES
    pytest.grain_available = GRAIN_AVAILABLE
    pytest.triton_available = TRITON_AVAILABLE
    pytest.huggingface_xlstm_available = HUGGINGFACE_XLSTM_AVAILABLE
