import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# For kernel tests, we need a GPU.
os.environ["JAX_PLATFORMS"] = ""
os.environ.pop("CUDA_VISIBLE_DEVICES", None)

# Use cuda_compat to enable XLA parallel compilation. First, check if
# cuda-compat is installed in the current environment.  We assume that jax is
# installed in the same environment as cuda-compat so first, we get folder where
# jax is installed.
jax_folder = jax.__path__[0]
# Then, we get the path to the cuda-compat folder. cuda-compat is installed in
# the root folder of the environment, which resides 4 levels above the jax
# folder.
cuda_compat_folder = os.path.abspath(os.path.join(jax_folder, "../../../../cuda-compat"))
# Now we check whether the cuda-compat folder exists.
cuda_compat_installed = os.path.exists(cuda_compat_folder)

# If cuda-compat is installed, we add the path to the LD_LIBRARY_PATH environment variable.
if cuda_compat_installed:
    ld_library_path = os.environ.get("LD_LIBRARY_PATH")
    if ld_library_path is not None:
        os.environ["LD_LIBRARY_PATH"] = f"{ld_library_path}:{cuda_compat_folder}"
    else:
        os.environ["LD_LIBRARY_PATH"] = f"{cuda_compat_folder}"


try:
    import jax_triton

    jt_version = jax_triton.__version__

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


# Share environment variables with pytest.
def pytest_configure():
    pytest.triton_available = TRITON_AVAILABLE


@pytest.fixture
def default_qkvif() -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    B = 1
    NH = 2
    S = 128
    DH = 64
    rng = np.random.default_rng(2)
    q = rng.normal(size=(B, NH, S, DH)).astype(np.float32)
    k = rng.normal(size=(B, NH, S, DH)).astype(np.float32)
    v = rng.normal(size=(B, NH, S, DH)).astype(np.float32)
    igate_preact = rng.normal(size=(B, NH, S)).astype(np.float32)
    fgate_preact = rng.normal(size=(B, NH, S)).astype(np.float32) + 4.5

    q = jnp.array(q)
    k = jnp.array(k)
    v = jnp.array(v)
    igate_preact = jnp.array(igate_preact)
    fgate_preact = jnp.array(fgate_preact)

    return q, k, v, igate_preact, fgate_preact
