import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# For kernel tests, we need a GPU.
os.environ["JAX_PLATFORMS"] = ""
os.environ.pop("CUDA_VISIBLE_DEVICES", None)

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
