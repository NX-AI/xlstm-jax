import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import tabulate
from flax import linen as nn

from xlstm_jax.distributed import set_XLA_flags
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend import (
    create_mlstm_backend,
    mLSTMBackend,
    mLSTMBackendNameAndKwargs,
)

LOGGER = logging.getLogger(__name__)
set_XLA_flags()


@dataclass
class ToyModelConfig:
    context_length: int


class BackendModule(nn.Module):
    context_length: int
    use_remat: bool
    backend_config: mLSTMBackendNameAndKwargs

    @nn.compact
    def __call__(self, *args, **kwargs):
        backend: mLSTMBackend = create_mlstm_backend(
            config=ToyModelConfig(context_length=self.context_length), nameandkwargs=self.backend_config
        )
        if backend.can_vmap_over_heads:
            backend = jax.vmap(
                backend,
                in_axes=(1, 1, 1, 1, 1),
                out_axes=1,
            )
        if self.use_remat:
            backend = nn.remat(backend, prevent_cse=False)
        return backend(*args, **kwargs)


def generate_input_data(
    batch_size: int,
    context_length: int,
    num_heads: int,
    head_dim: int,
    dtype: jnp.dtype,
    gate_dtype: jnp.dtype = jnp.float32,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Create input data for the mLSTM cell backend.

    Args:
        batch_size: Batch size.
        context_length: Sequence length.
        num_heads: Number of heads.
        head_dim: Dimension of the heads.
        dtype: Data type of the query, key, and value tensors.
        gate_dtype: Data type of the input and forget gate tensors.

    Returns:
        Tuple of query, key, value, input gate pre-activations, and forget gate pre-activations.
    """
    qkv_shape = (batch_size, num_heads, context_length, head_dim)
    gates_shape = (batch_size, num_heads, context_length, 1)
    rng = jax.random.PRNGKey(0)
    rng_q, rng_k, rng_v, rng_igate_preact, rng_fgate_preact = jax.random.split(rng, 5)
    q = jax.random.normal(rng_q, qkv_shape, dtype=dtype)
    k = jax.random.normal(rng_k, qkv_shape, dtype=dtype)
    v = jax.random.normal(rng_v, qkv_shape, dtype=dtype)
    igate_preact = jax.random.normal(rng_igate_preact, gates_shape, dtype=gate_dtype)
    fgate_preact = jax.random.normal(rng_fgate_preact, gates_shape, dtype=gate_dtype)
    return q, k, v, igate_preact, fgate_preact


def create_backend(name: str, context_length: int, use_remat: bool, **kwargs) -> BackendModule:
    backend_config = mLSTMBackendNameAndKwargs(name=name, kwargs=kwargs)
    backend = BackendModule(context_length=context_length, use_remat=use_remat, backend_config=backend_config)
    return backend


def create_step_function(backend: mLSTMBackend) -> callable:
    @jax.jit
    def step(
        q: jax.Array, k: jax.Array, v: jax.Array, igate_preact: jax.Array, fgate_preact: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        def loss_fn(q, k, v, i, f):
            out = backend.apply({"params": {}}, q, k, v, i, f)
            out = out.astype(jnp.float32)
            loss = out.sum()
            return loss

        grad_fn = jax.grad(loss_fn)
        return grad_fn(q, k, v, igate_preact, fgate_preact)

    return step


def run_benchmark(
    backend_name: str,
    batch_size: int,
    context_length: int,
    num_heads: int,
    head_dim: int,
    dtype: jnp.dtype,
    gate_dtype: jnp.dtype,
    use_remat: bool = False,
    num_steps: int = 10,
    num_repeats: int = 3,
    save_dir: Path | None = None,
    backend_kwargs: dict | None = None,
    backend_key: str | None = None,
) -> list[float]:
    dataset = [
        generate_input_data(batch_size, context_length, num_heads, head_dim, dtype, gate_dtype)
        for _ in range(num_steps)
    ]
    if backend_kwargs is None:
        backend_kwargs = {}
    backend = create_backend(backend_name, context_length, use_remat, **backend_kwargs)
    step = create_step_function(backend)
    # Warmup
    LOGGER.info("Warming up...")
    grads = []
    for data in dataset:
        grads.append(step(*data))
    grads = jax.tree.map(lambda x: x.block_until_ready(), grads)
    # Benchmark
    LOGGER.info("Running benchmark...")
    recorded_times = []
    for rep_idx in range(num_repeats):
        LOGGER.info(f"Repeat {rep_idx + 1}/{num_repeats}")
        grads = []
        start_time = time.time()
        for data in dataset:
            grads.append(step(*data))
        grads = jax.tree.map(lambda x: x.block_until_ready(), grads)
        end_time = time.time()
        recorded_times.append((end_time - start_time) / len(dataset))
    # Print statistics
    hparam_table = tabulate.tabulate(
        [
            ["Backend", backend_name],
            ["Backend Kwargs", backend_kwargs],
            ["Batch size", batch_size],
            ["Context length", context_length],
            ["Num heads", num_heads],
            ["Head dim", head_dim],
            ["dtype", dtype],
            ["gate_dtype", gate_dtype],
            ["Use remat", use_remat],
            ["Num steps", num_steps],
            ["Num repeats", num_repeats],
        ],
        headers=["Hyperparameter", "Value"],
    )
    time_table = tabulate.tabulate(
        [
            ["Mean", np.mean(recorded_times)],
            ["Median", np.median(recorded_times)],
            ["Std", np.std(recorded_times)],
            ["Min", np.min(recorded_times)],
            ["Max", np.max(recorded_times)],
        ],
        headers=["Statistic", "Value"],
    )
    print(hparam_table + "\n\n" + time_table)
    # Save results
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        if backend_key is None:
            backend_key = backend_name
        file_name = (
            f"{backend_key}_bs{batch_size}_ctx{context_length}_nh{num_heads}_hd{head_dim}_dt{dtype}_gdt{gate_dtype}"
            f"_remat{use_remat}.json"
        )
        file_path = save_dir / file_name
        hparams = {
            "backend": backend_name,
            "backend_kwargs": backend_kwargs,
            "backend_key": backend_key,
            "batch_size": batch_size,
            "context_length": context_length,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "dtype": str(dtype),
            "gate_dtype": str(gate_dtype),
            "use_remat": use_remat,
            "num_steps": num_steps,
            "num_repeats": num_repeats,
        }
        with open(file_path, "w") as f:
            json.dump({"config": hparams, "recorded_times": recorded_times}, f, indent=4)
    return recorded_times


def full_benchmark(
    vary_batch_size: bool = True,
    vary_context_length: bool = True,
    vary_head_dim: bool = True,
    vary_dtype: bool = True,
    save_dir: Path | None = None,
):
    backends = [
        ("parallel_stabilized", {}, "parallel_stabilized"),
        ("fwbw_stabilized", {}, "fwbw_stabilized"),
        ("triton_kernels", {"reduce_slicing": False}, "triton_kernels_no_reduce_slicing"),
        ("triton_kernels", {"reduce_slicing": True}, "triton_kernels_reduce_slicing"),
        ("triton_kernels", {"reduce_slicing": True, "chunk_size": 32}, "triton_kernels_chunk32"),
        ("triton_kernels", {"reduce_slicing": True, "chunk_size": 128}, "triton_kernels_chunk128"),
    ]
    if save_dir is None and "XLSTM_JAX_BENCHMARK_DATA" in os.environ:
        save_dir = Path(os.environ["XLSTM_JAX_BENCHMARK_DATA"]) / "mlstm_backends"
        save_dir.mkdir(parents=True, exist_ok=True)

    default_kwargs = {
        "batch_size": 8,
        "context_length": 2048,
        "num_heads": 4,
        "head_dim": 384,
        "dtype": "bfloat16",
        "gate_dtype": "float32",
        "use_remat": False,
        "num_steps": 5,
        "num_repeats": 5,
        "backend_kwargs": None,
        "save_dir": save_dir,
    }

    def _vary_arg(arg_name: str, values: list, results: dict):
        LOGGER.info(f"Varying {arg_name}...")
        results[arg_name] = {}
        for value in values:
            kwargs = default_kwargs.copy()
            kwargs[arg_name] = value
            results[arg_name][value] = {}
            for backend_name, backend_kwargs, backend_key in backends:
                kwargs["backend_name"] = backend_name
                kwargs["backend_kwargs"] = backend_kwargs
                kwargs["backend_key"] = backend_key
                results[arg_name][value][backend_key] = run_benchmark(**kwargs)

    results = {}
    if vary_batch_size:
        _vary_arg("batch_size", [1, 2, 4, 8, 16, 32], results)
    if vary_context_length:
        _vary_arg("context_length", [512, 1024, 2048, 4096, 8192], results)
    if vary_head_dim:
        _vary_arg("head_dim", [192, 384, 512, 768, 1024], results)
    if vary_dtype:
        _vary_arg("dtype", ["float32", "bfloat16"], results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run mLSTM backend benchmarks.")
    parser.add_argument("--backend", type=str, default="parallel_stabilized", help="Name of the mLSTM backend.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--context_length", type=int, default=2048, help="Context length.")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of heads.")
    parser.add_argument("--head_dim", type=int, default=384, help="Dimension of the heads.")
    parser.add_argument("--dtype", type=str, default="float32", help="Data type of the query, key, and value tensors.")
    parser.add_argument(
        "--gate_dtype", type=str, default="float32", help="Data type of the input and forget gate tensors."
    )
    parser.add_argument("--use_remat", action="store_true", help="Use remat.")
    parser.add_argument("--num_steps", type=int, default=10, help="Number of steps.")
    parser.add_argument("--num_repeats", type=int, default=3, help="Number of repeats.")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save results.")
    parser.add_argument("--backend_kwargs", type=json.loads, default=None, help="Backend kwargs.")
    parser.add_argument("--run_full_benchmark", action="store_true", help="Run full benchmark.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(name)s:%(lineno)d][%(levelname)s] - %(message)s")
    if args.run_full_benchmark:
        full_benchmark()
    else:
        dtype = getattr(jnp, args.dtype)
        gate_dtype = getattr(jnp, args.gate_dtype)
        save_dir = Path(args.save_dir) if args.save_dir is not None else None
        run_benchmark(
            args.backend,
            args.batch_size,
            args.context_length,
            args.num_heads,
            args.head_dim,
            dtype,
            gate_dtype,
            args.use_remat,
            args.num_steps,
            args.num_repeats,
            save_dir,
        )
