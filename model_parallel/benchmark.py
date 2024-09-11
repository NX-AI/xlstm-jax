import os
import time
from datetime import datetime
from typing import Any

from distributed.single_gpu import Batch, print_metrics
from distributed.tensor_parallel_transformer import split_array_over_mesh
from model_parallel.blocks.mlstm.block import mLSTMBlockConfig
from model_parallel.blocks.mlstm.layer import mLSTMLayerConfig
from model_parallel.training import get_train_step_fn, init_xlstm
from model_parallel.utils import ParallelConfig
from model_parallel.xlstm_lm_model import xLSTMLMModel, xLSTMLMModelConfig

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import torch
from flax import linen as nn
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from tqdm.auto import tqdm

from .checkpointing import save_checkpoint

PyTree = Any


def init_mesh(
    model_axis_size: int = 1,
    pipeline_axis_size: int = 1,
    model_axis_name: str = "tp",
    pipeline_axis_name: str = "pp",
    data_axis_name: str = "dp",
) -> Mesh:
    if "SLURM_STEP_NODELIST" in os.environ:
        jax.distributed.initialize(
            # coordinator_address=f"{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
            # local_device_ids={os.getenv("CUDA_VISIBLE_DEVICES")}
        )
    print(
        "Device count:",
        jax.device_count(),
        "Local device count:",
        jax.local_device_count(),
        "Process index:",
        jax.process_index(),
    )
    print("Devices:", jax.devices(), "Local devices:", jax.local_devices(), "Process Index:", jax.process_index())

    device_array = np.array(jax.devices()).reshape(-1, pipeline_axis_size, model_axis_size)
    mesh = Mesh(device_array, (data_axis_name, pipeline_axis_name, model_axis_name))
    return mesh


def benchmark_model(
    config: xLSTMLMModelConfig,
    model_axis_size: int = 1,
    pipeline_axis_size: int = 1,
    seed: int = 42,
    gradient_accumulate_steps: int = 1,
    batch_size: int = 32,
    optimizer: Any | None = None,
    log_dir: str | None = None,
    log_num_steps: int = 1,
    log_skip_steps: int = 5,
    num_steps: int = 100,
):
    if log_dir is None:
        log_dir = f"logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(log_dir, exist_ok=True)
    mesh = init_mesh(
        model_axis_size=model_axis_size,
        pipeline_axis_size=pipeline_axis_size,
        model_axis_name=config.parallel.model_axis_name,
        pipeline_axis_name=config.parallel.pipeline_axis_name,
        data_axis_name=config.parallel.data_axis_name,
    )

    rng = jax.random.PRNGKey(seed)
    model_rng, data_rng = jax.random.split(rng)
    input_array = jax.random.randint(
        data_rng, shape=(batch_size, config.context_length), minval=0, maxval=config.vocab_size
    )
    if optimizer is None:
        optimizer = optax.adamw(learning_rate=1e-3)
    print("Initializing model")
    state = init_xlstm(config=config, mesh=mesh, rng=model_rng, input_array=input_array, optimizer=optimizer)
    # save_checkpoint(state, log_dir)

    batch = Batch(
        inputs=jnp.pad(input_array[:, :-1], ((0, 0), (1, 0)), constant_values=0),
        labels=input_array,
    )
    train_step_fn, metrics = get_train_step_fn(
        state,
        batch=batch,
        mesh=mesh,
        config=config.parallel,
        gradient_accumulate_steps=gradient_accumulate_steps,
    )
    state, metrics = train_step_fn(
        state,
        metrics,
        batch,
    )
    for p in jax.tree.leaves(state.params):
        p.block_until_ready()
    iteration_times = []
    print("Starting iteration...")
    for step_idx in tqdm(range(num_steps), desc="Running model"):
        if step_idx == log_skip_steps:
            for p in jax.tree.leaves(state.params):
                p.block_until_ready()
            jax.profiler.start_trace(log_dir)
        if step_idx == log_skip_steps + log_num_steps:
            for p in jax.tree.leaves(state.params):
                p.block_until_ready()
            jax.profiler.stop_trace()
        start_time = time.time()
        with jax.profiler.StepTraceAnnotation("train_step", step_num=step_idx):
            state, metrics = train_step_fn(state, metrics, batch)
        end_time = time.time()
        iteration_times.append(end_time - start_time)
    for p in jax.tree.leaves(state.params):
        p.block_until_ready()
    for m in jax.tree.leaves(metrics):
        m.block_until_ready()
    final_metrics = jax.tree.map(jnp.zeros_like, metrics)
    state, final_metrics = train_step_fn(state, final_metrics, batch)
    print_metrics(final_metrics, title="Final Metrics")

    iteration_times = np.array([iteration_times[log_skip_steps + log_num_steps + 4 :]])  # Remove tracing steps.
    print(" Timings ".center(30, "="))
    print(f"-> Average: {iteration_times.mean():4.3f}s")
    print(f"-> Median: {np.median(iteration_times):4.3f}s")
    print(f"-> Std: {iteration_times.std():4.3f}s")
    for q in [0.25, 0.5, 0.75, 0.95]:
        print(f"-> Quantile {q}: {np.quantile(iteration_times, q):4.3f}s")
    print("=" * 30)
