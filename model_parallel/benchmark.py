import os
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import optax
import flax
from flax import linen as nn
import numpy as np
import torch
import pytest
from typing import Any
from model_parallel.xlstm_lm_model import xLSTMLMModel
from model_parallel.xlstm_lm_model import xLSTMLMModelConfig
from model_parallel.blocks.mlstm.block import mLSTMBlockConfig
from model_parallel.blocks.mlstm.layer import mLSTMLayerConfig
from model_parallel.training import init_xlstm, get_train_step_fn
from model_parallel.utils import ParallelConfig
from distributed.tensor_parallel_transformer import split_array_over_mesh
from jax.sharding import PartitionSpec as P
from jax.sharding import NamedSharding
from distributed.single_gpu import Batch
from tqdm.auto import tqdm
from datetime import datetime
from distributed.single_gpu import print_metrics
import time

PyTree = Any


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
    device_array = np.array(jax.devices()).reshape(-1, pipeline_axis_size, model_axis_size)
    mesh = Mesh(
        device_array, (config.parallel.data_axis_name, config.parallel.pipeline_axis_name, config.parallel.model_axis_name)
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
    for step_idx in tqdm(range(num_steps), desc="Running model"):
        if step_idx == log_skip_steps:
            jax.profiler.start_trace(log_dir)
        if step_idx == log_skip_steps + log_num_steps:
            jax.profiler.stop_trace()
        start_time = time.time()
        with jax.profiler.StepTraceAnnotation("train_step", step_num=step_idx):
            state, metrics = train_step_fn(
                state, metrics, batch
            )
        end_time = time.time()
        iteration_times.append(end_time - start_time)
    for p in jax.tree.leaves(state.params):
        p.block_until_ready()
    for m in jax.tree.leaves(metrics):
        m.block_until_ready()
    final_metrics = jax.tree.map(jnp.zeros_like, metrics)
    state, final_metrics = train_step_fn(
        state, final_metrics, batch
    )
    print_metrics(final_metrics, title="Final Metrics")

    iteration_times = np.array([iteration_times[log_skip_steps + log_num_steps + 4:]])  # Remove tracing steps.
    print(' Timings '.center(30, '='))
    print(f'-> Average: {iteration_times.mean():4.3f}s')
    print(f'-> Median: {np.median(iteration_times):4.3f}s')
    print(f'-> Std: {iteration_times.std():4.3f}s')
    for q in [0.25, 0.5, 0.75, 0.95]:
        print(f'-> Quantile {q}: {np.quantile(iteration_times, q):4.3f}s')
    print('=' * 30)
