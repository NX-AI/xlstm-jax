import os
import time
from datetime import datetime
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P
from tqdm.auto import tqdm

from xlstm_jax.distributed.array_utils import fold_rng_over_axis
from xlstm_jax.distributed.common_types import Batch
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.block import mLSTMBlockConfig
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.layer import mLSTMLayerConfig
from xlstm_jax.models.xlstm_parallel.training import get_train_step_fn, init_xlstm, print_metrics
from xlstm_jax.models.xlstm_parallel.utils import ParallelConfig
from xlstm_jax.models.xlstm_parallel.xlstm_lm_model import xLSTMLMModel, xLSTMLMModelConfig


def init_mesh(
    data_axis_size: int = -1,
    fsdp_axis_size: int = 1,
    pipeline_axis_size: int = 1,
    model_axis_size: int = 1,
    data_axis_name: str = "dp",
    fsdp_axis_name: str = "fsdp",
    pipeline_axis_name: str = "pp",
    model_axis_name: str = "tp",
) -> Mesh:
    if "SLURM_STEP_NODELIST" in os.environ:
        jax.distributed.initialize()

    device_array = np.array(jax.devices()).reshape(data_axis_size, fsdp_axis_size, pipeline_axis_size, model_axis_size)
    mesh = Mesh(device_array, (data_axis_name, fsdp_axis_name, pipeline_axis_name, model_axis_name))
    print("Mesh", mesh)
    return mesh


def create_batch(
    batch_size: int,
    context_length: int,
    vocab_size: int,
    rng: jax.Array,
    mesh: Mesh,
    config: ParallelConfig,
):
    data_axis_name = (config.parallel.data_axis_name, config.parallel.fsdp_axis_name)

    def _data_gen(rng: jax.Array):
        rng = fold_rng_over_axis(rng, axis_name=data_axis_name)
        dp_size = jax.lax.psum(1, axis_name=data_axis_name)
        tokens = jax.random.randint(rng, shape=(batch_size // dp_size, context_length), minval=0, maxval=vocab_size)
        batch = Batch(
            inputs=jnp.pad(tokens[:, :-1], ((0, 0), (1, 0)), constant_values=0),
            labels=tokens,
        )
        return batch

    data_gen = shard_map(
        _data_gen,
        mesh,
        in_specs=P(),
        out_specs=P(data_axis_name),
        check_rep=False,
    )
    return data_gen(rng)


def benchmark_model(
    config: xLSTMLMModelConfig,
    data_axis_size: int = -1,
    fsdp_axis_size: int = 1,
    pipeline_axis_size: int = 1,
    model_axis_size: int = 1,
    seed: int = 42,
    gradient_accumulate_steps: int = 1,
    batch_size_per_device: int = 32,
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
        data_axis_size=data_axis_size,
        fsdp_axis_size=fsdp_axis_size,
        pipeline_axis_size=pipeline_axis_size,
        model_axis_size=model_axis_size,
        data_axis_name=config.parallel.data_axis_name,
        fsdp_axis_name=config.parallel.fsdp_axis_name,
        pipeline_axis_name=config.parallel.pipeline_axis_name,
        model_axis_name=config.parallel.model_axis_name,
    )

    rng = jax.random.PRNGKey(seed)
    model_rng, data_rng = jax.random.split(rng)
    data_rng = jax.random.fold_in(data_rng, jax.process_index())
    batch = create_batch(
        batch_size_per_device * jax.device_count(),
        config.context_length,
        config.vocab_size,
        data_rng,
        mesh,
        config,
    )
    if optimizer is None:
        optimizer = optax.adamw(learning_rate=1e-3)
    print(f"[Process {jax.process_index()} / {jax.process_count()}] Initializing model...")
    state = init_xlstm(config=config, mesh=mesh, rng=model_rng, input_array=batch.inputs, optimizer=optimizer)
    # save_checkpoint(state, log_dir)

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
    print(f"[Process {jax.process_index()} / {jax.process_count()}] Starting iteration...")
    for step_idx in tqdm(range(num_steps), desc="Running model") if jax.process_index() == 0 else range(num_steps):
        if jax.process_index() == 0:
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
    if jax.process_index() == 0:
        print_metrics(final_metrics, title="Final Metrics")

        iteration_times = np.array([iteration_times[log_skip_steps + log_num_steps + 4 :]])  # Remove tracing steps.

        print(" Timings ".center(30, "="))
        print(f"-> Average: {iteration_times.mean():4.3f}s")
        print(f"-> Median: {np.median(iteration_times):4.3f}s")
        print(f"-> Std: {iteration_times.std():4.3f}s")
        for q in [0.25, 0.5, 0.75, 0.95]:
            print(f"-> Quantile {q}: {np.quantile(iteration_times, q):4.3f}s")
        print("=" * 30)
