import json
import logging
import os
from time import time

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig, OmegaConf

from xlstm_jax.define_hydra_schemas import register_configs
from xlstm_jax.distributed import set_XLA_flags
from xlstm_jax.train_init_fns import init_model_config, init_parallel, init_trainer, initialize_mesh
from xlstm_jax.trainer.llm.sampling import greedy_sampling
from xlstm_jax.trainer.llm.trainer import LLMTrainer

set_XLA_flags()  # Must be executed before any JAX operation.
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

# Register Hydra configs
register_configs()


def log_info(msg: str):
    if jax.process_index() == 0:
        LOGGER.info(msg)


def benchmark_generate(trainer: LLMTrainer, cfg: DictConfig):
    log_info("Starting benchmark.")
    num_tokens = 256
    local_batch_size = cfg.batch_size_per_device
    rng = jax.random.PRNGKey(42)

    generate_fn = trainer.get_generate_fn(
        max_length=num_tokens,
        eod_token_id=-1,  # Run full generation.
        token_sample_fn=greedy_sampling,
        gather_params_once=True,
    )
    trace_dir = (trainer.logger.log_path / "trace").absolute().as_posix()
    os.makedirs(trace_dir, exist_ok=True)
    benchmark_times = {
        "metadata": {
            "num_tokens": num_tokens,
            "local_batch_size": local_batch_size,
            "device_count": jax.device_count(),
            "device_type": jax.devices()[0].platform,
            "num_params": trainer.get_num_params(),
        },
        "times": {},
    }
    for batch_size in [2**i for i in range(np.log2(local_batch_size).astype(int) + 1)]:
        global_batch_size = jax.device_count() * batch_size
        log_info(f"Running benchmark for global batch size {batch_size}.")

        # Setup input data.
        prefix_tokens = jnp.ones((global_batch_size, num_tokens), dtype=jnp.int32)
        prefix_mask = jnp.zeros((global_batch_size, num_tokens), dtype=jnp.bool_)
        prefix_mask = prefix_mask.at[:, 0].set(True)

        # Run benchmark.
        all_times = []
        for i in range(10):
            log_info(f"Running iteration {i}...")
            if i == 1:
                log_info(f"Starting trace at {trace_dir}.")
                jax.profiler.start_trace(trace_dir)

            # Run generate function (full sequence is generated).
            start_time = time()
            with jax.profiler.StepTraceAnnotation("gen_step", step_num=i):
                tokens, is_valid = generate_fn(trainer.state, rng, prefix_tokens, prefix_mask)
            tokens.block_until_ready()
            is_valid.block_until_ready()
            end_time = time()

            if i == 2:
                log_info("Stopping trace.")
                jax.profiler.stop_trace()

            # Compute duration.
            duration = end_time - start_time
            if i > 2:
                all_times.append(duration)
            tokens_per_second = (num_tokens * global_batch_size) / duration
            seconds_per_step = duration / num_tokens
            log_info(
                f"Iteration {i} took {duration:.2f} seconds ({tokens_per_second:.2f} token/s, "
                f"{seconds_per_step:.4f} seconds/step)."
            )

        # Compute average times.
        avg_time = sum(all_times) / len(all_times)
        avg_tokens_per_second = (num_tokens * global_batch_size) / avg_time
        avg_seconds_per_token = 1 / avg_tokens_per_second
        avg_steps_per_second = num_tokens / avg_time
        avg_seconds_per_step = 1 / avg_steps_per_second
        log_info(f"Benchmark for global batch size {batch_size} complete.")
        log_info(f"Average time: {avg_time:.2f} seconds.")
        log_info(f"Average tokens per second: {avg_tokens_per_second:.2f}.")
        log_info(f"Average milliseconds per token: {avg_seconds_per_token * 1000:.2f}ms.")
        log_info(f"Average steps per second: {avg_steps_per_second:.2f}.")
        log_info(f"Average seconds per step: {avg_seconds_per_step:.4f}.")
        benchmark_times["times"][batch_size] = {
            "avg_time": avg_time,
            "avg_token_per_second": avg_tokens_per_second,
            "avg_seconds_per_token": avg_seconds_per_token,
            "avg_steps_per_second": avg_steps_per_second,
            "avg_seconds_per_step": avg_seconds_per_step,
            "times": all_times,
            "global_tokens_per_batch": num_tokens * global_batch_size,
            "local_tokens_per_batch": num_tokens * batch_size,
            "global_batch_size": global_batch_size,
            "local_batch_size": batch_size,
        }

        # Export every iteration to avoid losing data.
        if jax.process_index() == 0:
            log_info(f"Exporting benchmark times to {trainer.logger.log_path / 'benchmark_generate.json'}.")
            with open(trainer.logger.log_path / "benchmark_generate.json", "w") as f:
                json.dump(benchmark_times, f, indent=4)

    log_info("Benchmark complete.")


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def init_hydra(cfg: DictConfig):
    # Create mesh. Needs to be done before any JAX operation due to distribute initialize.
    parallel = init_parallel(cfg=cfg)

    # Initialize device mesh
    mesh = initialize_mesh(parallel_config=parallel)
    log_info("Mesh initialized.")

    log_info(f"Devices: {jax.devices()}.")

    # Compute global batch size.
    global_batch_size = cfg.batch_size_per_device * jax.device_count()
    cfg.global_batch_size = global_batch_size

    # Instatiate model config.
    model_config = init_model_config(cfg=cfg, parallel=parallel)

    # Instantiate trainer.
    trainer = init_trainer(cfg=cfg, data_iterator=None, model_config=model_config, mesh=mesh)

    # Save resolved config to output directory
    if jax.process_index() == 0:
        output_dir = cfg.logger.log_path
        with open(os.path.join(output_dir, "resolved_config.yaml"), "w") as f:
            OmegaConf.save(cfg, f, resolve=True)

    benchmark_generate(trainer, cfg)


if __name__ == "__main__":
    init_hydra()
