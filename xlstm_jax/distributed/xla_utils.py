import os


def simulate_CPU_devices(device_count: int = 8):
    """Simulate a CPU with a given number of devices.

    Args:
        device_count: The number of devices to simulate.
    """
    # Set XLA flags to simulate a CPU with a given number of devices
    flags = os.environ.get("XLA_FLAGS", "")
    flags += f" --xla_force_host_platform_device_count={device_count}"
    os.environ["XLA_FLAGS"] = flags
    # Disable CUDA to force XLA to use the CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def set_XLA_flags():
    """Set XLA flags for better performance.

    For performance flags, see https://jax.readthedocs.io/en/latest/gpu_performance_tips.html and
    https://github.com/NVIDIA/JAX-Toolbox/blob/main/rosetta/docs/GPU_performance.md.
    """
    if "XLA_PYTHON_CLIENT_MEM_FRACTION" not in os.environ:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"
    # Listing a lot of flags here, but only add a few made a noticable positive impact.
    os.environ["XLA_FLAGS"] = (
        "--xla_gpu_shard_autotuning=false "
        # '--xla_gpu_enable_triton_softmax_fusion=true '
        "--xla_gpu_triton_gemm_any=true "
        "--xla_gpu_enable_latency_hiding_scheduler=true "
        # '--xla_gpu_enable_async_collectives=true '
        "--xla_gpu_enable_highest_priority_async_stream=true "
        # '--xla_gpu_enable_while_loop_double_buffering=true '
        # '--xla_gpu_enable_pipelined_all_gather=true '
        # '--xla_gpu_enable_pipelined_reduce_scatter=true '
        # '--xla_gpu_enable_pipelined_all_reduce=true '
        # '--xla_gpu_enable_all_gather_combine_by_dim=false '
        # '--xla_gpu_enable_reduce_scatter_combine_by_dim=false '
        # '--xla_gpu_all_gather_combine_threshold_bytes=8589934592 '
        # '--xla_gpu_reduce_scatter_combine_threshold_bytes=8589934592 '
        # '--xla_gpu_all_reduce_combine_threshold_bytes=8589934592 '
        # '--xla_gpu_enable_pipelined_collectives=false '
        # '--xla_gpu_enable_pipelined_p2p=true '
        # '--xla_gpu_collective_permute_decomposer_threshold=1024 '
        # '--xla_gpu_lhs_enable_gpu_async_tracker=true '
        # '--xla_gpu_multi_streamed_windowed_einsum=true '
        # '--xla_gpu_threshold_for_windowed_einsum_mib=0 '
        # '--xla_gpu_enable_nccl_user_buffers=true '
    )
