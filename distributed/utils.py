import os

def simulate_CPU_devices(device_count: int = 8):
    # Set XLA flags to simulate a CPU with a given number of devices
    flags = os.environ.get("XLA_FLAGS", "")
    flags += f" --xla_force_host_platform_device_count={device_count}"
    os.environ["XLA_FLAGS"] = flags
    # Disable CUDA to force XLA to use the CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


import os
os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_enable_triton_softmax_fusion=true '
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_disable_async_collectives=allreduce,allgather,reducescatter,collectivebroadcast,alltoall,collectivepermute '
    '--xla_gpu_enable_pipelined_all_gather=true '
    '--xla_gpu_enable_pipelined_reduce_scatter=true '
    '--xla_gpu_enable_pipelined_all_reduce=true '
    '--xla_gpu_enable_pipelined_collectives=false '
)

# if __name__ == "__main__":
#     import jax
#     print("Devices", jax.devices())