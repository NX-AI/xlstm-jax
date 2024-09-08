import os

def simulate_CPU_devices(device_count: int = 8):
    # Set XLA flags to simulate a CPU with a given number of devices
    flags = os.environ.get("XLA_FLAGS", "")
    flags += f" --xla_force_host_platform_device_count={device_count}"
    os.environ["XLA_FLAGS"] = flags
    # Disable CUDA to force XLA to use the CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ""