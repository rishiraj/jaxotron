"""Run the whole test-suite on 16 simulated CPU devices (no accelerators needed)."""
import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=16"