import jax
import numpy as np
from jax.sharding import Mesh
from config import ParallelismConfig

def create_device_mesh(config: ParallelismConfig) -> Mesh:
    devices = jax.devices()
    assert len(devices) == config.world_size, \
        f"Hardware provides {len(devices)} devices, config requires {config.world_size}."

    device_array = np.array(devices).reshape((config.dp, config.fsdp, config.tp))
    return Mesh(device_array, ("dp", "fsdp", "tp"))