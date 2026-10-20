"""Device mesh creation.

The mesh is a 5D grid of devices. Axis order matters for performance: the
right-most axis is the "innermost" one, so it is placed on devices with the
fastest interconnect (NVLink / TPU ICI). Tensor parallelism communicates on
every layer, so it goes last; data parallelism communicates once per step,
so it goes first (and can span slower links between nodes).
"""
from __future__ import annotations

import numpy as np
import jax
from jax.sharding import Mesh

from jaxotron.config import ParallelConfig

MESH_AXES = ("dp", "pp", "fsdp", "cp", "tp")


def create_mesh(pcfg: ParallelConfig, devices=None) -> Mesh:
    all_devices = devices is None
    devices = list(jax.devices()) if devices is None else list(devices)
    shape = (pcfg.dp, pcfg.pp, pcfg.fsdp, pcfg.cp, pcfg.tp)
    if int(np.prod(shape)) != len(devices):
        raise ValueError(f"Mesh shape {dict(zip(MESH_AXES, shape))} needs {int(np.prod(shape))} devices, "
                         f"got {len(devices)}.")
    device_array = None
    if all_devices:
        try:  # topology-aware placement on TPU pods / multi-GPU nodes
            from jax.experimental import mesh_utils
            device_array = mesh_utils.create_device_mesh(shape, devices=devices)
        except Exception:
            device_array = None
    if device_array is None:
        device_array = np.asarray(devices, dtype=object).reshape(shape)
    return Mesh(device_array, MESH_AXES)
