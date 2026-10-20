"""Device mesh creation.

The mesh is a 5D grid of devices. Axis order matters for performance: the
right-most axis is the "innermost" one, so it is placed on devices with the
fastest interconnect (NVLink / TPU ICI). Tensor parallelism communicates on
every layer, so it goes last; data parallelism communicates once per step,
so it goes first (and can span slower links between nodes).
"""
from __future__ import annotations

import jax
import numpy as np
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
    if all_devices and len(devices) > 1:
        try:  # topology-aware placement on TPU pods / multi-GPU nodes
            from jax.experimental import mesh_utils
            device_array = mesh_utils.create_device_mesh(shape, devices=devices)
        except Exception:
            device_array = None
    if device_array is None:
        device_array = np.asarray(devices, dtype=object).reshape(shape)
    # Explicit AxisType.Auto ensures forward compatibility with JAX 0.9+ where
    # helper functions like jax.make_mesh changed defaults toward AxisType.Explicit.
    axis_types = getattr(jax.sharding, "AxisType", None)
    if axis_types is not None and hasattr(axis_types, "Auto"):
        try:
            return Mesh(device_array, MESH_AXES, axis_types=(axis_types.Auto,) * len(MESH_AXES))
        except TypeError:
            pass
    return Mesh(device_array, MESH_AXES)


def mesh_table(mesh: Mesh) -> str:
    """Which mesh coordinates every device owns. Handy on a slide next to rules.py."""
    names = [n for n in mesh.axis_names if mesh.shape[n] > 1] or list(mesh.axis_names)
    header = f"{'device':<8}" + "".join(f"{n:>6}" for n in names) + "   kind"
    lines = [header, "-" * len(header)]
    for coords in np.ndindex(mesh.devices.shape):
        d = mesh.devices[coords]
        c = dict(zip(mesh.axis_names, coords))
        lines.append(f"{d.id:<8}" + "".join(f"{c[n]:>6}" for n in names) + f"   {d.device_kind}")
    return "\n".join(lines)