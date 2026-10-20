"""Logging, throughput/MFU accounting, compilation cache and sharding inspection."""
from __future__ import annotations

import math
import os
from typing import Optional

import jax

# Dense bf16 peak FLOP/s per device. Matched as substrings of device_kind, so
# more specific names come first. Unknown hardware: pass --peak_tflops.
_PEAK_BF16_FLOPS = [
    ("gb200", 2450e12), ("b200", 2250e12), ("h100 pcie", 756e12), ("h200", 989e12), ("h100", 989e12),
    ("h800", 989e12), ("gh200", 989e12), ("a100", 312e12), ("a800", 312e12), ("l40s", 362e12),
    ("l4", 121e12), ("a10g", 70e12), ("rtx 4090", 165e12), ("mi300x", 1307e12), ("mi250", 362e12),
    ("tpu v6 lite", 918e12), ("tpu v6e", 918e12), ("tpu v5 lite", 197e12), ("tpu v5e", 197e12),
    ("tpu v5p", 459e12), ("tpu v5", 459e12), ("tpu v4", 275e12), ("tpu v3", 123e12), ("tpu v2", 45e12),
]


def is_main_process() -> bool:
    return jax.process_index() == 0


def print0(*args, **kwargs) -> None:
    if is_main_process():
        print(*args, **kwargs, flush=True)


def peak_flops_per_device() -> Optional[float]:
    kind = jax.devices()[0].device_kind.lower()
    for name, flops in _PEAK_BF16_FLOPS:
        if name in kind:
            return flops
    return None


def enable_compilation_cache(directory: Optional[str]) -> None:
    """Persist compiled XLA programs so the second run of a demo starts almost instantly."""
    if not directory:
        return
    try:
        jax.config.update("jax_compilation_cache_dir", os.path.expanduser(directory))
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.5)
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    except Exception as e:  # the cache is an optimisation, never a reason to fail
        print0(f"Compilation cache disabled: {e}")


def human(n: float, unit: str = "") -> str:
    for suffix, scale in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= scale:
            return f"{n / scale:.2f}{suffix}{unit}"
    return f"{n:.2f}{unit}"


def sharding_table(params) -> str:
    """Human-readable table: parameter, global shape, PartitionSpec, MB per device."""
    rows = [("parameter", "global shape", "partition spec", "MB/device")]
    total = 0
    for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]:
        shard_shape = leaf.sharding.shard_shape(leaf.shape)
        nbytes = math.prod(shard_shape) * leaf.dtype.itemsize
        total += nbytes
        spec = getattr(leaf.sharding, "spec", "")
        rows.append((jax.tree_util.keystr(path), str(tuple(leaf.shape)), str(spec), f"{nbytes / 2**20:.2f}"))
    rows.append(("TOTAL (params only)", "", "", f"{total / 2**20:.2f}"))
    widths = [max(len(r[i]) for r in rows) for i in range(4)]
    lines = ["  ".join(c.ljust(w) for c, w in zip(r, widths)) for r in rows]
    lines.insert(1, "-" * len(lines[0]))
    return "\n".join(lines)


class MetricLogger:
    """Prints to stdout and optionally logs to Weights & Biases (if installed and configured)."""

    def __init__(self, project: Optional[str], run_name: str, config: dict):
        self.run = None
        if project and is_main_process():
            try:
                import wandb
                self.run = wandb.init(project=project, name=run_name, config=config)
            except ImportError:
                print0("wandb is not installed; logging to stdout only (pip install wandb).")

    def log(self, metrics: dict, step: int) -> None:
        if self.run is not None:
            self.run.log(metrics, step=step)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
