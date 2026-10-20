"""Async, multi-host, topology-independent checkpointing with Orbax.

Parameters and optimizer state are saved as separate items, so inference can
restore only the weights. Because restoration takes the *target* shardings,
a checkpoint written on one mesh (say 4D on 16 GPUs) can be restored on any
other mesh (say a single GPU) and Orbax reshards automatically.
"""
from __future__ import annotations

import os
from typing import Optional

import jax
import orbax.checkpoint as ocp


def _resolve(directory: str) -> str:
    return directory if "://" in directory else os.path.abspath(directory)  # keep gs:// / s3:// as is


def _abstract(tree):
    return jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding), tree)


class Checkpointer:
    def __init__(self, directory: str, max_to_keep: int = 3):
        self.directory = _resolve(directory)
        self.manager = ocp.CheckpointManager(
            self.directory,
            options=ocp.CheckpointManagerOptions(max_to_keep=max_to_keep, create=True),
        )

    def latest_step(self) -> Optional[int]:
        return self.manager.latest_step()

    def save(self, step: int, params, opt_state) -> None:
        self.manager.save(step, args=ocp.args.Composite(
            params=ocp.args.StandardSave(params),
            opt_state=ocp.args.StandardSave(opt_state),
        ))

    def restore(self, params, opt_state, step: Optional[int] = None):
        """Returns (step, params, opt_state); unchanged inputs and step 0 if no checkpoint exists."""
        step = self.latest_step() if step is None else step
        if step is None:
            return 0, params, opt_state
        restored = self.manager.restore(step, args=ocp.args.Composite(
            params=ocp.args.StandardRestore(_abstract(params)),
            opt_state=ocp.args.StandardRestore(_abstract(opt_state)),
        ))
        return step, restored["params"], restored["opt_state"]

    def restore_params(self, abstract_params, step: Optional[int] = None):
        step = self.latest_step() if step is None else step
        if step is None:
            raise FileNotFoundError(f"No checkpoint found in {self.directory}")
        restored = self.manager.restore(step, args=ocp.args.Composite(
            params=ocp.args.StandardRestore(abstract_params)))
        return step, restored["params"]

    def wait(self) -> None:
        self.manager.wait_until_finished()

    def close(self) -> None:
        self.manager.wait_until_finished()
        self.manager.close()
