"""Runtime parallelism context passed through the forward pass."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding

from jaxotron.parallel.rules import ACTIVATION_RULES, ACTIVATION_RULES_SP, logical_to_spec


@dataclass(frozen=True)
class ParallelContext:
    mesh: Optional[Mesh] = None
    pp: int = 1
    pp_microbatches: int = 1
    sp: bool = False
    dtype: Any = jnp.float32
    remat: str = "full"
    attention_impl: str = "reference"

    def shard(self, x: jax.Array, *logical_axes) -> jax.Array:
        """Annotate an activation with its logical axes. GSPMD does the rest."""
        if self.mesh is None or self.mesh.size == 1:
            return x
        if len(logical_axes) != x.ndim:
            raise ValueError(f"Got {len(logical_axes)} logical axes {logical_axes} for an array of shape {x.shape}.")
        rules = ACTIVATION_RULES_SP if self.sp else ACTIVATION_RULES
        spec = logical_to_spec(logical_axes, rules)
        return jax.lax.with_sharding_constraint(x, NamedSharding(self.mesh, spec))