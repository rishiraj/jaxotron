"""Runtime parallelism context passed through the forward pass."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding

from jaxotron.parallel.rules import ACTIVATION_RULES, logical_to_spec


@dataclass(frozen=True)
class ParallelContext:
    mesh: Optional[Mesh] = None
    pp: int = 1
    pp_microbatches: int = 1
    dtype: Any = jnp.float32
    remat: str = "full"
    attention_impl: str = "reference"

    def shard(self, x: jax.Array, *logical_axes) -> jax.Array:
        """Annotate an activation with its logical axes. GSPMD does the rest."""
        if self.mesh is None or self.mesh.size == 1:
            return x
        spec = logical_to_spec(logical_axes, ACTIVATION_RULES)
        return jax.lax.with_sharding_constraint(x, NamedSharding(self.mesh, spec))
