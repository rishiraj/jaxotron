"""Logical-axis sharding rules: the single place where parallelism is decided.

Every tensor dimension gets a *logical* name ("batch", "heads", "mlp", ...).
These rules map logical names to *mesh* axes ("dp", "fsdp", "pp", "cp", "tp").
XLA's GSPMD partitioner then inserts the collectives (all-gather,
reduce-scatter, all-reduce, collective-permute) automatically.

Change a rule here and the whole model is re-parallelised. No model code changes.
"""
from __future__ import annotations

import jax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

# How parameters are laid out across the mesh.
WEIGHT_RULES: dict[str, object] = {
    "layers": "pp",      # Pipeline parallel: each stage owns a contiguous block of layers
    "embed": "fsdp",     # FSDP / ZeRO-3: hidden dim of every weight is sharded across fsdp
    "vocab": "tp",       # Tensor parallel: vocab split (Megatron VocabParallelEmbedding)
    "heads": "tp",       # Tensor parallel: attention heads split (column-parallel q/k/v)
    "kv_heads": "tp",
    "mlp": "tp",         # Tensor parallel: MLP hidden split (column-parallel up/gate, row-parallel down)
}

# How activations are laid out across the mesh.
ACTIVATION_RULES: dict[str, object] = {
    "batch": ("dp", "fsdp"),  # Data parallel: batch split across dp and fsdp
    "seq": "cp",              # Context parallel: sequence split
    "kv_seq": None,           # Keys/values are all-gathered across cp (all-gather context parallelism)
    "stage": "pp",            # Pipeline parallel: the pipeline state buffer has one slot per stage
    "heads": "tp",
    "kv_heads": "tp",
    "mlp": "tp",
    "vocab": "tp",
    "act_embed": None,        # Residual stream is replicated over tp (Megatron style)
}

# Logical axes of every parameter, keyed by its field name in the model.
PARAM_LOGICAL_AXES: dict[str, tuple] = {
    "embed": ("vocab", "embed"),
    "lm_head": ("embed", "vocab"),
    "final_norm": (None,),
    "attn_norm": ("layers", None),
    "mlp_norm": ("layers", None),
    "wq": ("layers", "embed", "heads", None),
    "wk": ("layers", "embed", "kv_heads", None),
    "wv": ("layers", "embed", "kv_heads", None),
    "wo": ("layers", "heads", None, "embed"),
    "w_gate": ("layers", "embed", "mlp"),
    "w_up": ("layers", "embed", "mlp"),
    "w_down": ("layers", "mlp", "embed"),
}


def logical_to_spec(axes, rules) -> P:
    return P(*(rules.get(a) if a is not None else None for a in axes))


def leaf_name(path) -> str:
    """Field name of a pytree leaf, e.g. 'wq' for model.blocks.wq."""
    key = path[-1]
    return getattr(key, "name", getattr(key, "key", str(key)))


_leaf_name = leaf_name  # backwards compatibility


def param_spec(path, leaf) -> P:
    name = leaf_name(path)
    if name not in PARAM_LOGICAL_AXES:
        raise KeyError(f"No sharding rule for parameter {name!r}. Add it to PARAM_LOGICAL_AXES.")
    axes = PARAM_LOGICAL_AXES[name]
    if len(axes) != len(leaf.shape):
        raise ValueError(f"Parameter {name} has shape {leaf.shape} but logical axes {axes}.")
    return logical_to_spec(axes, WEIGHT_RULES)


def param_shardings(params, mesh: Mesh):
    """A pytree of NamedShardings matching the parameter pytree."""
    return jax.tree_util.tree_map_with_path(lambda path, x: NamedSharding(mesh, param_spec(path, x)), params)


def batch_sharding(mesh: Mesh) -> NamedSharding:
    """Sharding of the token batch [global_batch, seq_len + 1] produced by the data loader."""
    return NamedSharding(mesh, logical_to_spec(("batch", None), ACTIVATION_RULES))
