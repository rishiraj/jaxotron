import re
import jax
import equinox as eqx
from jax.sharding import NamedSharding
from parallel.partitioning import LogicalRules

def _get_partition_spec(path_str: str):
    if re.match(r".*(q_proj|k_proj|v_proj|gate_proj|up_proj)\.weight", path_str):
        return LogicalRules.ColumnLinear 
    elif re.match(r".*(o_proj|down_proj)\.weight", path_str):
        return LogicalRules.RowLinear
    elif re.match(r".*embed_tokens\.weight|.*lm_head\.weight", path_str):
        return LogicalRules.Vocab
    elif re.match(r".*(layernorm|norm)\.weight", path_str):
        return LogicalRules.Norm
    return LogicalRules.Norm # Fallback for 1D biases/scalars

def shard_model(model: eqx.Module, mesh: jax.sharding.Mesh) -> eqx.Module:
    """Takes advantage of GSPMD compiler magic to shard leaf PyTree nodes across JAX mesh devices."""
    def _shard_leaf(path, leaf):
        if not isinstance(leaf, jax.Array):
            return leaf
        path_str = jax.tree_util.keystr(path)
        spec = _get_partition_spec(path_str)
        return jax.device_put(leaf, NamedSharding(mesh, spec))

    return jax.tree_util.tree_map_with_path(_shard_leaf, model)