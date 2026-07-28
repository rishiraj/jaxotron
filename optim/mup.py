import jax
import equinox as eqx
import hashlib
from typing import Tuple
from config import ModelConfig

def get_deterministic_hash(s: str) -> int:
    # Keeps it within 32-bit uint bounds for JAX random folding
    return int(hashlib.md5(s.encode('utf-8')).hexdigest()[:8], 16)

def get_fan_in_out(weight: jax.Array) -> Tuple[int, int]:
    # Standard eqx.nn.Linear stores weights as [out_features, in_features]
    if weight.ndim == 2:
        return weight.shape[1], weight.shape[0]
    return 1, 1

def compute_spectral_std(std: float, fan_in: int, fan_out: int) -> float:
    """Spectral parameterization from Greg Yang's muTransfer paper."""
    import math
    return (std / math.sqrt(fan_in)) * min(1.0, math.sqrt(fan_out / fan_in))

def apply_mup_initialization(model: eqx.Module, config: ModelConfig, key: jax.Array) -> eqx.Module:
    """Re-initializes matrices using Spectral muP rules."""
    def _init_leaf(path, leaf):
        if not isinstance(leaf, jax.Array):
            return leaf
            
        path_str = jax.tree_util.keystr(path)
        
        if "weight" in path_str and leaf.ndim == 2:
            fan_in, fan_out = get_fan_in_out(leaf)
            if "embed_tokens" in path_str or "lm_head" in path_str:
                std = 1.0 
            else:
                std = compute_spectral_std(1.0, fan_in, fan_out)
            key_leaf = jax.random.fold_in(key, get_deterministic_hash(path_str))
            return jax.random.normal(key_leaf, leaf.shape) * std
                
        return leaf

    return jax.tree_util.tree_map_with_path(_init_leaf, model)

def get_mup_lr_multiplier_tree(model: eqx.Module) -> eqx.Module:
    """
    Optax expects a PyTree matching parameter struct containing multipliers.
    """
    def _map_lr(path, leaf):
        if not isinstance(leaf, jax.Array):
            return None
            
        path_str = jax.tree_util.keystr(path)
        if "weight" in path_str and leaf.ndim == 2:
            fan_in, fan_out = get_fan_in_out(leaf)
            if "embed_tokens" in path_str or "lm_head" in path_str:
                return 1.0
            else:
                return fan_out / fan_in
        return 1.0

    return jax.tree_util.tree_map_with_path(_map_lr, model)