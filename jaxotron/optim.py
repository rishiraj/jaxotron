"""AdamW with warmup + cosine decay, gradient clipping, and decoupled weight decay."""
from __future__ import annotations

import jax
import optax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from jaxotron.config import TrainConfig
from jaxotron.parallel.rules import leaf_name


def _use_weight_decay(path, _) -> bool:
    """Exclude RMSNorm gains (attn_norm, mlp_norm, final_norm, q_norm, k_norm) and biases (bq, bk, bv)."""
    name = leaf_name(path)
    return not name.endswith("norm") and name not in ("bq", "bk", "bv")


def build_optimizer(tcfg: TrainConfig, params, total_steps: int | None = None) -> tuple[optax.GradientTransformation, optax.Schedule]:
    steps = total_steps if total_steps is not None else tcfg.train_steps
    warmup = min(tcfg.warmup_steps, max(steps - 1, 0))
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=tcfg.learning_rate,
        warmup_steps=warmup,
        decay_steps=max(steps, warmup + 1),
        end_value=tcfg.learning_rate * tcfg.min_lr_ratio,
    )
    decay_mask = jax.tree_util.tree_map_with_path(_use_weight_decay, params)
    optimizer = optax.chain(
        optax.clip_by_global_norm(tcfg.grad_clip),
        optax.adamw(schedule, b1=tcfg.beta1, b2=tcfg.beta2, eps=1e-8,
                    weight_decay=tcfg.weight_decay, mask=decay_mask),
    )
    return optimizer, schedule


def init_optimizer_state(optimizer: optax.GradientTransformation, params, param_shardings, mesh):
    """Create the optimizer state *already sharded* like the parameters (ZeRO-style).

    Adam moments follow the parameter shardings; scalars (step counts) are replicated.
    """
    abstract_state = jax.eval_shape(optimizer.init, params)
    replicated = NamedSharding(mesh, P())
    state_shardings = optax.tree_utils.tree_map_params(
        optimizer,
        lambda _, sharding: sharding,
        abstract_state,
        param_shardings,
        transform_non_params=lambda _: replicated,
    )
    return jax.jit(optimizer.init, out_shardings=state_shardings)(params), state_shardings