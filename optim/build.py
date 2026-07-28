import optax
import jax
import equinox as eqx

def scale_by_mup(lr_multiplier_tree: jax.tree_util.PyTreeDef) -> optax.GradientTransformation:
    def init_fn(params): return optax.EmptyState()
    def update_fn(updates, state, params=None):
        scaled_updates = jax.tree_util.tree_map(
            lambda u, m: u * m if m is not None else u, updates, lr_multiplier_tree
        )
        return scaled_updates, state
    return optax.GradientTransformation(init_fn, update_fn)

def build_optimizer(model, config, lr_multiplier_tree=None):
    train_cfg = config.training
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=train_cfg.learning_rate,
        warmup_steps=int(train_cfg.train_steps * 0.1), decay_steps=train_cfg.train_steps,
        end_value=train_cfg.learning_rate * 0.1,
    )

    # Exclude 1D weights (biases/LayerNorms) from Weight Decay natively
    wd_mask = jax.tree_util.tree_map_with_path(
        lambda p, x: x.ndim >= 2 and "norm" not in jax.tree_util.keystr(p) if eqx.is_inexact_array(x) else False,
        model
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=lr_schedule, 
            b1=0.9, 
            b2=0.95, 
            weight_decay=train_cfg.weight_decay, 
            mask=wd_mask
        )
    )

    # NOTE: muP scales learning rates per parameter, so it MUST be applied 
    # AFTER Adam normalizes the gradients!
    if lr_multiplier_tree is not None:
        optimizer = optax.chain(optimizer, scale_by_mup(lr_multiplier_tree))
        
    return optimizer