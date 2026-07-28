import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from jax.sharding import PartitionSpec as P

def compute_loss(model, input_ids, labels):
    logits = model(input_ids)
    loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(
        logits=logits.reshape(-1, logits.shape[-1]), labels=labels.reshape(-1)
    ))
    return loss, {"loss": loss, "perplexity": jnp.exp(loss)}

def make_train_step(optimizer):
    # Equinox handles functional differentiation flawlessly
    loss_and_grad_fn = eqx.filter_value_and_grad(compute_loss, has_aux=True)

    @eqx.filter_jit
    def train_step(model, opt_state, global_batch, num_microbatches):
        input_ids = global_batch["input_ids"]
        labels = global_batch["labels"]
        
        GBS, seq_len = input_ids.shape
        
        # Reshape to [num_microbatches, micro_batch_size_per_replica * dp * fsdp, seq_len]
        input_ids = input_ids.reshape(num_microbatches, -1, seq_len)
        labels = labels.reshape(num_microbatches, -1, seq_len)

        # Apply sharding constraint so XLA explicitly knows how to distribute the Data
        # Parallel shards *after* the reshape, ensuring the micro_step iteration slices cleanly
        input_ids = jax.lax.with_sharding_constraint(input_ids, P(None, ("dp", "fsdp"), None))
        labels = jax.lax.with_sharding_constraint(labels, P(None, ("dp", "fsdp"), None))

        accum_grads = jax.tree_util.tree_map(
            lambda x: jnp.zeros_like(x) if eqx.is_inexact_array(x) else None, 
            model
        )
        
        def micro_step(carry, micro_inputs):
            current_accum_grads = carry
            mb_input_ids, mb_labels = micro_inputs
            
            (loss, metrics), grads = loss_and_grad_fn(model, mb_input_ids, mb_labels)
            
            # Accumulate handling non-array nodes securely
            new_accum_grads = jax.tree_util.tree_map(
                lambda a, g: a + (g / num_microbatches) if g is not None else None, 
                current_accum_grads, grads
            )
            return new_accum_grads, metrics

        final_accum_grads, all_metrics = jax.lax.scan(micro_step, accum_grads, (input_ids, labels))
        avg_metrics = jax.tree_util.tree_map(jnp.mean, all_metrics)
        
        # Optimization step happens strictly once the micro-batch scanning computes all gradients
        updates, new_opt_state = optimizer.update(final_accum_grads, opt_state, params=model)
        new_model = eqx.apply_updates(model, updates)
        
        return new_model, new_opt_state, avg_metrics

    return train_step