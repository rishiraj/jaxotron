"""Pipeline parallelism in ~40 lines of pure, jit-compatible JAX.

This is the GPipe schedule expressed the "GSPMD way" (as in the GSPMD paper,
Praxis and MaxText):

  * Layer weights are stacked as [num_stages, layers_per_stage, ...] and the
    stage dimension is sharded over the "pp" mesh axis, so each pipeline rank
    holds only its own stage.
  * A state buffer [num_stages, microbatch, seq, hidden] holds the activation
    currently being processed by every stage. It is also sharded over "pp".
  * Every iteration, all stages run *in parallel* (jax.vmap over the stage
    dimension, sharded by spmd_axis_name="pp"), then the buffer is rotated by
    one slot (jnp.roll), which XLA lowers to a point-to-point
    collective-permute: stage i sends its output to stage i+1.
  * Microbatch m enters stage 0 at iteration m and leaves the last stage at
    iteration m + S - 1, so the loop runs M + S - 1 iterations.
    Bubble fraction = (S - 1) / (M + S - 1).

The backward pass needs no extra code: jax.grad differentiates through the
loop, producing the mirrored backward pipeline automatically.

Compare with Picotron's pipeline_parallel.py, where the forward and backward
schedules (AFAB / 1F1B), send/recv pairs and gradient bookkeeping are written
by hand.
"""
from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from jaxotron.parallel.context import ParallelContext


def bubble_fraction(num_stages: int, num_microbatches: int) -> float:
    """Fraction of pipeline time that each stage is idle in the GPipe schedule."""
    return (num_stages - 1) / (num_microbatches + num_stages - 1)


def split_microbatches(x: jax.Array, num: int) -> jax.Array:
    """[B, ...] -> [num, B // num, ...].

    Microbatch m takes rows m, m + num, m + 2*num, ... Because the batch is
    sharded in contiguous blocks, this keeps every row on the device that
    already owns it: no communication is needed to form microbatches.
    """
    b = x.shape[0]
    if b % num:
        raise ValueError(f"Batch size {b} is not divisible by {num} microbatches.")
    return x.reshape(b // num, num, *x.shape[1:]).swapaxes(0, 1)


def merge_microbatches(x: jax.Array) -> jax.Array:
    """Inverse of split_microbatches."""
    return x.swapaxes(0, 1).reshape(-1, *x.shape[2:])


def pipeline_apply(stage_fn: Callable, stage_params, x: jax.Array, ctx: ParallelContext) -> jax.Array:
    """Run x [B, T, D] through all pipeline stages.

    stage_fn(params_of_one_stage, h) -> h, applied to h of shape [B/M, T, D].
    stage_params: pytree with leading dimension = number of stages.
    """
    num_stages, num_micro = ctx.pp, ctx.pp_microbatches

    inputs = ctx.shard(split_microbatches(x, num_micro), None, "batch", "res_seq", "act_embed")
    outputs = jnp.zeros_like(inputs)
    state = jnp.zeros((num_stages,) + inputs.shape[1:], inputs.dtype)
    state = ctx.shard(state, "stage", "batch", "res_seq", "act_embed")

    # All stages compute simultaneously; the vmapped axis is sharded over "pp".
    run_all_stages = jax.vmap(stage_fn, spmd_axis_name="pp")

    def iteration(carry, i):
        state, outputs = carry
        # 1. Stage 0 receives microbatch i (zeros once all microbatches have entered).
        new_input = jax.lax.dynamic_index_in_dim(inputs, jnp.minimum(i, num_micro - 1), keepdims=False)
        new_input = jnp.where(i < num_micro, new_input, jnp.zeros_like(new_input))
        state = ctx.shard(state.at[0].set(new_input), "stage", "batch", "res_seq", "act_embed")

        # 2. Every stage processes its slot in parallel.
        state = run_all_stages(stage_params, state)

        # 3. The last stage finishes microbatch i - (S - 1).
        out_idx = i - (num_stages - 1)
        safe_idx = jnp.clip(out_idx, 0, num_micro - 1)
        previous = jax.lax.dynamic_index_in_dim(outputs, safe_idx, keepdims=False)
        finished = jnp.where(out_idx >= 0, state[num_stages - 1], previous)
        outputs = jax.lax.dynamic_update_index_in_dim(outputs, finished, safe_idx, axis=0)

        # 4. Send activations to the next stage (lowered to a collective-permute).
        state = ctx.shard(jnp.roll(state, shift=1, axis=0), "stage", "batch", "res_seq", "act_embed")
        return (state, outputs), None

    (_, outputs), _ = jax.lax.scan(iteration, (state, outputs), jnp.arange(num_micro + num_stages - 1))
    return merge_microbatches(outputs)