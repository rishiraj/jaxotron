"""Llama / SmolLM2 / Qwen3 / Qwen2.5 architecture (RMSNorm, RoPE, GQA, optional QK-Norm, SwiGLU) as an Equinox module.

Design choices that make 4D parallelism easy:
  * Weights are raw arrays with explicit head dimensions (e.g. wq is
    [hidden, heads, head_dim]) so tensor parallelism is "shard the heads axis".
  * All decoder layers are stacked into one Block whose arrays have a leading
    [num_layers] dim. This gives fast compilation (one lax.scan instead
    of an unrolled Python loop) and makes pipeline parallelism a reshape.
  * The parameter layout does not depend on the parallel configuration, so a
    checkpoint saved with one mesh can be restored on any other mesh.
"""
from __future__ import annotations

import math
from functools import partial
from typing import NamedTuple, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from jaxotron.config import ModelConfig
from jaxotron.models.layers import apply_rope, causal_attention, rms_norm, rope_tables
from jaxotron.parallel.context import ParallelContext
from jaxotron.parallel.pipeline import pipeline_apply


class KVCache(NamedTuple):
    """Per-layer KV cache for fast autoregressive generation: [layers, batch, max_seq, kv_heads, head_dim]."""
    k: jax.Array
    v: jax.Array


class Block(eqx.Module):
    attn_norm: jax.Array                # [D]
    wq: jax.Array                       # [D, H, Dh]
    wk: jax.Array                       # [D, K, Dh]
    wv: jax.Array                       # [D, K, Dh]
    wo: jax.Array                       # [H, Dh, D]
    mlp_norm: jax.Array                 # [D]
    w_gate: jax.Array                   # [D, F]
    w_up: jax.Array                     # [D, F]
    w_down: jax.Array                   # [F, D]
    q_norm: Optional[jax.Array] = None  # [Dh] (Qwen3 per-head QK-Norm)
    k_norm: Optional[jax.Array] = None  # [Dh]
    bq: Optional[jax.Array] = None      # [H, Dh] (Qwen2.5 QKV biases)
    bk: Optional[jax.Array] = None      # [K, Dh]
    bv: Optional[jax.Array] = None      # [K, Dh]

    def __init__(self, cfg: ModelConfig, key: jax.Array):
        d, h, kv, hd, f = (cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads,
                           cfg.head_dim, cfg.intermediate_size)
        std = cfg.init_std
        out_std = std / math.sqrt(2 * cfg.num_hidden_layers)  # GPT-2 style residual scaling
        ks = jax.random.split(key, 7)
        normal = lambda k, shape, s: s * jax.random.normal(k, shape, jnp.float32)
        self.attn_norm = jnp.ones((d,), jnp.float32)
        self.wq = normal(ks[0], (d, h, hd), std)
        self.wk = normal(ks[1], (d, kv, hd), std)
        self.wv = normal(ks[2], (d, kv, hd), std)
        self.wo = normal(ks[3], (h, hd, d), out_std)
        self.mlp_norm = jnp.ones((d,), jnp.float32)
        self.w_gate = normal(ks[4], (d, f), std)
        self.w_up = normal(ks[5], (d, f), std)
        self.w_down = normal(ks[6], (f, d), out_std)
        if cfg.qk_norm:
            self.q_norm = jnp.ones((hd,), jnp.float32)
            self.k_norm = jnp.ones((hd,), jnp.float32)
        if cfg.qkv_bias:
            self.bq = jnp.zeros((h, hd), jnp.float32)
            self.bk = jnp.zeros((kv, hd), jnp.float32)
            self.bv = jnp.zeros((kv, hd), jnp.float32)


class Llama(eqx.Module):
    embed: jax.Array       # [V, D]
    blocks: Block          # every array has a leading [num_layers] dim
    final_norm: jax.Array  # [D]
    lm_head: jax.Array     # [D, V]
    cfg: ModelConfig = eqx.field(static=True)

    def __init__(self, cfg: ModelConfig, key: jax.Array):
        k_emb, k_blocks, k_head = jax.random.split(key, 3)
        self.cfg = cfg
        self.embed = cfg.init_std * jax.random.normal(k_emb, (cfg.vocab_size, cfg.hidden_size), jnp.float32)
        layer_keys = jax.random.split(k_blocks, cfg.num_hidden_layers)
        self.blocks = eqx.filter_vmap(lambda k: Block(cfg, k))(layer_keys)
        self.final_norm = jnp.ones((cfg.hidden_size,), jnp.float32)
        self.lm_head = cfg.init_std * jax.random.normal(k_head, (cfg.hidden_size, cfg.vocab_size), jnp.float32)


def rope_scaling(cfg: ModelConfig):
    if not cfg.rope_scaling_factor:
        return None
    return (cfg.rope_scaling_factor, cfg.rope_low_freq_factor, cfg.rope_high_freq_factor,
            cfg.rope_original_max_position)


def _qkv(p: Block, h: jax.Array, ctx: ParallelContext, eps: float):
    q = jnp.einsum("btd,dhk->bthk", h, p.wq)
    k = jnp.einsum("btd,dhk->bthk", h, p.wk)
    v = jnp.einsum("btd,dhk->bthk", h, p.wv)
    if p.bq is not None:
        q = q + p.bq.astype(q.dtype)
        k = k + p.bk.astype(k.dtype)
        v = v + p.bv.astype(v.dtype)
    q = ctx.shard(q, "batch", "seq", "heads", None)
    k = ctx.shard(k, "batch", "seq", "kv_heads", None)
    v = ctx.shard(v, "batch", "seq", "kv_heads", None)
    if p.q_norm is not None:
        q = rms_norm(q, p.q_norm, eps)
        k = rms_norm(k, p.k_norm, eps)
    return q, k, v


def block_forward(p: Block, x: jax.Array, cos, sin, ctx: ParallelContext, eps: float) -> jax.Array:
    """One decoder layer. Sharding annotations name each activation's logical axes."""
    # --- Attention -----------------------------------------------------------
    h = ctx.shard(rms_norm(x, p.attn_norm, eps), "batch", "res_seq", "act_embed")
    q, k, v = _qkv(p, h, ctx, eps)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    # Context parallelism: each cp rank keeps its slice of queries, but needs
    # every key/value. Replicating K/V over "cp" makes XLA all-gather them.
    k = ctx.shard(k, "batch", "kv_seq", "kv_heads", None)
    v = ctx.shard(v, "batch", "kv_seq", "kv_heads", None)
    attn = ctx.shard(causal_attention(q, k, v, ctx.attention_impl), "batch", "seq", "heads", None)
    # Row-parallel output projection: XLA inserts the tensor-parallel all-reduce (or reduce-scatter with --sp).
    x = x + ctx.shard(jnp.einsum("bthk,hkd->btd", attn, p.wo), "batch", "res_seq", "act_embed")

    # --- SwiGLU MLP ----------------------------------------------------------
    h = ctx.shard(rms_norm(x, p.mlp_norm, eps), "batch", "res_seq", "act_embed")
    gate = ctx.shard(jnp.einsum("btd,df->btf", h, p.w_gate), "batch", "seq", "mlp")
    up = ctx.shard(jnp.einsum("btd,df->btf", h, p.w_up), "batch", "seq", "mlp")
    out = jnp.einsum("btf,fd->btd", jax.nn.silu(gate) * up, p.w_down)
    return x + ctx.shard(out, "batch", "res_seq", "act_embed")


def run_layers(blocks: Block, x: jax.Array, *, cos, sin, ctx: ParallelContext, eps: float) -> jax.Array:
    """Apply a stack of layers with lax.scan (+ activation rematerialisation)."""
    def body(h, layer):
        return block_forward(layer, h, cos, sin, ctx, eps), None

    if ctx.remat == "full":
        body = jax.checkpoint(body, prevent_cse=False, policy=jax.checkpoint_policies.nothing_saveable)
    elif ctx.remat == "dots":
        body = jax.checkpoint(body, prevent_cse=False,
                              policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
    x, _ = jax.lax.scan(body, x, blocks)
    return x


def forward(model: Llama, tokens: jax.Array, ctx: ParallelContext) -> jax.Array:
    """tokens [B, T] int32 -> logits [B, T, V] float32."""
    cfg, dtype = model.cfg, ctx.dtype
    seq_len = tokens.shape[1]

    tokens = ctx.shard(tokens, "batch", "res_seq")
    x = jnp.take(model.embed.astype(dtype), tokens, axis=0)
    x = ctx.shard(x, "batch", "res_seq", "act_embed")

    cos, sin = rope_tables(seq_len, cfg.head_dim, cfg.rope_theta, rope_scaling(cfg))
    blocks = jax.tree.map(lambda a: a.astype(dtype), model.blocks)  # mixed precision: bf16 compute
    stage_fn = partial(run_layers, cos=cos, sin=sin, ctx=ctx, eps=cfg.rms_norm_eps)

    if ctx.pp > 1:
        # [L, ...] -> [stages, L / stages, ...]; the stage dim inherits the "pp" sharding.
        stages = jax.tree.map(lambda a: a.reshape(ctx.pp, -1, *a.shape[1:]), blocks)
        x = pipeline_apply(stage_fn, stages, x, ctx)
    else:
        x = stage_fn(blocks, x)

    x = ctx.shard(rms_norm(x, model.final_norm, cfg.rms_norm_eps), "batch", "seq", "act_embed")
    logits = jnp.einsum("btd,dv->btv", x, model.lm_head.astype(dtype), preferred_element_type=jnp.float32)
    return ctx.shard(logits, "batch", "seq", "vocab")


def init_kv_cache(cfg: ModelConfig, batch_size: int, max_seq: int, ctx: ParallelContext) -> KVCache:
    """Allocate a sharded, zero-initialised KV cache for autoregressive generation (when pp == 1 and cp == 1)."""
    shape = (cfg.num_hidden_layers, batch_size, max_seq, cfg.num_key_value_heads, cfg.head_dim)
    if ctx.mesh is not None and ctx.mesh.size > 1:
        sharding = NamedSharding(ctx.mesh, P("pp", ("dp", "fsdp"), None, "tp", None))
        zeros = jax.jit(lambda: jnp.zeros(shape, ctx.dtype), out_shardings=sharding)
        return KVCache(k=zeros(), v=zeros())
    return KVCache(k=jnp.zeros(shape, ctx.dtype), v=jnp.zeros(shape, ctx.dtype))


def forward_cached(
    model: Llama,
    tokens: jax.Array,
    pos: jax.Array,
    cache: KVCache,
    ctx: ParallelContext,
) -> tuple[jax.Array, KVCache]:
    """KV-cached forward pass for both prompt prefill (`tokens` [B, P], `pos=0`) and single-token decode (`[B, 1]`)."""
    cfg, dtype = model.cfg, ctx.dtype
    seq_len, max_seq = tokens.shape[1], cache.k.shape[2]
    pos = jnp.asarray(pos, jnp.int32)

    tokens = ctx.shard(tokens, "batch", "seq")
    x = ctx.shard(jnp.take(model.embed.astype(dtype), tokens, axis=0), "batch", "seq", "act_embed")

    cos_all, sin_all = rope_tables(max_seq, cfg.head_dim, cfg.rope_theta, rope_scaling(cfg))
    cos = jax.lax.dynamic_slice_in_dim(cos_all, pos, seq_len, axis=0)
    sin = jax.lax.dynamic_slice_in_dim(sin_all, pos, seq_len, axis=0)
    blocks = jax.tree.map(lambda a: a.astype(dtype), model.blocks)

    def body(h, layer_and_kv):
        p, k_cache, v_cache = layer_and_kv
        u = rms_norm(h, p.attn_norm, cfg.rms_norm_eps)
        q, k, v = _qkv(p, u, ctx, cfg.rms_norm_eps)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        k_cache = jax.lax.dynamic_update_slice_in_dim(k_cache, k.astype(k_cache.dtype), pos, axis=1)
        v_cache = jax.lax.dynamic_update_slice_in_dim(v_cache, v.astype(v_cache.dtype), pos, axis=1)
        k_cache = ctx.shard(k_cache, "batch", "kv_seq", "kv_heads", None)
        v_cache = ctx.shard(v_cache, "batch", "kv_seq", "kv_heads", None)
        attn = causal_attention(q, k_cache, v_cache, impl="reference", pos_offset=pos)
        attn = ctx.shard(attn, "batch", "seq", "heads", None)
        h = h + ctx.shard(jnp.einsum("bthk,hkd->btd", attn, p.wo), "batch", "seq", "act_embed")

        u = rms_norm(h, p.mlp_norm, cfg.rms_norm_eps)
        gate = ctx.shard(jnp.einsum("btd,df->btf", u, p.w_gate), "batch", "seq", "mlp")
        up = ctx.shard(jnp.einsum("btd,df->btf", u, p.w_up), "batch", "seq", "mlp")
        out = jnp.einsum("btf,fd->btd", jax.nn.silu(gate) * up, p.w_down)
        return h + ctx.shard(out, "batch", "seq", "act_embed"), (k_cache, v_cache)

    x, (new_k, new_v) = jax.lax.scan(body, x, (blocks, cache.k, cache.v))
    x = rms_norm(x, model.final_norm, cfg.rms_norm_eps)
    logits = jnp.einsum("btd,dv->btv", x, model.lm_head.astype(dtype), preferred_element_type=jnp.float32)
    return ctx.shard(logits, "batch", "seq", "vocab"), KVCache(k=new_k, v=new_v)


def count_params(cfg: ModelConfig) -> int:
    d, f, l, v = cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers, cfg.vocab_size
    h, kv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    per_layer = d * h * hd + 2 * d * kv * hd + h * hd * d + 3 * d * f + 2 * d
    if cfg.qk_norm:
        per_layer += 2 * hd
    if cfg.qkv_bias:
        per_layer += (h + 2 * kv) * hd
    return 2 * v * d + l * per_layer + d


def flops_per_token(cfg: ModelConfig, seq_len: int) -> float:
    """Training FLOPs per token: 6 * matmul params + attention (PaLM / nanoGPT convention)."""
    matmul_params = count_params(cfg) - cfg.vocab_size * cfg.hidden_size  # embedding lookup is free
    attn = 12 * cfg.num_hidden_layers * cfg.num_attention_heads * cfg.head_dim * seq_len
    return 6 * matmul_params + attn