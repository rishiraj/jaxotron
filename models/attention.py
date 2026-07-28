import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional

from config import ModelConfig
from models.layers import apply_rotary_emb

class LlamaAttention(eqx.Module):
    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    o_proj: eqx.nn.Linear
    num_heads: int = eqx.field(static=True)
    num_kv_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    def __init__(self, config: ModelConfig, key: jax.Array):
        kq, kk, kv, ko = jax.random.split(key, 4)
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // self.num_heads
        
        self.q_proj = eqx.nn.Linear(config.hidden_size, self.num_heads * self.head_dim, use_bias=False, key=kq)
        self.k_proj = eqx.nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, use_bias=False, key=kk)
        self.v_proj = eqx.nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, use_bias=False, key=kv)
        self.o_proj = eqx.nn.Linear(self.num_heads * self.head_dim, config.hidden_size, use_bias=False, key=ko)

    def __call__(self, x: jax.Array, freqs_cis: jax.Array, mask: jax.Array) -> jax.Array:
        batch_size, seq_len, _ = x.shape
        
        xq = jax.vmap(jax.vmap(self.q_proj))(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        xk = jax.vmap(jax.vmap(self.k_proj))(x).reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        xv = jax.vmap(jax.vmap(self.v_proj))(x).reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
        
        # GQA Broadcast logic
        if self.num_kv_heads != self.num_heads:
            num_repeats = self.num_heads // self.num_kv_heads
            xk = jnp.repeat(xk, num_repeats, axis=2)
            xv = jnp.repeat(xv, num_repeats, axis=2)
            
        # FIX: JAX natively expects (batch, seq_len, num_heads, head_dim)
        # We removed the PyTorch-style jnp.swapaxes that was confusing XLA.
        attn_out = jax.nn.dot_product_attention(xq, xk, xv, mask=mask)
        
        # Reshape directly back to flat hidden size
        attn_out = attn_out.reshape(batch_size, seq_len, -1)
        
        return jax.vmap(jax.vmap(self.o_proj))(attn_out)

class LlamaMLP(eqx.Module):
    gate_proj: eqx.nn.Linear
    up_proj: eqx.nn.Linear
    down_proj: eqx.nn.Linear

    def __init__(self, config: ModelConfig, key: jax.Array):
        kg, ku, kd = jax.random.split(key, 3)
        self.gate_proj = eqx.nn.Linear(config.hidden_size, config.intermediate_size, use_bias=False, key=kg)
        self.up_proj = eqx.nn.Linear(config.hidden_size, config.intermediate_size, use_bias=False, key=ku)
        self.down_proj = eqx.nn.Linear(config.intermediate_size, config.hidden_size, use_bias=False, key=kd)

    def __call__(self, x: jax.Array) -> jax.Array:
        gate = jax.nn.silu(jax.vmap(jax.vmap(self.gate_proj))(x))
        up = jax.vmap(jax.vmap(self.up_proj))(x)
        return jax.vmap(jax.vmap(self.down_proj))(gate * up)