import jax
import jax.numpy as jnp
import equinox as eqx

class RMSNorm(eqx.Module):
    weight: jax.Array
    # FIX: Must explicitly mark non-JAX array types as static in Equinox
    eps: float = eqx.field(static=True)

    def __init__(self, size: int, eps: float = 1e-5):
        self.weight = jnp.ones(size)
        self.eps = eps

    def __call__(self, x: jax.Array) -> jax.Array:
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        return (x * jax.lax.rsqrt(variance + self.eps)) * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> jax.Array:
    """Precomputes rotary frequencies matching Hugging Face LLaMA's format."""
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2).astype(jnp.float32) / dim))
    t = jnp.arange(end, dtype=jnp.float32)
    freqs_outer = jnp.outer(t, freqs)
    # LLaMA groups the rotation by splitting the embedding dim in half
    return jnp.concatenate([jnp.cos(freqs_outer), jnp.sin(freqs_outer)], axis=-1)

def apply_rotary_emb(xq: jax.Array, xk: jax.Array, freqs_cis: jax.Array):
    """Applies RoPE using pure real arithmetic math to exactly match HuggingFace."""
    d = xq.shape[-1]
    
    # Split queries and keys in half along the feature dimension
    xq1, xq2 = xq[..., :d//2], xq[..., d//2:]
    xk1, xk2 = xk[..., :d//2], xk[..., d//2:]
    
    # Cosine and sine frequencies
    cos, sin = freqs_cis[..., :d//2], freqs_cis[..., d//2:]
    cos = cos[None, :, None, :]  # Broadcast: (1, seq_len, 1, head_dim/2)
    sin = sin[None, :, None, :]
    
    # Rotate
    rotated_xq1 = xq1 * cos - xq2 * sin
    rotated_xq2 = xq2 * cos + xq1 * sin
    
    rotated_xk1 = xk1 * cos - xk2 * sin
    rotated_xk2 = xk2 * cos + xk1 * sin
    
    xq_out = jnp.concatenate([rotated_xq1, rotated_xq2], axis=-1)
    xk_out = jnp.concatenate([rotated_xk1, rotated_xk2], axis=-1)
    
    return xq_out.astype(xq.dtype), xk_out.astype(xk.dtype)