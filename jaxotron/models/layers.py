"""Llama building blocks: RMSNorm, rotary embeddings and causal GQA attention."""
from __future__ import annotations

import math
from typing import Optional

import jax
import jax.numpy as jnp


def rms_norm(x: jax.Array, weight: jax.Array, eps: float) -> jax.Array:
    """RMSNorm computed in float32 for stability (matches Hugging Face LlamaRMSNorm)."""
    dtype = x.dtype
    x32 = x.astype(jnp.float32)
    x32 = x32 * jax.lax.rsqrt(jnp.mean(jnp.square(x32), axis=-1, keepdims=True) + eps)
    return x32.astype(dtype) * weight.astype(dtype)


def rope_inv_freq(head_dim: int, theta: float, scaling: Optional[tuple[float, float, float, int]] = None):
    """Inverse frequencies [head_dim // 2], optionally with Llama 3.1 "llama3" scaling.

    scaling = (factor, low_freq_factor, high_freq_factor, original_max_position).
    Long wavelengths are divided by `factor`, short ones are kept and the band in
    between is interpolated (identical to Hugging Face `rope_type="llama3"`).
    """
    inv_freq = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    if not scaling or not scaling[0]:
        return inv_freq
    factor, low, high, original = scaling
    low_wavelen, high_wavelen = original / low, original / high
    wavelen = 2 * math.pi / inv_freq
    scaled = jnp.where(wavelen > low_wavelen, inv_freq / factor, inv_freq)
    smooth = (original / wavelen - low) / (high - low)
    smoothed = (1 - smooth) * scaled / factor + smooth * scaled
    is_medium = (wavelen >= high_wavelen) & (wavelen <= low_wavelen)
    return jnp.where(is_medium, smoothed, scaled)


def rope_tables(seq_len: int, head_dim: int, theta: float,
                scaling: Optional[tuple[float, float, float, int]] = None) -> tuple[jax.Array, jax.Array]:
    """cos/sin tables of shape [seq_len, head_dim // 2] (float32)."""
    angles = jnp.outer(jnp.arange(seq_len, dtype=jnp.float32), rope_inv_freq(head_dim, theta, scaling))
    return jnp.cos(angles), jnp.sin(angles)


def apply_rope(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    """Rotary embedding with the "rotate half" convention used by Hugging Face Llama.

    x: [batch, seq, heads, head_dim]
    """
    dtype = x.dtype
    x = x.astype(jnp.float32)
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    out = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
    return out.astype(dtype)


def causal_attention(q: jax.Array, k: jax.Array, v: jax.Array, impl: str = "reference") -> jax.Array:
    """Causal grouped-query attention.

    q: [B, T, H, D]   k, v: [B, S, K, D]   with H a multiple of K.
    Returns [B, T, H, D].

    The reference implementation is written with plain einsums so GSPMD can
    partition it along any axis (batch, heads, and query sequence for context
    parallelism). Positions are global, so the causal mask stays correct when
    the sequence is sharded.
    """
    if impl == "cudnn":
        return jax.nn.dot_product_attention(q, k, v, is_causal=True, implementation="cudnn")

    b, t, h, d = q.shape
    s, kv = k.shape[1], k.shape[2]
    q = q.reshape(b, t, kv, h // kv, d)  # query head h uses kv head h // (H/K), like HF repeat_kv
    scores = jnp.einsum("btkgd,bskd->bkgts", q, k, preferred_element_type=jnp.float32) * (d ** -0.5)
    mask = jnp.arange(t)[:, None] >= jnp.arange(s)[None, :]
    scores = jnp.where(mask, scores, jnp.finfo(jnp.float32).min)
    probs = jax.nn.softmax(scores, axis=-1).astype(v.dtype)
    out = jnp.einsum("bkgts,bskd->btkgd", probs, v)
    return out.reshape(b, t, h, d)
