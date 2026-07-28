import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional

from config import ModelConfig
from models.layers import RMSNorm, precompute_freqs_cis
from models.attention import LlamaAttention, LlamaMLP

class LlamaBlock(eqx.Module):
    input_layernorm: RMSNorm
    self_attn: LlamaAttention
    post_attention_layernorm: RMSNorm
    mlp: LlamaMLP

    def __init__(self, config: ModelConfig, key: jax.Array):
        k_attn, k_mlp = jax.random.split(key, 2)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = LlamaAttention(config, key=k_attn)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = LlamaMLP(config, key=k_mlp)

    def __call__(self, x: jax.Array, freqs_cis: jax.Array, mask: Optional[jax.Array] = None) -> jax.Array:
        h = x + self.self_attn(self.input_layernorm(x), freqs_cis, mask)
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out

class LlamaForCausalLM(eqx.Module):
    embed_tokens: eqx.nn.Embedding
    layers: tuple[LlamaBlock, ...]
    norm: RMSNorm
    lm_head: eqx.nn.Linear
    
    config: ModelConfig = eqx.field(static=True)

    def __init__(self, config: ModelConfig, key: jax.Array):
        self.config = config
        
        k_emb, k_head, k_layers = jax.random.split(key, 3)
        layer_keys = jax.random.split(k_layers, config.num_hidden_layers)
        
        self.embed_tokens = eqx.nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_size=config.hidden_size,
            key=k_emb
        )
        
        self.layers = tuple(LlamaBlock(config, key=l_key) for l_key in layer_keys)
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = eqx.nn.Linear(config.hidden_size, config.vocab_size, use_bias=False, key=k_head)

    def __call__(self, input_ids: jax.Array) -> jax.Array:
        batch_size, seq_len = input_ids.shape
        
        # Memory-efficient bool mask creation
        mask = jnp.arange(seq_len)[:, None] >= jnp.arange(seq_len)[None, :]
        mask = mask.reshape(1, 1, seq_len, seq_len)
        
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        freqs_cis = precompute_freqs_cis(dim=head_dim, end=seq_len, theta=self.config.rope_theta)
        
        x = jax.vmap(jax.vmap(self.embed_tokens))(input_ids)
        
        for layer in self.layers:
            # Gradient checkpointing to save memory
            layer_fn = eqx.filter_checkpoint(layer)
            x = layer_fn(x, freqs_cis, mask)
            
        x = self.norm(x)
        logits = jax.vmap(jax.vmap(self.lm_head))(x)
        return logits