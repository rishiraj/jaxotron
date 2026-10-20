"""Load Hugging Face Llama-family checkpoints (Llama 2/3.x, SmolLM/SmolLM2, TinyLlama, ...).

    params, cfg = load_hf_llama("HuggingFaceTB/SmolLM2-135M", mesh)

Weights go straight from the safetensors files on disk into their *sharded*
layout: each device only receives the slices it owns, so the same call works
on one laptop CPU, an 8-GPU node or a multi-host TPU pod.

Safetensors files are parsed with NumPy (8-byte header length + JSON header +
raw little-endian data), so no PyTorch or `safetensors` package is required.
Downloading by repo id needs `pip install "jaxotron[hf]"`.
"""
from __future__ import annotations

import glob
import json
import os
import struct
from collections.abc import Callable

import numpy as np

from jaxotron.config import ModelConfig

_ST_DTYPES = {"F64": np.float64, "F32": np.float32, "F16": np.float16, "I64": np.int64, "I32": np.int32}


def _bf16():
    import ml_dtypes  # installed with jax
    return ml_dtypes.bfloat16


def resolve_hf_path(name_or_path: str) -> str:
    """Local directory for a Hugging Face repo id (downloading it if needed) or a local path."""
    if os.path.isdir(name_or_path):
        return name_or_path
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError('Loading from the Hub needs huggingface_hub: pip install "jaxotron[hf]"') from e
    return snapshot_download(name_or_path, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"])


def hf_model_config(path: str) -> ModelConfig:
    """Translate a Hugging Face Llama config.json into a ModelConfig (raises on unsupported features)."""
    with open(os.path.join(path, "config.json")) as f:
        c = json.load(f)
    if c.get("model_type") != "llama":
        raise ValueError(f"Only Llama-architecture checkpoints are supported (model_type='llama'); "
                         f"got {c.get('model_type')!r}.")
    if c.get("attention_bias") or c.get("mlp_bias"):
        raise ValueError("Checkpoints with attention/MLP biases are not supported.")
    if c.get("hidden_act", "silu") != "silu":
        raise ValueError(f"Unsupported activation {c.get('hidden_act')!r}; expected 'silu'.")
    hidden, heads = c["hidden_size"], c["num_attention_heads"]
    if c.get("head_dim") not in (None, hidden // heads):
        raise ValueError(f"head_dim {c['head_dim']} != hidden_size / num_attention_heads ({hidden // heads}).")

    rope = {}
    scaling = c.get("rope_scaling")
    if scaling:
        kind = scaling.get("rope_type", scaling.get("type"))
        if kind == "llama3":
            rope = dict(rope_scaling_factor=float(scaling["factor"]),
                        rope_low_freq_factor=float(scaling["low_freq_factor"]),
                        rope_high_freq_factor=float(scaling["high_freq_factor"]),
                        rope_original_max_position=int(scaling["original_max_position_embeddings"]))
        elif kind != "default":
            raise ValueError(f"Unsupported rope_scaling type {kind!r} (supported: llama3).")

    return ModelConfig(
        vocab_size=c["vocab_size"], hidden_size=hidden, intermediate_size=c["intermediate_size"],
        num_hidden_layers=c["num_hidden_layers"], num_attention_heads=heads,
        num_key_value_heads=c.get("num_key_value_heads", heads), rms_norm_eps=c.get("rms_norm_eps", 1e-5),
        rope_theta=float(c.get("rope_theta", 10000.0)), **rope,
    )


def _tied_embeddings(path: str) -> bool:
    with open(os.path.join(path, "config.json")) as f:
        return bool(json.load(f).get("tie_word_embeddings", False))


def safetensors_index(path: str) -> dict[str, Callable[[], np.ndarray]]:
    """{tensor name: loader returning a float32 NumPy array} for every *.safetensors file in path."""
    files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"No .safetensors files in {path}.")
    index = {}
    for file in files:
        with open(file, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        data = np.memmap(file, dtype=np.uint8, mode="r", offset=8 + n)
        for name, info in header.items():
            if name == "__metadata__":
                continue
            index[name] = _loader(data, info)
    return index


def _loader(data: np.ndarray, info: dict) -> Callable[[], np.ndarray]:
    def load() -> np.ndarray:
        dtype = _bf16() if info["dtype"] == "BF16" else _ST_DTYPES.get(info["dtype"])
        if dtype is None:
            raise ValueError(f"Unsupported safetensors dtype {info['dtype']}.")
        start, end = info["data_offsets"]
        arr = np.frombuffer(data[start:end], dtype=dtype).reshape(info["shape"])
        return arr.astype(np.float32)
    return load


def hf_to_jaxotron(path: str, cfg: ModelConfig) -> dict[str, Callable[[], np.ndarray]]:
    """Builders for every Jaxotron parameter (by field name), in Jaxotron's layout, from HF tensors."""
    st = safetensors_index(path)
    d, h, kv, hd, n = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.num_hidden_layers

    def get(name: str) -> np.ndarray:
        for key in (name, f"model.{name}" if not name.startswith("model.") else name[len("model."):]):
            if key in st:
                return st[key]()
        raise KeyError(f"Tensor {name!r} not found in checkpoint {path}.")

    def stacked(component: str, convert: Callable[[np.ndarray], np.ndarray]) -> Callable[[], np.ndarray]:
        return lambda: np.stack([convert(get(f"model.layers.{i}.{component}.weight")) for i in range(n)])

    tied = _tied_embeddings(path) or not any(k.endswith("lm_head.weight") for k in st)
    return {
        "embed": lambda: get("model.embed_tokens.weight"),                                   # [V, D]
        "lm_head": (lambda: get("model.embed_tokens.weight").T) if tied else (lambda: get("lm_head.weight").T),
        "final_norm": lambda: get("model.norm.weight"),
        "attn_norm": stacked("input_layernorm", lambda w: w),
        "mlp_norm": stacked("post_attention_layernorm", lambda w: w),
        "wq": stacked("self_attn.q_proj", lambda w: w.T.reshape(d, h, hd)),                  # [H*Dh, D] -> [D, H, Dh]
        "wk": stacked("self_attn.k_proj", lambda w: w.T.reshape(d, kv, hd)),
        "wv": stacked("self_attn.v_proj", lambda w: w.T.reshape(d, kv, hd)),
        "wo": stacked("self_attn.o_proj", lambda w: w.T.reshape(h, hd, d)),                  # [D, H*Dh] -> [H, Dh, D]
        "w_gate": stacked("mlp.gate_proj", lambda w: w.T),                                   # [F, D] -> [D, F]
        "w_up": stacked("mlp.up_proj", lambda w: w.T),
        "w_down": stacked("mlp.down_proj", lambda w: w.T),                                   # [D, F] -> [F, D]
    }


def load_hf_llama(name_or_path: str, mesh, cfg: ModelConfig | None = None):
    """(params, model config) with every parameter created directly in its sharded layout."""
    import jax

    from jaxotron.parallel.rules import leaf_name
    from jaxotron.trainer import abstract_model

    path = resolve_hf_path(name_or_path)
    cfg = cfg or hf_model_config(path)
    builders = hf_to_jaxotron(path, cfg)
    params_abs, _, _ = abstract_model(cfg, mesh)

    def materialise(p, leaf):
        arr = builders[leaf_name(p)]()
        if arr.shape != leaf.shape:
            raise ValueError(f"{leaf_name(p)}: checkpoint shape {arr.shape} != model shape {leaf.shape}.")
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        return jax.make_array_from_callback(leaf.shape, leaf.sharding, lambda idx: arr[idx])

    params = jax.tree_util.tree_map_with_path(materialise, params_abs)
    return params, cfg
