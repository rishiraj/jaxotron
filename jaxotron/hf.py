"""Load Hugging Face Llama / SmolLM2 / Qwen3 / Qwen2.5 checkpoints into Jaxotron's sharded layout.

    params, cfg = load_hf_llama("HuggingFaceTB/SmolLM2-135M", mesh)
    params, cfg = load_hf_llama("Qwen/Qwen3-0.6B", mesh)

Weights go straight from the safetensors files on disk into their *sharded*
layout. Every parameter is assembled lazily, one Hugging Face tensor at a time,
and only for the slices that local devices own: with pipeline parallelism
(`pp > 1`) a host only reads the layers of its own stages, and on multi-host
setups each host only touches the shards of its own devices.

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
from typing import Optional

import numpy as np

from jaxotron.config import ModelConfig

SUPPORTED_MODEL_TYPES = ("llama", "qwen3", "qwen2")
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
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    return snapshot_download(
        name_or_path,
        token=token,
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"],
    )


def hf_model_config(path: str) -> ModelConfig:
    """Translate a Hugging Face Llama / Qwen3 / Qwen2 config.json into a ModelConfig."""
    with open(os.path.join(path, "config.json")) as f:
        c = json.load(f)
    model_type = c.get("model_type")
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Only Llama/Qwen3/Qwen2-architecture checkpoints are supported "
            f"(model_type in {SUPPORTED_MODEL_TYPES}); got {model_type!r}."
        )
    if c.get("mlp_bias"):
        raise ValueError("Checkpoints with MLP biases are not supported.")
    if c.get("use_sliding_window") and c.get("sliding_window") is not None:
        raise ValueError("Checkpoints with active sliding-window attention are not supported.")
    if c.get("hidden_act", "silu") != "silu":
        raise ValueError(f"Unsupported activation {c.get('hidden_act')!r}; expected 'silu'.")

    hidden, heads = int(c["hidden_size"]), int(c["num_attention_heads"])
    default_hd = hidden // heads
    explicit_hd = int(c.get("head_dim") or default_hd)
    attn_head_dim = explicit_hd if explicit_hd != default_hd else 0

    rope_params = c.get("rope_parameters") or {}  # transformers v5 layout
    rope_theta = float(c.get("rope_theta", rope_params.get("rope_theta", 10000.0)))

    rope = {}
    scaling = c.get("rope_scaling") or (rope_params if "rope_type" in rope_params else None)
    if scaling:
        kind = scaling.get("rope_type", scaling.get("type"))
        if kind == "llama3":
            rope = dict(
                rope_scaling_factor=float(scaling["factor"]),
                rope_low_freq_factor=float(scaling.get("low_freq_factor", 1.0)),
                rope_high_freq_factor=float(scaling.get("high_freq_factor", 4.0)),
                rope_original_max_position=int(scaling.get("original_max_position_embeddings", 8192)),
            )
        elif kind not in (None, "default"):
            raise ValueError(f"Unsupported rope_scaling type {kind!r} (supported: llama3).")

    qk_norm = bool(c.get("qk_norm", model_type == "qwen3"))
    qkv_bias = bool(c.get("attention_bias", model_type == "qwen2"))

    return ModelConfig(
        vocab_size=int(c["vocab_size"]),
        hidden_size=hidden,
        intermediate_size=int(c["intermediate_size"]),
        num_hidden_layers=int(c["num_hidden_layers"]),
        num_attention_heads=heads,
        num_key_value_heads=int(c.get("num_key_value_heads", heads)),
        rms_norm_eps=float(c.get("rms_norm_eps", 1e-5)),
        rope_theta=rope_theta,
        attn_head_dim=attn_head_dim,
        qk_norm=qk_norm,
        qkv_bias=qkv_bias,
        **rope,
    )


def _tied_embeddings(path: str) -> bool:
    with open(os.path.join(path, "config.json")) as f:
        return bool(json.load(f).get("tie_word_embeddings", False))


def safetensors_index(path: str) -> dict[str, Callable[[], np.ndarray]]:
    """{tensor name: loader returning a float32 NumPy array} for *.safetensors files in path."""
    index_json = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(index_json):
        with open(index_json) as f:
            weight_map = json.load(f).get("weight_map", {})
        files = sorted({os.path.join(path, fname) for fname in weight_map.values()})
    else:
        files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"No .safetensors files in {path}.")
    index = {}
    for file in files:
        with open(file, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        data = np.memmap(file, dtype=np.uint8, mode="r", offset=8 + n)  # nothing is read until a loader runs
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


class LazyParam:
    """One Jaxotron parameter, assembled from Hugging Face tensors only where a device needs it.

    Call it with a device's index (a tuple of slices, as given by
    jax.make_array_from_callback) and the full parameter shape. Stacked
    parameters ([num_layers, ...]) only load the layers in that index, so a
    pipeline stage never reads another stage's layers from disk.
    """

    def __init__(self, name: str, load: Callable, num_layers: Optional[int] = None):
        self.name, self._load, self._n = name, load, num_layers
        self._cache: dict = {}

    def _tensor(self, layer: Optional[int]) -> np.ndarray:
        if layer not in self._cache:  # several tensor-parallel shards reuse the same layer
            self._cache[layer] = self._load() if layer is None else self._load(layer)
        return self._cache[layer]

    def _check(self, got, want, where: str = "") -> None:
        if tuple(got) != tuple(want):
            raise ValueError(f"{self.name}{where}: checkpoint shape {tuple(got)} != model shape {tuple(want)}.")

    def __call__(self, index: tuple, shape: tuple) -> np.ndarray:
        if self._n is None:
            full = self._tensor(None)
            self._check(full.shape, shape)
            return full[index]
        self._check((self._n,), shape[:1], " (number of layers)")
        rows = []
        for i in range(self._n)[index[0]]:
            w = self._tensor(i)
            self._check(w.shape, shape[1:], f" (layer {i})")
            rows.append(w[index[1:]])
        return np.stack(rows)

    def clear(self) -> None:
        self._cache.clear()


def hf_to_jaxotron(path: str, cfg: ModelConfig) -> dict[str, LazyParam]:
    """A LazyParam for every Jaxotron parameter (by field name), in Jaxotron's layout, from HF tensors."""
    st = safetensors_index(path)
    d, h, kv, hd, n = (
        cfg.hidden_size,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim,
        cfg.num_hidden_layers,
    )

    def get(name: str) -> np.ndarray:
        for key in (name, f"model.{name}" if not name.startswith("model.") else name[len("model."):]):
            if key in st:
                return st[key]()
        raise KeyError(f"Tensor {name!r} not found in checkpoint {path}.")

    def single(load: Callable[[], np.ndarray]):
        return lambda name: LazyParam(name, load)

    def stacked(component: str, convert: Callable[[np.ndarray], np.ndarray], suffix: str = "weight"):
        return lambda name: LazyParam(name, lambda i: convert(get(f"model.layers.{i}.{component}.{suffix}")), n)

    tied = _tied_embeddings(path) or not any(k.endswith("lm_head.weight") for k in st)
    makers = {
        "embed": single(lambda: get("model.embed_tokens.weight")),                          # [V, D]
        "lm_head": single((lambda: get("model.embed_tokens.weight").T) if tied else (lambda: get("lm_head.weight").T)),
        "final_norm": single(lambda: get("model.norm.weight")),
        "attn_norm": stacked("input_layernorm", lambda w: w),
        "mlp_norm": stacked("post_attention_layernorm", lambda w: w),
        "wq": stacked("self_attn.q_proj", lambda w: w.T.reshape(d, h, hd)),                 # [H*Dh, D] -> [D, H, Dh]
        "wk": stacked("self_attn.k_proj", lambda w: w.T.reshape(d, kv, hd)),
        "wv": stacked("self_attn.v_proj", lambda w: w.T.reshape(d, kv, hd)),
        "wo": stacked("self_attn.o_proj", lambda w: w.T.reshape(h, hd, d)),                 # [D, H*Dh] -> [H, Dh, D]
        "w_gate": stacked("mlp.gate_proj", lambda w: w.T),                                  # [F, D] -> [D, F]
        "w_up": stacked("mlp.up_proj", lambda w: w.T),
        "w_down": stacked("mlp.down_proj", lambda w: w.T),                                  # [D, F] -> [F, D]
    }
    if cfg.qk_norm:
        makers["q_norm"] = stacked("self_attn.q_norm", lambda w: w)
        makers["k_norm"] = stacked("self_attn.k_norm", lambda w: w)
    if cfg.qkv_bias:
        makers["bq"] = stacked("self_attn.q_proj", lambda b: b.reshape(h, hd), suffix="bias")
        makers["bk"] = stacked("self_attn.k_proj", lambda b: b.reshape(kv, hd), suffix="bias")
        makers["bv"] = stacked("self_attn.v_proj", lambda b: b.reshape(kv, hd), suffix="bias")
    return {name: make(name) for name, make in makers.items()}


def load_hf_llama(name_or_path: str, mesh, cfg: ModelConfig | None = None):
    """(params, model config) with every parameter created directly in its sharded layout."""
    import jax

    from jaxotron.parallel.rules import leaf_name
    from jaxotron.trainer import abstract_model

    path = resolve_hf_path(name_or_path)
    cfg = cfg or hf_model_config(path)
    lazy = hf_to_jaxotron(path, cfg)
    params_abs, _, _ = abstract_model(cfg, mesh)

    def materialise(p, leaf):
        param = lazy[leaf_name(p)]

        def shard(index):  # called once per addressable device, only for the slice it owns
            return np.ascontiguousarray(param(index, leaf.shape), dtype=np.float32)

        try:
            return jax.make_array_from_callback(leaf.shape, leaf.sharding, shard)
        finally:
            param.clear()  # free host memory before the next parameter

    params = jax.tree_util.tree_map_with_path(materialise, params_abs)
    return params, cfg


load_hf_pretrained = load_hf_llama  # alias supporting Llama, SmolLM2, Qwen3, and Qwen2.5
