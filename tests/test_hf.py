"""Hugging Face Llama, SmolLM2, Qwen3, and Qwen2.5 checkpoints load into Jaxotron's sharded layout and match logits."""
import json
import struct

import equinox as eqx
import jax
import ml_dtypes
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P

from jaxotron.config import Config, ParallelConfig, TrainConfig
from jaxotron.hf import hf_model_config, load_hf_llama
from jaxotron.models.llama import forward
from jaxotron.parallel.mesh import create_mesh
from jaxotron.trainer import abstract_model, make_context

HF_CONFIG = dict(model_type="llama", vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                 num_attention_heads=4, num_key_value_heads=2, rms_norm_eps=1e-5, rope_theta=10000.0,
                 hidden_act="silu", tie_word_embeddings=False)


def _write_safetensors(path, tensors: dict, dtype="F32"):
    """Minimal safetensors writer (the format is an 8-byte header size, a JSON header and raw bytes)."""
    header, blobs, offset = {}, [], 0
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr.astype(ml_dtypes.bfloat16 if dtype == "BF16" else np.float32))
        header[name] = {"dtype": dtype, "shape": list(arr.shape), "data_offsets": [offset, offset + arr.nbytes]}
        blobs.append(arr.tobytes())
        offset += arr.nbytes
    raw = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)) + raw + b"".join(blobs))


def _random_hf_checkpoint(directory, tied=False, dtype="F32", model_type="llama", head_dim=None, qkv_bias=False):
    c = dict(HF_CONFIG, model_type=model_type, tie_word_embeddings=tied, attention_bias=qkv_bias)
    if head_dim is not None:
        c["head_dim"] = head_dim
    rng = np.random.default_rng(0)
    d, f, v = c["hidden_size"], c["intermediate_size"], c["vocab_size"]
    h, kv = c["num_attention_heads"], c["num_key_value_heads"]
    hd = c.get("head_dim") or (d // h)
    qd, kvd = h * hd, kv * hd
    t = {"model.embed_tokens.weight": rng.normal(size=(v, d)), "model.norm.weight": rng.normal(size=d)}
    if not tied:
        t["lm_head.weight"] = rng.normal(size=(v, d))
    for i in range(c["num_hidden_layers"]):
        p = f"model.layers.{i}."
        for name, shape in (("self_attn.q_proj", (qd, d)), ("self_attn.k_proj", (kvd, d)), ("self_attn.v_proj", (kvd, d)),
                            ("self_attn.o_proj", (d, qd)), ("mlp.gate_proj", (f, d)), ("mlp.up_proj", (f, d)),
                            ("mlp.down_proj", (d, f)), ("input_layernorm", (d,)), ("post_attention_layernorm", (d,))):
            t[p + name + ".weight"] = rng.normal(size=shape) * 0.1
        if model_type == "qwen3":
            t[p + "self_attn.q_norm.weight"] = rng.uniform(0.5, 1.5, size=(hd,))
            t[p + "self_attn.k_norm.weight"] = rng.uniform(0.5, 1.5, size=(hd,))
        if qkv_bias:
            t[p + "self_attn.q_proj.bias"] = rng.normal(size=(qd,)) * 0.05
            t[p + "self_attn.k_proj.bias"] = rng.normal(size=(kvd,)) * 0.05
            t[p + "self_attn.v_proj.bias"] = rng.normal(size=(kvd,)) * 0.05
    with open(directory / "config.json", "w") as fh:
        json.dump(c, fh)
    names = sorted(t)  # split across two shards + index.json like real checkpoints
    s1, s2 = "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"
    _write_safetensors(directory / s1, {n: t[n] for n in names[::2]}, dtype)
    _write_safetensors(directory / s2, {n: t[n] for n in names[1::2]}, dtype)
    weight_map = {n: (s1 if idx % 2 == 0 else s2) for idx, n in enumerate(names)}
    with open(directory / "model.safetensors.index.json", "w") as fh:
        json.dump({"metadata": {}, "weight_map": weight_map}, fh)
    return t


@pytest.mark.parametrize("tied, dtype", [(False, "F32"), (True, "BF16")])
def test_safetensors_conversion_layout(tmp_path, tied, dtype):
    hf = _random_hf_checkpoint(tmp_path, tied=tied, dtype=dtype)
    mesh = create_mesh(ParallelConfig(fsdp=2, tp=2), jax.devices()[:4])
    params, cfg = load_hf_llama(str(tmp_path), mesh)
    cast = (lambda a: a.astype(ml_dtypes.bfloat16).astype(np.float32)) if dtype == "BF16" else (lambda a: a.astype(np.float32))
    d, h, hd = cfg.hidden_size, cfg.num_attention_heads, cfg.head_dim
    np.testing.assert_array_equal(np.asarray(params.embed), cast(hf["model.embed_tokens.weight"]))
    head = hf["model.embed_tokens.weight"] if tied else hf["lm_head.weight"]
    np.testing.assert_array_equal(np.asarray(params.lm_head), cast(head).T)
    wq = cast(hf["model.layers.2.self_attn.q_proj.weight"]).T.reshape(d, h, hd)
    np.testing.assert_array_equal(np.asarray(params.blocks.wq[2]), wq)
    np.testing.assert_array_equal(np.asarray(params.blocks.w_down[3]), cast(hf["model.layers.3.mlp.down_proj.weight"]).T)
    assert params.blocks.wq.sharding.spec == P("pp", "fsdp", "tp", None)  # stacked layers stay sharded


def test_rejects_unsupported_architectures(tmp_path):
    with open(tmp_path / "config.json", "w") as fh:
        json.dump(dict(HF_CONFIG, model_type="mamba"), fh)
    with pytest.raises(ValueError, match="model_type"):
        hf_model_config(str(tmp_path))
    with open(tmp_path / "config.json", "w") as fh:
        json.dump(dict(HF_CONFIG, rope_scaling={"rope_type": "yarn", "factor": 4.0}), fh)
    with pytest.raises(ValueError, match="rope_scaling"):
        hf_model_config(str(tmp_path))


def _numpy_hf_reference(t: dict, c: dict, tokens: np.ndarray) -> np.ndarray:
    """Independent float64 NumPy port of Llama / Qwen3 / Qwen2.5 forward, on raw HF tensors."""
    d, h, kv = c["hidden_size"], c["num_attention_heads"], c["num_key_value_heads"]
    hd = c.get("head_dim") or (d // h)
    T = tokens.shape[1]
    inv_freq = 1.0 / (c["rope_theta"] ** (np.arange(0, hd, 2) / hd))
    rs = c.get("rope_scaling")
    if rs:  # transformers' _compute_llama3_parameters
        low_w, high_w = rs["original_max_position_embeddings"] / rs["low_freq_factor"], \
            rs["original_max_position_embeddings"] / rs["high_freq_factor"]
        wavelen = 2 * np.pi / inv_freq
        llama = np.where(wavelen > low_w, inv_freq / rs["factor"], inv_freq)
        smooth = (rs["original_max_position_embeddings"] / wavelen - rs["low_freq_factor"]) / \
            (rs["high_freq_factor"] - rs["low_freq_factor"])
        smoothed = (1 - smooth) * llama / rs["factor"] + smooth * llama
        medium = ~(wavelen < high_w) & ~(wavelen > low_w)
        inv_freq = np.where(medium, smoothed, llama)
    emb = np.concatenate([np.outer(np.arange(T), inv_freq)] * 2, axis=-1)
    cos, sin = np.cos(emb), np.sin(emb)

    def rotate_half(x):
        return np.concatenate([-x[..., hd // 2:], x[..., :hd // 2]], axis=-1)

    def norm(x, w):
        return w * x / np.sqrt((x ** 2).mean(-1, keepdims=True) + c["rms_norm_eps"])

    W = lambda n: t[n].astype(np.float64)
    x = W("model.embed_tokens.weight")[tokens]
    mask = np.triu(np.full((T, T), -np.inf), 1)
    for i in range(c["num_hidden_layers"]):
        p = f"model.layers.{i}."
        y = norm(x, W(p + "input_layernorm.weight"))
        q = y @ W(p + "self_attn.q_proj.weight").T
        k = y @ W(p + "self_attn.k_proj.weight").T
        v = y @ W(p + "self_attn.v_proj.weight").T
        if p + "self_attn.q_proj.bias" in t:
            q = q + W(p + "self_attn.q_proj.bias")
            k = k + W(p + "self_attn.k_proj.bias")
            v = v + W(p + "self_attn.v_proj.bias")
        q = q.reshape(*y.shape[:2], h, hd)
        k = k.reshape(*y.shape[:2], kv, hd)
        v = v.reshape(*y.shape[:2], kv, hd)
        if p + "self_attn.q_norm.weight" in t:
            q = norm(q, W(p + "self_attn.q_norm.weight"))
            k = norm(k, W(p + "self_attn.k_norm.weight"))
        q, k, v = q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3)
        q, k = q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
        k, v = np.repeat(k, h // kv, axis=1), np.repeat(v, h // kv, axis=1)  # repeat_kv
        s = q @ k.transpose(0, 1, 3, 2) / np.sqrt(hd) + mask
        s = np.exp(s - s.max(-1, keepdims=True))
        a = (s / s.sum(-1, keepdims=True)) @ v
        x = x + a.transpose(0, 2, 1, 3).reshape(*x.shape[:2], h * hd) @ W(p + "self_attn.o_proj.weight").T
        y = norm(x, W(p + "post_attention_layernorm.weight"))
        g = y @ W(p + "mlp.gate_proj.weight").T
        x = x + ((g / (1 + np.exp(-g))) * (y @ W(p + "mlp.up_proj.weight").T)) @ W(p + "mlp.down_proj.weight").T
    x = norm(x, W("model.norm.weight"))
    head = t["lm_head.weight"] if "lm_head.weight" in t else t["model.embed_tokens.weight"]
    return x @ head.astype(np.float64).T


@pytest.mark.parametrize("variant", ["untied", "tied", "llama3_rope", "qwen3", "qwen2"])
@pytest.mark.parametrize("layout", [dict(), dict(tp=2, pp=2), dict(fsdp=2, cp=2)])
def test_logits_match_numpy_reference(tmp_path, variant, layout):
    model_type = "qwen3" if variant == "qwen3" else ("qwen2" if variant == "qwen2" else "llama")
    head_dim = 32 if variant == "qwen3" else None  # explicit head_dim != hidden_size // heads (64 // 4 = 16)
    qkv_bias = variant == "qwen2"
    hf = _random_hf_checkpoint(tmp_path, tied=variant == "tied", model_type=model_type,
                               head_dim=head_dim, qkv_bias=qkv_bias)
    with open(tmp_path / "config.json") as fh:
        c = json.load(fh)
    if variant == "llama3_rope":
        c.update(rope_theta=500000.0, rope_scaling={"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
                                                    "high_freq_factor": 4.0, "original_max_position_embeddings": 8})
        with open(tmp_path / "config.json", "w") as fh:
            json.dump(c, fh)
    tokens = np.random.default_rng(1).integers(0, c["vocab_size"], (2, 16))
    expected = _numpy_hf_reference(hf, c, tokens)

    pcfg = ParallelConfig(pp_microbatches=1, **layout)
    mesh = create_mesh(pcfg, jax.devices()[:pcfg.world_size])
    params, cfg_m = load_hf_llama(str(tmp_path), mesh)
    _, static, _ = abstract_model(cfg_m, mesh)
    ctx = make_context(Config(parallel=pcfg, model=cfg_m, train=TrainConfig(dtype="float32", remat="none")), mesh)
    logits = jax.jit(lambda p, t: forward(eqx.combine(p, static), t, ctx))(params, jax.numpy.asarray(tokens, np.int32))
    np.testing.assert_allclose(np.asarray(logits), expected, rtol=1e-4, atol=1e-4)