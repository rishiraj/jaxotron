"""Strongly-typed configuration for Jaxotron.

This module deliberately does NOT import JAX, so the CLI can set XLA flags
(for example simulated CPU devices) before JAX initialises its backends.
"""
from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ParallelConfig:
    """Sizes of each axis of the device mesh. Their product must equal the device count.

    dp   : data parallel        (replicated weights, sharded batch)
    fsdp : fully sharded DP     (ZeRO-3: weights/grads/optimizer state sharded, sharded batch)
    pp   : pipeline parallel    (layers split into stages)
    cp   : context parallel     (sequence split across devices)
    tp   : tensor parallel      (attention heads / MLP columns split across devices)
    """

    dp: int = 1
    fsdp: int = 1
    pp: int = 1
    cp: int = 1
    tp: int = 1
    pp_microbatches: int = 0  # 0 = auto (equal to micro_batch_size)

    @property
    def world_size(self) -> int:
        return self.dp * self.fsdp * self.pp * self.cp * self.tp

    @property
    def data_parallel_size(self) -> int:
        return self.dp * self.fsdp


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 1024
    hidden_size: int = 256
    intermediate_size: int = 768
    num_hidden_layers: int = 4
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    init_std: float = 0.02

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


# Llama-style architectures (trained from scratch).
MODEL_PRESETS: dict[str, ModelConfig] = {
    "tiny": ModelConfig(),
    "125m": ModelConfig(vocab_size=49152, hidden_size=768, intermediate_size=2048,
                        num_hidden_layers=12, num_attention_heads=12, num_key_value_heads=4),
    "smollm2-1.7b": ModelConfig(vocab_size=49152, hidden_size=2048, intermediate_size=8192,
                                num_hidden_layers=24, num_attention_heads=32, num_key_value_heads=32,
                                rope_theta=130000.0),
    "llama3.2-1b": ModelConfig(vocab_size=128256, hidden_size=2048, intermediate_size=8192,
                               num_hidden_layers=16, num_attention_heads=32, num_key_value_heads=8,
                               rope_theta=500000.0),
    "llama3.1-8b": ModelConfig(vocab_size=128256, hidden_size=4096, intermediate_size=14336,
                               num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=8,
                               rope_theta=500000.0),
}


@dataclass
class TrainConfig:
    global_batch_size: int = 32        # sequences per optimizer step
    micro_batch_size: int = 4          # sequences per data-parallel replica per forward/backward
    sequence_length: int = 256
    train_steps: int = 300
    learning_rate: float = 1e-3
    min_lr_ratio: float = 0.1
    warmup_steps: int = 30
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95
    seed: int = 42
    dtype: str = "auto"                # auto | bfloat16 | float32  (compute dtype; params stay float32)
    remat: str = "full"                # full | dots | none
    attention_impl: str = "reference"  # reference (portable, supports CP) | cudnn (GPU fused attention)
    log_every: int = 10
    eval_every: int = 0                # 0 disables evaluation
    eval_batches: int = 10
    save_every: int = 500
    keep_checkpoints: int = 3


@dataclass
class Config:
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data_path: str = "data/synthetic/train.bin"
    val_path: Optional[str] = None
    checkpoint_dir: str = "checkpoints/run"
    run_name: str = "jaxotron"
    wandb_project: Optional[str] = None
    distributed: bool = False
    profile_dir: Optional[str] = None
    dry_run: bool = False
    show_sharding: bool = False

    # ---------------------------------------------------------------- derived
    @property
    def grad_accum_steps(self) -> int:
        t, p = self.train, self.parallel
        return t.global_batch_size // (t.micro_batch_size * p.data_parallel_size)

    @property
    def pp_microbatches(self) -> int:
        if self.parallel.pp == 1:
            return 1
        return self.parallel.pp_microbatches or self.train.micro_batch_size

    # ------------------------------------------------------------- validation
    def validate(self, num_devices: int) -> None:
        p, m, t = self.parallel, self.model, self.train
        errors = []
        if p.world_size != num_devices:
            errors.append(f"dp*fsdp*pp*cp*tp = {p.world_size} but {num_devices} devices are available.")
        if m.hidden_size % m.num_attention_heads:
            errors.append("hidden_size must be divisible by num_attention_heads.")
        if m.head_dim % 2:
            errors.append("head_dim must be even for RoPE.")
        if m.num_attention_heads % m.num_key_value_heads:
            errors.append("num_attention_heads must be divisible by num_key_value_heads (GQA).")
        if m.num_key_value_heads % p.tp:
            errors.append(f"num_key_value_heads ({m.num_key_value_heads}) must be divisible by tp ({p.tp}).")
        if m.num_hidden_layers % p.pp:
            errors.append(f"num_hidden_layers ({m.num_hidden_layers}) must be divisible by pp ({p.pp}).")
        if t.sequence_length % p.cp:
            errors.append(f"sequence_length ({t.sequence_length}) must be divisible by cp ({p.cp}).")
        if t.global_batch_size % (t.micro_batch_size * p.data_parallel_size):
            errors.append("global_batch_size must be divisible by micro_batch_size * dp * fsdp.")
        if p.pp > 1 and t.micro_batch_size % self.pp_microbatches:
            errors.append(f"micro_batch_size ({t.micro_batch_size}) must be divisible by "
                          f"pp_microbatches ({self.pp_microbatches}).")
        if t.attention_impl not in ("reference", "cudnn"):
            errors.append("attention_impl must be 'reference' or 'cudnn'.")
        if t.attention_impl == "cudnn" and p.cp > 1:
            errors.append("attention_impl='cudnn' does not support context parallelism; use 'reference'.")
        if t.dtype not in ("auto", "bfloat16", "float32"):
            errors.append("dtype must be auto, bfloat16 or float32.")
        if t.remat not in ("full", "dots", "none"):
            errors.append("remat must be full, dots or none.")
        if errors:
            raise ValueError("Invalid configuration:\n  - " + "\n  - ".join(errors))

    # ---------------------------------------------------------- serialisation
    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def save_json(self, path: str) -> None:
        from etils import epath  # installed with orbax; supports local paths and gs://
        p = epath.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        d = dict(d)
        return cls(
            parallel=ParallelConfig(**d.pop("parallel")),
            model=ModelConfig(**d.pop("model")),
            train=TrainConfig(**d.pop("train")),
            **d,
        )

    @classmethod
    def load_json(cls, path: str) -> "Config":
        from etils import epath
        return cls.from_dict(json.loads(epath.Path(path).read_text()))


def _parse_value(raw: str, current: Any) -> Any:
    if isinstance(current, bool):
        return raw.lower() in ("1", "true", "yes", "on")
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if current is None:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw


def apply_override(cfg: Config, key: str, raw: str) -> None:
    """Apply an override such as ``train.learning_rate=3e-4`` or ``model.rope_theta=5e5``."""
    section, _, name = key.partition(".")
    if not name:
        setattr(cfg, section, _parse_value(raw, getattr(cfg, section)))
        return
    obj = getattr(cfg, section)
    if not hasattr(obj, name):
        raise AttributeError(f"Unknown config key: {key}")
    value = _parse_value(raw, getattr(obj, name))
    if obj.__dataclass_params__.frozen:
        setattr(cfg, section, dataclasses.replace(obj, **{name: value}))
    else:
        setattr(obj, name, value)
