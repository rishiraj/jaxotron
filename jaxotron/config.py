"""Strongly-typed configuration for Jaxotron.

This module deliberately does NOT import JAX, so the CLI can set XLA flags
(for example simulated CPU devices) before JAX initialises its backends.
"""
from __future__ import annotations

import dataclasses
import json
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
    sp   : sequence parallel    (with tp > 1: residual stream & norms sharded along sequence over tp)
    """

    dp: int = 1
    fsdp: int = 1
    pp: int = 1
    cp: int = 1
    tp: int = 1
    sp: bool = False          # Megatron-style sequence parallelism over the tp axis
    pp_microbatches: int = 0  # 0 = auto (equal to micro_batch_size)

    @property
    def world_size(self) -> int:
        return self.dp * self.fsdp * self.pp * self.cp * self.tp

    @property
    def data_parallel_size(self) -> int:
        return self.dp * self.fsdp

    @property
    def sequence_shard_size(self) -> int:
        return self.cp * (self.tp if self.sp else 1)

    def describe(self) -> str:
        active = []
        for n in ("dp", "fsdp", "pp", "cp", "tp"):
            val = getattr(self, n)
            if val > 1:
                label = f"{n.upper()}={val}"
                if n == "tp" and self.sp:
                    label += "(+SP)"
                active.append(label)
        return " x ".join(active) if active else "single device"


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
    # Llama 3.1+ "llama3" RoPE frequency scaling. A factor of 0 disables it.
    rope_scaling_factor: float = 0.0
    rope_low_freq_factor: float = 1.0
    rope_high_freq_factor: float = 4.0
    rope_original_max_position: int = 8192
    # Qwen3 / Qwen2.5 extensions (0 / False = standard Llama / SmolLM2):
    attn_head_dim: int = 0   # 0 = hidden_size // num_attention_heads (Qwen3 uses 128)
    qk_norm: bool = False    # Per-head RMSNorm on Q and K before RoPE (Qwen3)
    qkv_bias: bool = False   # Bias on Q, K, V projections (Qwen2 / Qwen2.5)

    @property
    def head_dim(self) -> int:
        return self.attn_head_dim or (self.hidden_size // self.num_attention_heads)


_LLAMA3_ROPE = dict(rope_theta=500000.0, rope_low_freq_factor=1.0, rope_high_freq_factor=4.0,
                    rope_original_max_position=8192)

# Llama, SmolLM2, Qwen3, and Qwen2.5 architectures. Shapes match public Hugging Face checkpoints,
# so `--init_from_hf` checkpoints and from-scratch runs share the exact same code path.
MODEL_PRESETS: dict[str, ModelConfig] = {
    "tiny": ModelConfig(),
    "125m": ModelConfig(vocab_size=49152, hidden_size=768, intermediate_size=2048,
                        num_hidden_layers=12, num_attention_heads=12, num_key_value_heads=4),
    "smollm2-135m": ModelConfig(vocab_size=49152, hidden_size=576, intermediate_size=1536,
                                num_hidden_layers=30, num_attention_heads=9, num_key_value_heads=3,
                                rope_theta=100000.0),
    "smollm2-360m": ModelConfig(vocab_size=49152, hidden_size=960, intermediate_size=2560,
                                num_hidden_layers=32, num_attention_heads=15, num_key_value_heads=5,
                                rope_theta=100000.0),
    "smollm2-1.7b": ModelConfig(vocab_size=49152, hidden_size=2048, intermediate_size=8192,
                                num_hidden_layers=24, num_attention_heads=32, num_key_value_heads=32,
                                rope_theta=130000.0),
    "qwen2.5-0.5b": ModelConfig(vocab_size=151936, hidden_size=896, intermediate_size=4864,
                                num_hidden_layers=24, num_attention_heads=14, num_key_value_heads=2,
                                rms_norm_eps=1e-6, rope_theta=1000000.0, qkv_bias=True),
    "qwen2.5-1.5b": ModelConfig(vocab_size=151936, hidden_size=1536, intermediate_size=8960,
                                num_hidden_layers=28, num_attention_heads=12, num_key_value_heads=2,
                                rms_norm_eps=1e-6, rope_theta=1000000.0, qkv_bias=True),
    "qwen3-0.6b": ModelConfig(vocab_size=151936, hidden_size=1024, intermediate_size=3072,
                              num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8,
                              attn_head_dim=128, rms_norm_eps=1e-6, rope_theta=1000000.0, qk_norm=True),
    "qwen3-1.7b": ModelConfig(vocab_size=151936, hidden_size=2048, intermediate_size=6144,
                              num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8,
                              attn_head_dim=128, rms_norm_eps=1e-6, rope_theta=1000000.0, qk_norm=True),
    "qwen3-4b": ModelConfig(vocab_size=151936, hidden_size=2560, intermediate_size=9728,
                            num_hidden_layers=36, num_attention_heads=32, num_key_value_heads=8,
                            attn_head_dim=128, rms_norm_eps=1e-6, rope_theta=1000000.0, qk_norm=True),
    "qwen3-8b": ModelConfig(vocab_size=151936, hidden_size=4096, intermediate_size=12288,
                            num_hidden_layers=36, num_attention_heads=32, num_key_value_heads=8,
                            attn_head_dim=128, rms_norm_eps=1e-6, rope_theta=1000000.0, qk_norm=True),
    "llama3.2-1b": ModelConfig(vocab_size=128256, hidden_size=2048, intermediate_size=8192,
                               num_hidden_layers=16, num_attention_heads=32, num_key_value_heads=8,
                               rope_scaling_factor=32.0, **_LLAMA3_ROPE),
    "llama3.2-3b": ModelConfig(vocab_size=128256, hidden_size=3072, intermediate_size=8192,
                               num_hidden_layers=28, num_attention_heads=24, num_key_value_heads=8,
                               rope_scaling_factor=32.0, **_LLAMA3_ROPE),
    "llama3.1-8b": ModelConfig(vocab_size=128256, hidden_size=4096, intermediate_size=14336,
                               num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=8,
                               rope_scaling_factor=8.0, **_LLAMA3_ROPE),
}


@dataclass
class TrainConfig:
    global_batch_size: int = 32        # sequences per optimizer step
    micro_batch_size: int = 4          # sequences per data-parallel replica per forward/backward
    sequence_length: int = 256
    train_steps: int = 300
    max_tokens: Optional[int] = None   # optional token budget (Picotron style); overrides train_steps if set
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
    init_from_hf: Optional[str] = None  # Hugging Face repo id or local dir with Llama/Qwen safetensors
    run_name: str = "jaxotron"
    wandb_project: Optional[str] = None
    distributed: bool = False
    profile_dir: Optional[str] = None
    dry_run: bool = False
    show_sharding: bool = False
    show_mesh: bool = False
    show_collectives: bool = False
    peak_tflops: Optional[float] = None  # per-device peak for MFU on hardware that is not auto-detected

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

    @property
    def total_train_steps(self) -> int:
        t = self.train
        if t.max_tokens is not None and t.max_tokens > 0:
            tokens_per_step = max(1, t.global_batch_size * t.sequence_length)
            return max(1, (t.max_tokens + tokens_per_step - 1) // tokens_per_step)
        return t.train_steps

    # ------------------------------------------------------------- validation
    def validate(self, num_devices: int) -> None:
        p, m, t = self.parallel, self.model, self.train
        errors = []
        for name in ("dp", "fsdp", "pp", "cp", "tp"):
            if getattr(p, name) < 1:
                errors.append(f"{name} must be >= 1.")
        if p.world_size != num_devices:
            errors.append(f"dp*fsdp*pp*cp*tp = {p.world_size} but {num_devices} devices are available. "
                          f"(Tip: on a laptop use --cpu_devices {p.world_size}.)")
        errors += model_errors(m, p)
        if t.sequence_length % p.cp:
            errors.append(f"sequence_length ({t.sequence_length}) must be divisible by cp ({p.cp}).")
        if p.sp and t.sequence_length % p.sequence_shard_size:
            errors.append(f"sequence_length ({t.sequence_length}) must be divisible by cp * tp "
                          f"({p.sequence_shard_size}) when sequence parallelism (--sp) is enabled.")
        if t.micro_batch_size < 1 or t.global_batch_size < 1:
            errors.append("micro_batch_size and global_batch_size must be positive.")
        elif t.global_batch_size % (t.micro_batch_size * p.data_parallel_size):
            errors.append(f"global_batch_size ({t.global_batch_size}) must be divisible by "
                          f"micro_batch_size * dp * fsdp ({t.micro_batch_size * p.data_parallel_size}).")
        if p.pp > 1 and t.micro_batch_size % self.pp_microbatches:
            errors.append(f"micro_batch_size ({t.micro_batch_size}) must be divisible by "
                          f"pp_microbatches ({self.pp_microbatches}).")
        if t.attention_impl not in ("reference", "cudnn"):
            errors.append("attention_impl must be 'reference' or 'cudnn'.")
        if t.attention_impl == "cudnn" and p.cp > 1:
            errors.append("attention_impl='cudnn' does not support context parallelism; use 'reference'.")
        if t.attention_impl == "cudnn" and t.dtype == "float32":
            errors.append("attention_impl='cudnn' requires bfloat16 compute (use --dtype bfloat16 or auto).")
        if t.dtype not in ("auto", "bfloat16", "float32"):
            errors.append("dtype must be auto, bfloat16 or float32.")
        if t.remat not in ("full", "dots", "none"):
            errors.append("remat must be full, dots or none.")
        if t.train_steps < 1:
            errors.append("train_steps must be >= 1.")
        if t.log_every < 1 or t.save_every < 1:
            errors.append("log_every and save_every must be >= 1.")
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
    def from_dict(cls, d: dict[str, Any]) -> Config:
        """Build a Config, ignoring unknown keys so older/newer config.json files still load."""
        d = dict(d)
        return cls(
            parallel=_known(ParallelConfig, d.pop("parallel", {})),
            model=_known(ModelConfig, d.pop("model", {})),
            train=_known(TrainConfig, d.pop("train", {})),
            **_known_kwargs(cls, d),
        )

    @classmethod
    def load_json(cls, path: str) -> Config:
        from etils import epath
        return cls.from_dict(json.loads(epath.Path(path).read_text()))


def model_errors(m: ModelConfig, p: ParallelConfig) -> list[str]:
    """Shape constraints between the model and the mesh (shared by training and generation)."""
    errors = []
    if m.attn_head_dim == 0 and m.hidden_size % m.num_attention_heads:
        errors.append("hidden_size must be divisible by num_attention_heads.")
    if m.head_dim <= 0 or m.head_dim % 2:
        errors.append("head_dim must be positive and even for RoPE.")
    if m.num_attention_heads % m.num_key_value_heads:
        errors.append("num_attention_heads must be divisible by num_key_value_heads (GQA).")
    if m.num_key_value_heads % p.tp:
        errors.append(f"num_key_value_heads ({m.num_key_value_heads}) must be divisible by tp ({p.tp}).")
    if m.intermediate_size % p.tp:
        errors.append(f"intermediate_size ({m.intermediate_size}) must be divisible by tp ({p.tp}).")
    if m.vocab_size % p.tp:
        errors.append(f"vocab_size ({m.vocab_size}) must be divisible by tp ({p.tp}).")
    if m.hidden_size % p.fsdp:
        errors.append(f"hidden_size ({m.hidden_size}) must be divisible by fsdp ({p.fsdp}).")
    if m.num_hidden_layers % p.pp:
        errors.append(f"num_hidden_layers ({m.num_hidden_layers}) must be divisible by pp ({p.pp}).")
    return errors


def _known_kwargs(cls, d: dict) -> dict:
    names = {f.name for f in dataclasses.fields(cls)}
    return {k: v for k, v in d.items() if k in names}


def _known(cls, d: dict):
    return cls(**_known_kwargs(cls, d))


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
        if not hasattr(cfg, section):
            raise AttributeError(f"Unknown config key: {key}")
        setattr(cfg, section, _parse_value(raw, getattr(cfg, section)))
        return
    obj = getattr(cfg, section, None)
    if obj is None or not dataclasses.is_dataclass(obj) or not hasattr(obj, name):
        raise AttributeError(f"Unknown config key: {key}")
    value = _parse_value(raw, getattr(obj, name))
    if obj.__dataclass_params__.frozen:
        setattr(cfg, section, dataclasses.replace(obj, **{name: value}))
    else:
        setattr(obj, name, value)
