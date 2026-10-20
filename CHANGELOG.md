# Changelog

## 0.4.0

### Added
- **Qwen3 & Qwen2.5 architecture support** alongside Llama 2/3.x and SmolLM2 (`model_type` in `llama`, `qwen3`, `qwen2`):
  - Explicit `head_dim` independent of `hidden_size // num_attention_heads` (needed by `Qwen3-0.6B` and `Qwen3-4B`).
  - Optional per-head RMSNorm on queries and keys (`qk_norm=True` for Qwen3).
  - Optional QKV attention biases (`qkv_bias=True` for Qwen2.5).
  - Presets `qwen3-0.6b`, `qwen3-1.7b`, `qwen3-4b`, and `qwen2.5-0.5b`.
- **Megatron-style Sequence Parallelism (`--sp`)**: shards the residual stream and RMSNorm activations along the sequence axis across `(cp, tp)`, converting tensor-parallel `all-reduce` into `all-gather` + `reduce-scatter` via a single logical-axis rule.
- **KV-cached autoregressive generation** in `jaxotron-generate` (with `--no_kv_cache` fallback and tok/s reporting), supporting single-device, FSDP, and tensor-parallel inference.
- **Live 4D scorecard CLI (`jaxotron-compare` / `scripts/compare_4d.py`)**: compiles and runs 8 parallelism layouts on 16 simulated CPU devices (or real accelerators) and prints a single comparison table of per-device memory, XLA collectives, and numerical parity (`max |grad diff|`).
- **Picotron-compatible workflows**: `--config path/to/config.json` to load a full experiment config, `--max_tokens` stopping budget, `trained_tokens` counter, and per-device peak memory (`GiB`) reporting on GPU/TPU.
- **Slice-aware safetensors loading**: `model.safetensors.index.json` shard resolution and per-slice memmap loading so pipeline stages only read the layers they own.
- **JAX 0.9+ forward compatibility**: explicit `AxisType.Auto` mesh construction and broader XLA/Shardy HLO collective parsing.

### Fixed
- Fixed throughput and MFU calculation skew when `--eval_every` triggers between `--log_every` intervals.
- Excluded QK-norm weights and attention biases from AdamW weight decay.
- Handled newer XLA/Shardy `mesh[...]` replica group syntax with unquoted or multi-axis specifications.

## 0.3.0

### Added
- `--show_collectives`: the all-reduces, all-gathers and collective-permutes XLA inserted, attributed to mesh axes.
- `--show_mesh`: the mesh coordinates of every device.
- Hugging Face Llama-family weights: `--init_from_hf` for training and `--hf` for generation (safetensors, no PyTorch).
- Llama 3.1 RoPE scaling; presets `smollm2-135m`, `smollm2-360m`, `llama3.2-3b`; `--list_presets`.
- Parallel inference in `generate` (`--tp`, `--pp`, `--cp`, `--cpu_devices`), streaming output, top-p, EOS stopping.
- `jaxotron-train`, `jaxotron-generate`, `jaxotron-prepare-data` console scripts.
- Persistent XLA compilation cache (`--compile_cache`) and `scripts/demo_4d.sh --warmup`.
- Ctrl+C saves a checkpoint; NaN guard; `--peak_tflops`; MFU table for B200, H100 PCIe, MI300X, TPU v6e.
- CI (lint, latest and minimum JAX, Python 3.13, demo smoke test), PyPI release workflow, tour notebook.

### Fixed
- Multi-host loader: each host reads only the rows its devices own.
- Profiler trace is always stopped, even for short runs.
- Saving a step twice no longer errors; data files and `meta.json` are written atomically.
- `config.json` from older or newer versions loads (unknown keys are ignored).
- More shape constraints validated up front (`vocab_size % tp`, `intermediate_size % tp`, `hidden_size % fsdp`).

### Changed
- The train step is compiled ahead of time; compile time is reported separately from throughput.

## 0.2.0
- DP, FSDP, TP, PP and CP for Llama-style models, Orbax checkpoints, deterministic data loader, demo scripts.