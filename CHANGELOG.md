# Changelog

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
