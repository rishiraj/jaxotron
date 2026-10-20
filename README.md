# Jaxotron 🚀

[![CI](https://github.com/rishiraj/jaxotron/actions/workflows/ci.yml/badge.svg)](https://github.com/rishiraj/jaxotron/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/jaxotron.svg)](https://pypi.org/project/jaxotron/)
![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue)
![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)
![JAX](https://img.shields.io/badge/JAX-jit%20%2B%20GSPMD-orange)
![Parallelism](https://img.shields.io/badge/parallelism-DP%20%7C%20FSDP%20%7C%20TP%2FSP%20%7C%20PP%20%7C%20CP-purple)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rishiraj/jaxotron/blob/main/notebooks/jaxotron_tour.ipynb)
[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/rishiraj/jaxotron/blob/main/notebooks/jaxotron_tour.ipynb)

**Minimal, hackable 4D-parallel LLM pre-training and inference in pure JAX.**
Data, Tensor (+ Sequence), Pipeline, and Context parallelism (plus FSDP/ZeRO-3) for **Llama 3.x**, **Qwen3**, **Qwen2.5**, and **SmolLM2** models in ~2,500 lines of readable code (the parallelism core in parallel/ is under 400). Runs on a laptop CPU (with simulated devices), a multi-GPU node, or a TPU pod, and loads real Hugging Face `.safetensors` weights with zero PyTorch dependency.

Inspired by Hugging Face's [Picotron](https://github.com/huggingface/picotron) / [Nanotron](https://github.com/huggingface/nanotron) and Google DeepMind's [How to Scale Your Model](https://jax-ml.github.io/scaling-book/).

```bash
pip install jaxotron
jaxotron-train --cpu_devices 16 --dp 2 --pp 2 --cp 2 --tp 2 --show_collectives   # 4D parallelism on your laptop
jaxotron-compare --cpu_devices 16                                                # 8-layout parity & collective scorecard
```

---

## Why Jaxotron?

| Capability | PyTorch ([Picotron](https://github.com/huggingface/picotron)-style) | Jaxotron (JAX + GSPMD) |
|---|---|---|
| **Tensor parallel (TP)** | Custom `ColumnParallelLinear` / `RowParallelLinear` + autograd functions | One line per tensor: name its logical axes |
| **Sequence parallel (TP+SP)** | Custom LayerNorm hooks, manual `all_gather` / `reduce_scatter` autograd | Toggle `--sp`: `"res_seq": ("cp", "tp")` |
| **Pipeline parallel (PP)** | Hand-written send/recv, AFAB / 1F1B schedules, separate backward | ~40 lines; `jax.grad` derives the backward pipeline automatically |
| **Context parallel (CP)** | Custom ring attention with manual backward pass | One sharding annotation on K/V (`"kv_seq": None`) |
| **FSDP / ZeRO-3** | Wrapper module, manual gradient bucketing, post-backward hooks | A single rule: `"embed": "fsdp"` |
| **Communication** | Explicit `dist.all_reduce` / `batch_isend_irecv` calls | Inserted by the XLA compiler; `--show_collectives` shows every op |
| **Checkpoints** | One file per `(tp, pp)` rank; requires identical topology to resume | Topology-independent (Orbax): train on 16 devices, resume on 2, sample on 1 |
| **Runs on** | GPUs (or slow `gloo` processes) | GPUs, TPUs, and single-process simulated CPU devices (`--cpu_devices N`) |

Every parallel layout is **tested to produce the same loss and gradients to float32 round-off (< 1e‑5) as a single device**, and pretrained checkpoints (`Llama`, `SmolLM2`, `Qwen3`, `Qwen2.5`) are **tested to produce the exact same logits as reference implementations**.

---

## ✨ Features

- **5 parallelism axes + Sequence Parallelism, freely combinable**: `dp × fsdp × pp × cp × tp (+ sp)`.
- **See the communication**: `--show_collectives` lists every `all-reduce`, `all-gather`, `reduce-scatter`, and `collective-permute` XLA inserted, attributed to the mesh axis it runs over.
- **One-command 4D scorecard**: `jaxotron-compare` compiles 8 layouts on 16 simulated CPU devices and verifies that every strategy matches single-device gradients to `< 1e-5` while showing per-device memory and collectives.
- **Pretrained Hugging Face weights**: `--init_from_hf HuggingFaceTB/SmolLM2-135M` or `Qwen/Qwen3-0.6B` (or any Llama 2 / 3.x, SmolLM2, Qwen3, or Qwen2.5 checkpoint) streams `.safetensors` straight into the sharded layout. No PyTorch required.
- **Modern architecture support**: RoPE (including Llama 3.1 scaling), RMSNorm, SwiGLU, Grouped-Query Attention, explicit `head_dim` + QK-Norm (`Qwen3`), and QKV biases (`Qwen2.5`). Presets: `tiny`, `125m`, `smollm2-135m/360m/1.7b`, `qwen3-0.6b/1.7b/4b`, `qwen2.5-0.5b`, `llama3.2-1b/3b`, `llama3.1-8b`.
- **Fast KV-cached parallel generation**: `jaxotron-generate` runs KV-cached autoregressive decoding across single-device or tensor-parallel (`--tp`) meshes with live token streaming.
- **Runs anywhere**: `--cpu_devices N` simulates N accelerators on a laptop—ideal for DevFests, university lectures, and CI. Free Kaggle TPU v5e-8 and Colab work out of the box.
- **Sharded from birth**: parameters and AdamW state are allocated directly in their sharded layout, so no device ever holds the full unsharded model.
- **Topology-independent checkpoints** (Orbax, async, multi-host): train with 4D parallelism, resume or sample on any other layout. Ctrl+C saves a checkpoint, and re-running resumes seamlessly.
- **Production essentials**: bf16 mixed precision, activation rematerialisation (`full`, `dots`, `none`), gradient accumulation via `lax.scan`, AdamW with warmup and cosine decay, JSON `--config` files, `--max_tokens` budget, MFU & device memory reporting, Perfetto profiler traces, NaN guard, persistent XLA compilation cache, and optional Weights & Biases.

---

## 🚀 Quick start

```bash
pip install jaxotron                 # CPU (laptops, Colab, CI)
pip install "jaxotron[cuda]"         # NVIDIA GPUs (CUDA 12; use [cuda13] for CUDA 13)
pip install "jaxotron[tpu]"          # Cloud TPU / Kaggle TPU
pip install "jaxotron[hf,data]"      # + Hugging Face weights, tokenizers and datasets
```

From source: `git clone https://github.com/rishiraj/jaxotron && cd jaxotron && pip install -e ".[dev]"`. Every `jaxotron-*` command is also available as `python train.py`, `python generate.py`, `python prepare_data.py`, and `python scripts/compare_4d.py`.

**1. Train with 4D parallelism on a laptop (16 simulated devices).** A learnable synthetic dataset is generated automatically:

```bash
jaxotron-train --cpu_devices 16 --dp 2 --pp 2 --cp 2 --tp 2 \
    --seq_len 128 --global_batch_size 16 --micro_batch_size 4 --steps 60
```

```text
Mesh: {'dp': 2, 'pp': 2, 'fsdp': 1, 'cp': 2, 'tp': 2}  ->  DP=2 x PP=2 x CP=2 x TP=2
Pipeline bubble: 20% of pipeline time is idle (raise micro_batch_size / --pp_microbatches to shrink it)
Compiled the train step in 19.3s
step      1 | loss 6.9893 | ...
step     30 | loss 5.3702 | ...
step     60 | loss 2.9933 | ...
```

**2. See the communication XLA inserted** (compile only, no training):

```bash
jaxotron-train --cpu_devices 16 --dp 2 --pp 2 --cp 2 --tp 2 --seq_len 128 \
    --global_batch_size 16 --micro_batch_size 4 --dry_run --show_collectives
```

```text
collective           mesh axes   ops   bytes (static)
-----------------------------------------------------
all-reduce           cp            5         2.57 MiB
all-reduce           dp            4         2.50 MiB
all-reduce           pp            4       384.07 KiB
all-reduce           tp            9       771.07 KiB
all-gather           cp            4       128.00 KiB
collective-permute   cp            1             32 B
collective-permute   pp            5       320.00 KiB
collective-permute   pp+cp+tp      2         1.00 MiB
```

Add `--sp` (Sequence Parallelism) with `--tp` and watch the tensor-parallel `all-reduce` turn into `all-gather` + `reduce-scatter` over `tp`!

**3. Run the 8-layout scorecard (`jaxotron-compare`)**:

```bash
jaxotron-compare --cpu_devices 16
```

```text
strategy             devices  KB/device  loss     max |Δgrad|  XLA collectives (mesh axis)
------------------------------------------------------------------------------------------
1-device baseline    1           1344.2  4.86102  0.00e+00     none
DP = 4               4           1344.2  4.86102  1.19e-07     all-reduce(dp)x12
FSDP = 4 (ZeRO-3)    4            336.6  4.86102  2.38e-07     all-gather(fsdp)x21, reduce-scatter(fsdp)x10
TP = 4               4            340.2  4.86102  2.38e-07     all-reduce(tp)x17, all-gather(tp)x2
TP = 4 + SP          4            340.2  4.86102  2.38e-07     all-gather(tp)x9, reduce-scatter(tp)x6
CP = 4 (seq split)   4           1344.2  4.86102  1.79e-07     all-reduce(cp)x14, all-gather(cp)x4
PP = 4 (GPipe)       4            478.2  4.86102  1.19e-07     all-reduce(pp)x4, collective-permute(pp)x5
4D (2x2x2x2)         16           369.2  4.86102  2.98e-07     all-reduce(dp/pp/cp/tp), all-gather(cp), collective-permute(pp)
```

**4. Sample from the 16-device checkpoint on a single device with KV caching.** Orbax reshards automatically:

```bash
jaxotron-generate --cpu --checkpoint_dir checkpoints/run --prompt "5 8 11 14" --temperature 0
```

**5. Run real pretrained SmolLM2 or Qwen3 models with tensor-parallel inference**:

```bash
pip install "jaxotron[hf]"
jaxotron-generate --hf HuggingFaceTB/SmolLM2-135M --cpu_devices 3 --tp 3 --prompt "Gravity is"
jaxotron-generate --hf Qwen/Qwen3-0.6B --cpu_devices 4 --tp 4 --prompt "Parallelism in JAX is"
```

---

## 🧠 How 4D parallelism works in Jaxotron

### The mesh

All devices form a 5D grid. The innermost axis gets the fastest links, so `tp` (which communicates every layer) sits innermost and `dp` (once per step) sits outermost:

```python
# jaxotron/parallel/mesh.py
MESH_AXES = ("dp", "pp", "fsdp", "cp", "tp")
```

### Logical axes: the one file that decides everything

Every tensor dimension has a *name*. [`parallel/rules.py`](jaxotron/parallel/rules.py) maps names to mesh axes, and XLA's GSPMD compiler inserts the communication:

```python
WEIGHT_RULES = {
    "layers": "pp",    # pipeline: each stage owns a block of layers
    "embed":  "fsdp",  # FSDP/ZeRO-3: hidden dim of every weight is sharded
    "heads":  "tp",    # tensor parallel: attention heads split
    "mlp":    "tp",    # tensor parallel: MLP hidden split
    "vocab":  "tp",
}
ACTIVATION_RULES = {
    "batch":   ("dp", "fsdp"),  # data parallel
    "seq":     "cp",            # context parallel
    "res_seq": "cp",            # or ("cp", "tp") when --sp is enabled!
    "kv_seq":  None,            # K/V gathered across cp
    "stage":   "pp",
    ...
}
```

### What each strategy compiles to

| Strategy | What is split | Communication XLA inserts | Where in the code |
|---|---|---|---|
| **DP** | batch | `all-reduce` of gradients | `"batch": ("dp", "fsdp")` |
| **FSDP** | batch + every weight | `all-gather` of weights, `reduce-scatter` of gradients | `"embed": "fsdp"` |
| **TP** | heads, MLP columns, vocab | `all-reduce` after `wo` and `w_down` | `"heads"/"mlp": "tp"` |
| **TP + SP** | TP + residual sequence | `all-gather` before QKV/MLP, `reduce-scatter` after `wo`/`w_down` | `--sp` (`"res_seq": ("cp", "tp")`) |
| **CP** | sequence | `all-gather` of K/V | `"seq": "cp"`, `"kv_seq": None` |
| **PP** | layers | `collective-permute` between stages | [`parallel/pipeline.py`](jaxotron/parallel/pipeline.py) |

---

## 🎤 Giving a talk at DevFest with Jaxotron

[`scripts/demo_4d.sh`](scripts/demo_4d.sh) is a 7-act live demo that runs on one laptop, offline (Act 7 is optional and uses Hugging Face weights):

1. **Baseline** on one device.
2. **FSDP × 8**: `--show_sharding` shows every weight and AdamW state shrinking 8×.
3. **TP vs TP + SP (`--sp`)**: watch XLA replace `all-reduce` with `all-gather` + `reduce-scatter`.
4. **TP + CP**: same loss curve, different layout, and `--show_collectives` shows the `all-gather` of K/V over `cp`.
5. **Full 4D** (`DP × PP × CP × TP`) on 16 simulated devices + live `jaxotron-compare` scorecard.
6. **KV-cached Generate** from the 16-device checkpoint on *one* device.
7. **Pretrained SmolLM2-135M / Qwen3-0.6B** with tensor-parallel KV-cached inference.

Speaker tips:
- Run `bash scripts/demo_4d.sh --warmup` before your talk to pre-populate `~/.cache/jaxotron/xla` and cache pretrained weights so every act starts instantaneously on stage.
- Keep [`parallel/rules.py`](jaxotron/parallel/rules.py) open side-by-side with your terminal: change one rule, rerun `--dry_run --show_collectives`, and show the audience how the compiler changes the communication schedule.

---

## 🙏 Acknowledgments

- [Picotron](https://github.com/huggingface/picotron) and [Nanotron](https://github.com/huggingface/nanotron) by Hugging Face
- [How to Scale Your Model](https://jax-ml.github.io/scaling-book/) by Google DeepMind
- [MaxText](https://github.com/AI-Hypercomputer/maxtext) and the GSPMD paper for the pipelining approach
- [Equinox](https://github.com/patrick-kidger/equinox), [Optax](https://github.com/google-deepmind/optax), [Orbax](https://github.com/google/orbax)

Licensed under Apache 2.0. Citation metadata is in [CITATION.cff](CITATION.cff).