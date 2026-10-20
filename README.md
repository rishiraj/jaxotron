# Jaxotron 🚀

[![CI](https://github.com/rishiraj/jaxotron/actions/workflows/ci.yml/badge.svg)](https://github.com/rishiraj/jaxotron/actions/workflows/ci.yml)
![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)
![JAX](https://img.shields.io/badge/JAX-jit%20%2B%20GSPMD-orange)
![Parallelism](https://img.shields.io/badge/parallelism-DP%20%7C%20FSDP%20%7C%20TP%20%7C%20PP%20%7C%20CP-purple)

**Minimal, hackable 4D-parallel LLM pre-training in pure JAX.**
Data, Tensor, Pipeline and Context parallelism (plus FSDP/ZeRO-3) for Llama-style models in about 1,000 lines of readable code.

Inspired by Hugging Face's [Picotron](https://github.com/huggingface/picotron) / [Nanotron](https://github.com/huggingface/nanotron) and Google DeepMind's [How to Scale Your Model](https://jax-ml.github.io/scaling-book/).

```bash
pip install -e .
python train.py --cpu_devices 16 --dp 2 --pp 2 --cp 2 --tp 2   # 4D parallelism on your laptop
```

---

## Why Jaxotron?

| | PyTorch (Picotron-style) | Jaxotron |
|---|---|---|
| Tensor parallel | Custom `ColumnParallelLinear` / `RowParallelLinear` + autograd functions | One line per tensor: name its axes |
| Pipeline parallel | Hand-written send/recv, 1F1B schedule, separate backward | ~40 lines; `jax.grad` derives the backward pipeline |
| Context parallel | Custom ring attention with manual backward | One sharding annotation on K/V |
| FSDP / ZeRO-3 | Wrapper module, bucketing, hooks | A rule: `"embed" → "fsdp"` |
| Communication | Explicit `all_reduce` / `isend` calls | Inserted automatically by the XLA compiler |
| Runs on | GPUs | GPUs, TPUs, and CPU (simulated devices) |

Every parallel layout is **tested to produce the same loss and gradients as a single device**.

---

## ✨ Features

- **5 parallelism axes, freely combinable**: `dp × fsdp × pp × cp × tp`.
- **Llama architecture**: RoPE, RMSNorm, SwiGLU, Grouped-Query Attention. Presets for `tiny`, `125m`, `smollm2-1.7b`, `llama3.2-1b` and `llama3.1-8b` shapes.
- **Runs anywhere**: `--cpu_devices N` simulates N accelerators on a laptop, which makes it ideal for talks, classrooms and CI.
- **Sharded from birth**: parameters and optimizer state are created directly in their sharded layout, so no device ever holds the full model.
- **Topology-independent checkpoints** (Orbax, async, multi-host): train on 16 GPUs with 4D parallelism, resume or sample on 1 GPU.
- **Deterministic, resumable data**: memory-mapped token files; batch order is a pure function of the step.
- **Production basics**: bf16 mixed precision, activation rematerialisation, gradient accumulation, AdamW with warmup and cosine decay, gradient clipping, eval loop, MFU reporting, profiler traces, optional Weights & Biases.
- **Inspectable**: `--show_sharding` prints how every weight is split; `--dry_run` prints per-device memory.
- **Minimal dependencies**: `jax`, `equinox`, `optax`, `orbax-checkpoint`, `numpy`. No PyTorch.

---

## 🚀 Quick start

```bash
git clone https://github.com/rishiraj/jaxotron && cd jaxotron
pip install -e ".[dev]"            # CPU
pip install -e ".[cuda]"           # NVIDIA GPUs
pip install -e ".[tpu]"            # Cloud TPU
```

**1. Train with 4D parallelism on a laptop (16 simulated devices).** A learnable synthetic dataset is generated automatically:

```bash
python train.py --cpu_devices 16 --dp 2 --pp 2 --cp 2 --tp 2 \
    --seq_len 128 --global_batch_size 16 --micro_batch_size 4 --steps 60 --show_sharding
```

```
Mesh: {'dp': 2, 'pp': 2, 'fsdp': 1, 'cp': 2, 'tp': 2}
Pipeline bubble: 20% of pipeline time is idle (raise micro_batch_size to shrink it)
step      1 | loss 6.9893 | ... | (includes compilation)
step     30 | loss 5.3702 | ...
step     60 | loss 2.9933 | ...
```

**2. Sample from the checkpoint on a single device.** Orbax reshards automatically:

```bash
python generate.py --cpu --checkpoint_dir checkpoints/run --prompt "5 8 11 14" --temperature 0
```

**3. Run the test-suite**, which checks every layout against a single device:

```bash
pytest -q
```

> Simulated devices share your CPU cores. Keep demos small (`--seq_len 128 --global_batch_size 16`).

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
    "batch":  ("dp", "fsdp"),  # data parallel
    "seq":    "cp",            # context parallel
    "kv_seq": None,            # K/V gathered across cp
    "stage":  "pp",
    ...
}
```

Model code just labels activations:

```python
q = ctx.shard(jnp.einsum("btd,dhk->bthk", h, p.wq), "batch", "seq", "heads", None)
```

### What each strategy compiles to

These collectives were verified in the compiled HLO:

| Strategy | What is split | Communication XLA inserts | Where in the code |
|---|---|---|---|
| **DP** | batch | `all-reduce` of gradients | `"batch": ("dp", "fsdp")` |
| **FSDP** | batch + every weight | `all-gather` of weights, sharded gradient reduction | `"embed": "fsdp"` |
| **TP** | heads, MLP columns, vocab | `all-reduce` after `wo` and `w_down` | `"heads"/"mlp": "tp"` |
| **CP** | sequence | `all-gather` of K/V | `"seq": "cp"`, `"kv_seq": None` |
| **PP** | layers | `collective-permute` between stages | [`parallel/pipeline.py`](jaxotron/parallel/pipeline.py) |

### Pipeline parallelism in ~40 lines

Layers are stacked as `[num_layers, ...]` and sharded over `pp`. Inside the step they are reshaped to `[stages, layers_per_stage, ...]`. A state buffer holds one microbatch per stage. Each iteration, all stages run in parallel (`jax.vmap(..., spmd_axis_name="pp")`), then `jnp.roll` shifts activations to the next stage, which XLA lowers to a point-to-point `collective-permute`:

```mermaid
flowchart LR
  subgraph t0["iteration i"]
    S0["stage 0<br/>microbatch i"] --> S1["stage 1<br/>microbatch i-1"] --> S2["stage 2<br/>microbatch i-2"]
  end
  S2 -- "roll = collective-permute" --> next["iteration i+1"]
```

This is the GPipe schedule; the bubble fraction is `(pp - 1) / (microbatches + pp - 1)`. The backward pipeline is not written at all: `jax.grad` differentiates through the loop.

### Context parallelism

Each `cp` rank keeps its slice of queries but needs every key and value. Annotating K/V as replicated over `cp` makes XLA all-gather them, the same approach Llama 3 used for long-context training. The causal mask uses global positions, so it stays correct when the sequence is sharded.

---

## 🛠️ Usage guide

### Choosing a layout (`dp × fsdp × pp × cp × tp = #devices`)

| Hardware | Suggested layout | Why |
|---|---|---|
| 1 GPU / laptop | all `1` | baseline |
| 8× GPU node, model < ~3B | `--fsdp 8` | ZeRO-3: minimal memory, simple |
| 8× GPU node, large model | `--fsdp 4 --tp 2` | TP over NVLink cuts activation memory |
| Long sequences (32k+) | add `--cp 2/4` | splits attention memory across devices |
| Many nodes | `--dp N` across nodes, `tp`/`fsdp` within | DP traffic is once per step |
| Very deep models | add `--pp 2/4` with `dp` | splits layers. Prefer `dp` over `fsdp` with `pp` (FSDP re-gathers weights for every pipeline microbatch) |

Constraints are checked at startup with clear error messages: `num_key_value_heads % tp == 0`, `num_layers % pp == 0`, `seq_len % cp == 0`, and so on.

### Batch size and gradient accumulation

```
global_batch_size = micro_batch_size × (dp × fsdp) × grad_accum_steps
```

`--micro_batch_size` is per data-parallel replica. Accumulation is a memory-friendly `lax.scan`. With `pp > 1`, each micro batch is further split into pipeline microbatches (`--pp_microbatches`, default `micro_batch_size`).

### Real data

```bash
pip install -e ".[data]"
python prepare_data.py --dataset roneneldan/TinyStories --tokenizer HuggingFaceTB/SmolLM2-135M \
    --out_dir data/tinystories --max_docs 200000

python train.py --preset 125m --fsdp 8 --data data/tinystories/train.bin \
    --seq_len 1024 --global_batch_size 256 --micro_batch_size 8 \
    --lr 6e-4 --warmup 500 --steps 5000 --eval_every 250 --wandb_project jaxotron

python generate.py --checkpoint_dir checkpoints/run --prompt "Once upon a time"
```

`vocab_size` is inferred from `meta.json` and padded to a multiple of 128. Data files are flat `uint16`/`uint32` token arrays, memory-mapped, so terabyte-scale files open instantly.

### Useful flags

| Flag | Purpose |
|---|---|
| `--show_sharding` | table of every parameter: shape, PartitionSpec, MB per device |
| `--dry_run` | compile one step, print per-device memory, exit |
| `--profile_dir profiles/` | trace a few steps; open in TensorBoard or [Perfetto](https://ui.perfetto.dev) to *see* the collectives |
| `--dtype bfloat16` | mixed precision (default `auto`: bf16 on GPU/TPU, fp32 on CPU) |
| `--remat full\|dots\|none` | activation checkpointing policy |
| `--attention cudnn` | fused flash attention on NVIDIA GPUs (not with `cp > 1`) |
| `--set train.weight_decay=0.05` | override any config field |

### Multi-node

* **SLURM / GPUs**: [`scripts/slurm.sbatch`](scripts/slurm.sbatch) (one process per node, `--distributed`).
* **Cloud TPU pods**: [`scripts/tpu_pod.sh`](scripts/tpu_pod.sh). Use a `gs://` checkpoint directory shared by all hosts.

Checkpoints are written every `--save_every` steps. Re-running the same command resumes the exact optimizer state *and* data position, even with a different layout.

---

## 🎤 Giving a talk with Jaxotron

[`scripts/demo_4d.sh`](scripts/demo_4d.sh) is a five-act live demo that runs on one laptop, offline:

1. **Baseline** on one device.
2. **FSDP × 8**: `--show_sharding` shows every weight shrinking 8×.
3. **TP + CP**: same loss curve, different layout.
4. **Full 4D** (DP × PP × CP × TP) on 16 simulated devices.
5. **Generate** from the 16-device checkpoint on *one* device.

Tips: pre-run once so the compile cache is warm, and show `parallel/rules.py` side by side with the `--show_sharding` output.

---

## 📁 Project structure

```text
jaxotron/
├── train.py                 # CLI entry point
├── generate.py              # sampling from any checkpoint
├── prepare_data.py          # tokenise HF datasets -> .bin (or --synthetic)
├── jaxotron/
│   ├── config.py            # typed configs, presets, validation
│   ├── trainer.py           # jit-compiled train step, loop, logging, eval
│   ├── optim.py             # AdamW + schedule, sharded optimizer state
│   ├── checkpoint.py        # Orbax async, multi-host, reshardable checkpoints
│   ├── utils.py             # MFU, sharding table, logging
│   ├── models/
│   │   ├── layers.py        # RMSNorm, RoPE, causal GQA attention
│   │   └── llama.py         # Llama model, stacked layers, forward pass
│   ├── parallel/
│   │   ├── mesh.py          # 5D device mesh
│   │   ├── rules.py         # logical axis -> mesh axis rules (THE parallelism file)
│   │   ├── context.py       # activation sharding helper
│   │   └── pipeline.py      # pipeline parallelism (GPipe via vmap + roll)
│   └── data/
│       ├── loader.py        # mmap dataset, deterministic multi-host loader
│       └── synthetic.py     # learnable synthetic data for offline demos
├── tests/                   # every layout == single device; checkpoints; data
└── scripts/                 # live demo, SLURM, TPU pod
```

---

## ❓ FAQ

**Is this as fast as Megatron or MaxText?** No; it optimises for readability. It already uses the techniques that matter most: bf16, remat, sharded optimizer state, scan over layers, and compiler-scheduled collectives. Use `--attention cudnn` on NVIDIA GPUs.

**Why GPipe instead of 1F1B?** GPipe falls out of `jax.grad` with zero hand-written backward code and is stable across JAX versions. With remat and enough microbatches, the memory and bubble trade-off is acceptable for teaching and moderate scales.

**Can I load pretrained Hugging Face weights?** Not yet; Jaxotron trains from scratch. The parameter layout mirrors Llama, so a converter is a good first contribution.

---

## 🤝 Acknowledgments

- [Picotron](https://github.com/huggingface/picotron) and [Nanotron](https://github.com/huggingface/nanotron) by Hugging Face
- [How to Scale Your Model](https://jax-ml.github.io/scaling-book/) by Google DeepMind
- [MaxText](https://github.com/AI-Hypercomputer/maxtext) and the GSPMD paper for the pipelining approach
- [Equinox](https://github.com/patrick-kidger/equinox), [Optax](https://github.com/google-deepmind/optax), [Orbax](https://github.com/google/orbax)

Licensed under Apache 2.0. If you use Jaxotron in a talk or course, a ⭐ and a link back is appreciated!
