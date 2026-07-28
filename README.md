# Jaxotron 🚀

![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)
![JAX](https://img.shields.io/badge/JAX-GSPMD-orange)
![Equinox](https://img.shields.io/badge/Equinox-Ready-brightgreen)

**Jaxotron** is a highly-scalable, educational, and production-ready JAX adaptation of Hugging Face's Nanotron, heavily inspired by the Google DeepMind ["How to Scale Your Model"](https://jax-ml.github.io/scaling-book/) JAX book.

Jaxotron proves that you can build state-of-the-art 3D parallelized LLM training pipelines in a fraction of the code compared to PyTorch, all by leveraging JAX's GSPMD compiler magic and functional programming.

---

## ✨ Key Features

- **Native 3D Parallelism:** Seamlessly mix Data Parallelism (DP), Fully Sharded Data Parallelism (FSDP/ZeRO-3), and Tensor Parallelism (TP). JAX's XLA compiler automatically inserts `AllGather` and `ReduceScatter` collectives based on semantic mesh rules.
- **Exact LLaMA Architecture:** Faithfully replicates Meta's original Rotary Positional Embeddings (RoPE), SwiGLU gated MLPs, and Grouped-Query Attention (GQA). Perfect for training models like **SmolLM**.
- **Spectral muTransfer (muP):** Automatically scales learning rates and initialization bounds depending on width/depth ratios using Greg Yang's muTransfer rules. Train a proxy model and perfectly transfer hyperparameters to scale.
- **Memory-Efficient Micro-batching:** A `jax.lax.scan` loop cleanly calculates gradient accumulations without blowing out accelerator HBM memory.
- **Modern Infrastructure:** Uses `equinox` for clean PyTree OOP modules, `optax` for optimization, `orbax` for async multi-host checkpointing, and `wandb` for logging.

---

## 📁 Project Structure

```text
jaxotron/
├── app.py                  # Entry point for training
├── config.py               # Strongly typed dataclasses for topology/model/training
├── data/
│   ├── collator.py         # Asynchronous JAX data prefetcher
│   └── nanoset.py          # Fast memory-mapped binary dataset reader
├── models/
│   ├── attention.py        # GQA Attention & SwiGLU MLP
│   ├── layers.py           # RMSNorm & Exact RoPE math
│   └── llama.py            # Main LlamaForCausalLM model
├── optim/
│   ├── build.py            # Optax AdamW + LR Schedulers
│   └── mup.py              # Spectral muP initialization & LR multipliers
├── parallel/
│   ├── mesh.py             # JAX device mesh creation
│   ├── partitioning.py     # Megatron/FSDP logical axis rules
│   └── sharding_utils.py   # PyTree node sharding applier
└── serialize/
    └── orbax_utils.py      # Async PyTree saving/loading
```

---

## 🚀 Quick Start (Testing the code)

You can run Jaxotron out-of-the-box on a laptop, a single GPU, or a massive TPU pod.

**1. Install Dependencies**
```bash
# For NVIDIA GPUs
pip install "jax[cuda12]" equinox optax orbax-checkpoint wandb numpy torch

# For TPUs
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html equinox optax orbax-checkpoint wandb numpy torch
```

**2. Run the Test Script**
```bash
python app.py
```
> **What this does:** `app.py` automatically generates a 2MB dummy `.bin` dataset to test the dataloader. It detects your hardware, sets up an FSDP mesh equal to your device count, and begins training a small toy model.

---

## 🛠️ How to Use Jaxotron (Advanced Guide)

### 1. Bringing Your Own Dataset
Jaxotron expects data to be pre-tokenized and saved as a flat binary file of `uint16` (or `int32`) integers. This allows the `TokenizedBytesDataset` to memory-map (mmap) terabytes of data instantly without loading it into RAM.

**How to prepare your data (Python snippet):**
```python
import numpy as np
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-360M")
text_data = ["Jaxotron is fast.", "Train your own LLMs!"] # Replace with your dataset

# Tokenize and flatten
tokens = [tokenizer.encode(t) + [tokenizer.eos_token_id] for t in text_data]
flat_tokens = [item for sublist in tokens for item in sublist]

# Save to binary file
arr = np.array(flat_tokens, dtype=np.uint16)
arr.tofile("my_custom_dataset.bin")
```
Update `app.py` to point to `my_custom_dataset.bin`.

### 2. Configuring the 3D Parallel Mesh
Matrix multiplication in JAX is sharded simply by assigning logical names to axes. You control this in `config.py -> ParallelismConfig`.

*Rule of thumb: `dp * fsdp * tp` MUST equal the total number of GPUs/TPUs you have.*

* **Single GPU / Laptop (CPU):** 
  `dp=1, fsdp=1, tp=1`
* **8x A100 Node (Standard ZeRO-3):** 
  `dp=1, fsdp=8, tp=1`
* **Large Cluster (Megatron-style 3D):** 
  `dp=4, fsdp=8, tp=4` (Requires 128 devices). Tensor Parallelism (`tp`) splits the actual matrix math across devices (great for massive models).

### 3. Gradient Accumulation & Memory
If you run out of memory (OOM), you don't need to change the model size. Adjust the micro-batch size!
In `config.py`:
```python
global_batch_size = 512
micro_batch_size = 4
```
*If you have 8 GPUs using FSDP, your "per-step" batch size is `4 * 8 = 32`. Jaxotron will automatically use a memory-efficient `jax.lax.scan` loop to run `16` micro-steps (512 / 32) before updating the optimizer, keeping your memory footprint tiny.*

### 4. Toggle muTransfer (muP)
Maximal Hyperparameter Transfer (muP) allows you to find the perfect learning rate on a tiny proxy model (e.g., 50M params) and perfectly apply it to a massive model (e.g., 3B params).
* In `app.py`, set `ModelConfig(use_mup=True)`.
* Jaxotron automatically calculates Fan-In/Fan-Out, applies Spectral Standard Deviation to weights, and scales the Optax learning rates per-layer based on width/depth ratios.

---

## 🏗️ Case Study: Training a SmolLM-style Model
Inspired by the **SmolLM3 Blueprint**, Jaxotron natively supports the exact features required to train a high-performance, compact, long-context reasoning model. 

To train a model similar to SmolLM, update your `app.py` config like so:

```python
config = JaxotronConfig(
    parallelism=ParallelismConfig(
        dp=1, fsdp=jax.device_count(), tp=1 # Auto-shard across all local GPUs
    ),
    model=ModelConfig(
        vocab_size=49152,           # Standard Llama 3 / Smol tokenizer size
        hidden_size=2048,           
        intermediate_size=8192,     
        num_hidden_layers=24,       # Deep network for reasoning
        num_attention_heads=32,     
        num_key_value_heads=8,      # Enables Grouped-Query Attention (GQA)
        max_position_embeddings=8192, 
        rope_theta=100000.0,        # High theta for long-context
        use_mup=True
    ),
    training=TrainingConfig(
        global_batch_size=1024,     
        micro_batch_size=8,         
        sequence_length=4096,
        learning_rate=3e-4,
        weight_decay=0.1
    )
)
```

## 📈 Monitoring & Checkpointing
* **WandB:** If `jax.process_index() == 0` (Main node), Jaxotron automatically logs Loss, Perplexity, and Tokens/sec to Weights & Biases.
* **Orbax:** Checkpoints are saved asynchronously in PyTree format. If your training crashes, simply rerun `python app.py`. Jaxotron checks the `./checkpoints` directory and automatically resumes from the exact micro-step and optimizer state you left off at.

## 🤝 Acknowledgments
- The [Google DeepMind Team](https://jax-ml.github.io/scaling-book/) for demystifying TPU scaling math.
- The [Hugging Face Nanotron](https://github.com/huggingface/nanotron) team for the inspiration behind a hyper-minimalist distributed LLM trainer.
- Patrick Kidger for [Equinox](https://github.com/patrick-kidger/equinox).