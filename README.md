# Jaxotron 🚀

A highly-scalable, educational, and production-ready JAX adaptation of Hugging Face's Nanotron, heavily inspired by the Google DeepMind ["How to Scale Your Model"](https://jax-ml.github.io/scaling-book/) JAX book. 

Jaxotron proves that you can build state-of-the-art 3D parallelized LLM training pipelines in a fraction of the code compared to PyTorch, all by leveraging JAX's GSPMD compiler magic and functional programming.

## ✨ Features

- **Native 3D Parallelism (DP, FSDP/ZeRO-3, TP):** Uses `jax.sharding.PartitionSpec` to abstract away manual NCCL/MPI calls. The XLA compiler automatically inserts `AllGather`, `ReduceScatter`, and `AllReduce` collectives perfectly.
- **Pure Equinox Modules:** Clean, OOP-style PyTrees using `equinox`, filtering static configurations from traceable arrays out of the box.
- **Spectral muTransfer (muP):** Automatically scales learning rates and initialization bounds depending on width/depth ratios using Greg Yang's muTransfer rules.
- **Asynchronous Orbax Checkpointing:** Fully non-blocking, multi-host PyTree checkpointing using the latest Google `orbax` APIs.
- **Micro-batched Gradient Accumulation:** A memory-efficient `jax.lax.scan` loop that cleanly calculates updates without blowing out accelerator memory.
- **Exact LLaMA Math:** Faithfully replicates Meta's original Rotary Positional Embeddings (RoPE), SwiGLU gated MLPs, and Grouped-Query Attention (GQA).

## 🧠 How the Sharding Works
Following the *DeepMind Scaling Book*, matrix multiplication with partitioned arrays boils down to specific layout rules. By giving JAX semantic mesh names `("dp", "fsdp", "tp")`, we tell the compiler exactly how we want memory distributed:

- **Fully Sharded Data Parallelism (ZeRO-3):** We partition both the micro-batch and the weight's input features over the `"fsdp"` axis. XLA detects the conflict and automatically injects an `AllGather` for the weights before the forward pass, and a `ReduceScatter` for the gradients in the backward pass.
- **Tensor Parallelism (TP):** We partition the column-linear out-features and row-linear in-features over the `"tp"` axis. XLA performs a local matrix multiplication and automatically injects an `AllReduce` (or `ReduceScatter`) to synchronize the heads!

## 🚀 Quick Start

1. **Install Dependencies:**
   ```bash
   pip install jax[tpu] equinox optax orbax-checkpoint wandb numpy torch
   ```
2. **Run Jaxotron:**
   ```bash
   python app.py
   ```
   *(Note: The `app.py` script will automatically generate a dummy `.bin` dataset and detect your available GPU/TPU devices to initialize the JAX mesh).*

## 📚 Acknowledgments
- The [Google DeepMind Team](https://jax-ml.github.io/scaling-book/) for demystifying TPU scaling math.
- The [Hugging Face Nanotron](https://github.com/huggingface/nanotron) team for the inspiration behind a hyper-minimalist distributed LLM trainer.
- Patrick Kidger for [Equinox](https://github.com/patrick-kidger/equinox).