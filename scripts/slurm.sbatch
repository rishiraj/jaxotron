#!/bin/bash
# Multi-node GPU training with SLURM. One process per node; JAX sees all local GPUs.
# Submit: sbatch scripts/slurm.sbatch
# Prepare data once on shared storage first:
#   jaxotron-prepare-data --dataset roneneldan/TinyStories --out_dir data/tinystories
#SBATCH --job-name=jaxotron
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
mkdir -p logs
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
export JAXOTRON_CHECKPOINT_DIR=${JAXOTRON_CHECKPOINT_DIR:-checkpoints/125m}

# 16 GPUs: dp crosses nodes (cheap, once per step); fsdp and tp (+sp) stay inside a node (NVLink).
# Re-submitting the same job resumes from the latest checkpoint.
srun python train.py --distributed \
  --preset 125m --data data/tinystories/train.bin \
  --dp 2 --fsdp 4 --tp 2 --sp \
  --seq_len 2048 --global_batch_size 256 --micro_batch_size 8 \
  --dtype bfloat16 --attention cudnn \
  --lr 6e-4 --warmup 500 --steps 10000 --eval_every 500 --save_every 1000 \
  --checkpoint_dir "$JAXOTRON_CHECKPOINT_DIR" --wandb_project jaxotron