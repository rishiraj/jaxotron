#!/usr/bin/env bash
# Cloud TPU pod slice (e.g. v5e-16 = 4 hosts x 4 chips). Run the same command on every host:
#   gcloud compute tpus tpu-vm ssh $TPU_NAME --zone $ZONE --worker=all \
#     --command "cd jaxotron && bash scripts/tpu_pod.sh"
# Requirements:
#   * the dataset exists at the same local path on every host (run prepare_data.py on each), and
#   * CHECKPOINT_DIR is shared storage visible to all hosts (a GCS bucket).
set -euo pipefail
CHECKPOINT_DIR=${CHECKPOINT_DIR:-gs://YOUR_BUCKET/jaxotron/125m}

pip install -q -e ".[tpu,data]"
python train.py --distributed \
  --preset 125m --data data/tinystories/train.bin \
  --fsdp 8 --tp 2 \
  --seq_len 2048 --global_batch_size 256 --micro_batch_size 16 \
  --lr 6e-4 --warmup 500 --steps 10000 \
  --checkpoint_dir "$CHECKPOINT_DIR"
