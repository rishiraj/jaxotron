#!/usr/bin/env bash
# Cloud TPU pod slice (e.g. v5e-16 = 4 hosts x 4 chips). Run the same command on every host:
#   gcloud compute tpus tpu-vm ssh $TPU_NAME --zone $ZONE --worker=all \
#     --command "cd jaxotron && CHECKPOINT_DIR=gs://my-bucket/jaxotron/125m bash scripts/tpu_pod.sh"
# Requirements:
#   * the dataset exists at the same local path on every host (this script prepares it), and
#   * CHECKPOINT_DIR is shared storage visible to all hosts (a GCS bucket).
# Single-host TPUs (v5e-8, Kaggle) do not need --distributed; see the README.
set -euo pipefail
CHECKPOINT_DIR=${CHECKPOINT_DIR:-gs://YOUR_BUCKET/jaxotron/125m}
if [[ "$CHECKPOINT_DIR" == *YOUR_BUCKET* ]]; then
  echo "Set CHECKPOINT_DIR to a gs:// path shared by all hosts." >&2
  exit 1
fi

pip install -q -e ".[tpu,data]"
[[ -f data/tinystories/train.bin ]] || \
  python prepare_data.py --dataset roneneldan/TinyStories --out_dir data/tinystories --max_docs 500000

python train.py --distributed \
  --preset 125m --data data/tinystories/train.bin \
  --fsdp 8 --tp 2 \
  --seq_len 2048 --global_batch_size 256 --micro_batch_size 16 \
  --lr 6e-4 --warmup 500 --steps 10000 --eval_every 500 \
  --checkpoint_dir "$CHECKPOINT_DIR"
