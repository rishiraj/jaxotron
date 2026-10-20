#!/usr/bin/env bash
# A live-demo script for talks: 5 acts, all on one laptop CPU with simulated devices.
# Usage: bash scripts/demo_4d.sh            (press Enter between acts)
set -euo pipefail
cd "$(dirname "$0")/.."

SMALL="--seq_len 128 --global_batch_size 16 --micro_batch_size 4 --log_every 5"
pause() { echo; read -rp "▶ $1  [Enter]"; echo; }

pause "Act 1: one device. The baseline."
python train.py --cpu_devices 1 $SMALL --steps 20 --checkpoint_dir /tmp/demo/1dev

pause "Act 2: FSDP across 8 devices. Watch MB/device shrink 8x in the sharding table."
python train.py --cpu_devices 8 --fsdp 8 $SMALL --micro_batch_size 2 --steps 1 --show_sharding --dry_run --checkpoint_dir /tmp/demo/fsdp

pause "Act 3: tensor + context parallel. Same math, different layout."
python train.py --cpu_devices 4 --tp 2 --cp 2 $SMALL --steps 20 --checkpoint_dir /tmp/demo/tpcp

pause "Act 4: full 4D. DP x PP x CP x TP on 16 devices."
python train.py --cpu_devices 16 --dp 2 --pp 2 --cp 2 --tp 2 $SMALL --steps 40 --show_sharding \
  --checkpoint_dir /tmp/demo/4d

pause "Act 5: load the 16-device checkpoint on ONE device and generate."
python generate.py --cpu --checkpoint_dir /tmp/demo/4d --prompt "5 8 11 14" --max_new_tokens 16 --temperature 0
