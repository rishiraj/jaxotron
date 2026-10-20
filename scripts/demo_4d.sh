#!/usr/bin/env bash
# Live demo for talks: 6 acts on one laptop CPU with simulated devices.
#
#   bash scripts/demo_4d.sh            # interactive: press Enter between acts
#   bash scripts/demo_4d.sh --warmup   # run every act once, non-interactively, before the talk
#                                      # (fills the XLA compilation cache, downloads SmolLM2)
#   DEMO_OFFLINE=1 bash scripts/demo_4d.sh   # skip act 6 (needs internet + jaxotron[hf])
set -euo pipefail
cd "$(dirname "$0")/.."

WARMUP=0
[[ "${1:-}" == "--warmup" ]] && WARMUP=1
OUT=${DEMO_DIR:-/tmp/jaxotron-demo}
SMALL="--seq_len 128 --global_batch_size 16 --micro_batch_size 4 --log_every 5"

pause() {
  echo
  echo "▶ $1"
  if [[ $WARMUP == 0 ]]; then read -rp "  [Enter] "; fi
  echo
}

# Each act starts from a clean checkpoint directory so the demo is repeatable.
rm -rf "$OUT"

pause "Act 1: one device. The baseline."
python train.py --cpu_devices 1 $SMALL --steps 20 --checkpoint_dir "$OUT/1dev"

pause "Act 2: FSDP across 8 devices. Watch MB/device shrink 8x in the sharding table."
python train.py --cpu_devices 8 --fsdp 8 $SMALL --micro_batch_size 2 --steps 1 \
  --show_sharding --dry_run --checkpoint_dir "$OUT/fsdp"

pause "Act 3: tensor + context parallel. Same math, different layout. Look at the communication."
python train.py --cpu_devices 4 --tp 2 --cp 2 $SMALL --steps 20 --show_collectives \
  --checkpoint_dir "$OUT/tpcp"

pause "Act 4: full 4D. DP x PP x CP x TP on 16 devices."
python train.py --cpu_devices 16 --dp 2 --pp 2 --cp 2 --tp 2 $SMALL --steps 40 \
  --show_mesh --show_collectives --checkpoint_dir "$OUT/4d"

pause "Act 5: load the 16-device checkpoint on ONE device and generate."
python generate.py --cpu --checkpoint_dir "$OUT/4d" --prompt "5 8 11 14" --max_new_tokens 16 --temperature 0

if [[ "${DEMO_OFFLINE:-0}" != "1" ]]; then
  pause "Act 6: a real pretrained model (SmolLM2-135M), tensor-parallel over 3 devices."
  python generate.py --hf HuggingFaceTB/SmolLM2-135M --cpu_devices 3 --tp 3 \
    --prompt "The three laws of motion are" --max_new_tokens 40 --temperature 0 \
    || echo "(Act 6 skipped: needs internet and pip install \"jaxotron[hf]\")"
fi

echo
if [[ $WARMUP == 1 ]]; then
  echo "Warm-up done. The compilation cache is filled; run without --warmup on stage."
else
  echo "That's 4D parallelism in JAX. ⭐ github.com/rishiraj/jaxotron"
fi
