#!/usr/bin/env bash
# Live demo for DevFest / conference talks: 7 acts on one laptop CPU with simulated devices.
#
#   bash scripts/demo_4d.sh            # interactive: press Enter between acts
#   bash scripts/demo_4d.sh --warmup   # run every act once, non-interactively, before the talk
#                                      # (fills the XLA compilation cache, downloads SmolLM2)
#   DEMO_OFFLINE=1 bash scripts/demo_4d.sh   # skip act 7 (needs internet + jaxotron[hf])
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

pause "Act 2: FSDP (ZeRO-3) across 8 devices. Watch MB/device shrink 8x in the sharding table."
python train.py --cpu_devices 8 --fsdp 8 $SMALL --micro_batch_size 2 --steps 1 \
  --show_sharding --dry_run --checkpoint_dir "$OUT/fsdp"

pause "Act 3: Tensor Parallel vs Tensor + Sequence Parallel (--sp). Watch all-reduce become all-gather + reduce-scatter."
python train.py --cpu_devices 4 --tp 4 --sp $SMALL --steps 1 --dry_run --show_collectives \
  --checkpoint_dir "$OUT/tpsp"

pause "Act 4: Tensor + Context Parallel. Same math, different layout. Look at the communication."
python train.py --cpu_devices 4 --tp 2 --cp 2 $SMALL --steps 20 --show_collectives \
  --checkpoint_dir "$OUT/tpcp"

pause "Act 5: Full 4D (DP x PP x CP x TP) on 16 simulated devices + side-by-side scorecard."
python train.py --cpu_devices 16 --dp 2 --pp 2 --cp 2 --tp 2 $SMALL --steps 40 \
  --show_mesh --show_collectives --checkpoint_dir "$OUT/4d"
python scripts/compare_4d.py --cpu_devices 16

pause "Act 6: Load the 16-device checkpoint on ONE device and generate with KV-cache."
python generate.py --cpu --checkpoint_dir "$OUT/4d" --prompt "5 8 11 14" --max_new_tokens 16 --temperature 0

if [[ "${DEMO_OFFLINE:-0}" != "1" ]]; then
  pause "Act 7: A real pretrained model (SmolLM2-135M), tensor-parallel KV-cached inference over 3 devices."
  python generate.py --hf HuggingFaceTB/SmolLM2-135M --cpu_devices 3 --tp 3 \
    --prompt "The three laws of motion are" --max_new_tokens 40 --temperature 0 \
    || echo "(Act 7 skipped: needs internet and pip install \"jaxotron[hf]\")"
fi

echo
if [[ $WARMUP == 1 ]]; then
  echo "Warm-up done. The compilation cache is filled; run without --warmup on stage."
else
  echo "That's 4D parallelism in JAX. ⭐ github.com/rishiraj/jaxotron"
fi