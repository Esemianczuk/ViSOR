#!/usr/bin/env bash
set -euo pipefail

source "/home/eric/Documents/ViSOR/_launch_common.sh"

RUN_NAME="$(consume_run_name_arg "$@")"
if [[ -n "$RUN_NAME" ]]; then
  shift
fi

activate_visor_env

RUN_DIR="$(resolve_train_run_dir "$RUN_NAME")"
RUN_NAME="$(basename "$RUN_DIR")"
prepare_run_dir "$RUN_DIR"
print_run_summary "$RUN_DIR"

CHECKPOINT="$RUN_DIR/checkpoint.pt"
DIAG_JSONL="$RUN_DIR/diag.jsonl"
TRAIN_LOG="$RUN_DIR/logs/train.log"

echo "Checkpoint: $CHECKPOINT"
echo "Diagnostics: $DIAG_JSONL"
echo "Train log: $TRAIN_LOG"

python -s -m visor.train \
  --renders-dir "$ROOT/renders1" \
  --checkpoint "$CHECKPOINT" \
  --extra-steps 2000 \
  --batch 2048 \
  --chunk 8 \
  --tau-weight 0 \
  --sh_file_front '' \
  --diag-every 100 \
  --diag-jsonl "$DIAG_JSONL" \
  --slab-splats 48 \
  --slab-strength-weight 0.0005 \
  --slab-gain-weight 0.3 \
  --slab-opacity-weight 0.001 \
  --non-slab-lr-scale 1.0 \
  --slab-lr-scale 1.5 \
  --slab-ramp-warmup-steps 400 \
  --slab-ramp-steps 1200 \
  --slab-ramp-start-scale 0.10 \
  --slab-ramp-head-div-threshold 0.08 \
  --slab-ramp-spread-threshold 0.40 \
  "$@" 2>&1 | tee -a "$TRAIN_LOG"
