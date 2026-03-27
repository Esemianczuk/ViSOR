#!/usr/bin/env bash
set -euo pipefail

source "/home/eric/Documents/ViSOR/_launch_common.sh"

RUN_NAME="$(consume_run_name_arg "$@")"
if [[ -n "$RUN_NAME" ]]; then
  shift
fi

activate_visor_env

RUN_DIR="$(resolve_consumer_run_dir "$RUN_NAME")"
ensure_run_exists "$RUN_DIR"
mkdir -p "$RUN_DIR/watch/history" "$RUN_DIR/logs"
print_run_summary "$RUN_DIR"

CHECKPOINT="$RUN_DIR/checkpoint.pt"
WATCH_OUT="$RUN_DIR/watch/latest.png"
WATCH_HISTORY="$RUN_DIR/watch/history"
WATCH_LOG="$RUN_DIR/logs/watch.log"

echo "Watching checkpoint: $CHECKPOINT"
echo "Latest preview: $WATCH_OUT"
echo "Watch history: $WATCH_HISTORY"

python -s -m visor.watch_training_progress \
  --checkpoint "$CHECKPOINT" \
  --renders-dir "$ROOT/renders1" \
  --split heldout \
  --holdout-mode phi_bucket \
  --holdout-every 8 \
  --holdout-offset 0 \
  --pred full \
  --poll-seconds 15 \
  --output "$WATCH_OUT" \
  --history-dir "$WATCH_HISTORY" \
  "$@" 2>&1 | tee -a "$WATCH_LOG"
