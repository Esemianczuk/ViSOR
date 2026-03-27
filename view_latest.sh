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
print_run_summary "$RUN_DIR"

CHECKPOINT="$RUN_DIR/checkpoint.pt"
echo "Viewer checkpoint: $CHECKPOINT"

python -s -m visor.viewer \
  --renders-dir "$ROOT/renders1" \
  --checkpoint "$CHECKPOINT" \
  --sh-file-front '' \
  --gate-temperature 3 \
  --hard-gate-temperature 0.75 \
  --adaptive-router-strength 1.0 \
  "$@"
