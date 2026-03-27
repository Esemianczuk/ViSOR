#!/usr/bin/env bash

ROOT="/home/eric/Documents/ViSOR"
RUNS_DIR="$ROOT/runs"
DEFAULT_RUN="tri_transport_slab_sched_scratch_2000"
MAMBA_ROOT="/home/eric/.local/share/visor-micromamba"
MAMBA_BIN="/home/eric/.local/bin/micromamba"
MAMBA_ENV="visor-cu118"

activate_visor_env() {
  mkdir -p "$RUNS_DIR"
  local had_u=0
  case $- in
    *u*) had_u=1; set +u ;;
  esac
  export MAMBA_ROOT_PREFIX="$MAMBA_ROOT"
  eval "$("$MAMBA_BIN" shell hook -s bash)"
  micromamba activate "$MAMBA_ENV"
  if [[ "$had_u" -eq 1 ]]; then
    set -u
  fi
  export PYTHONNOUSERSITE=1
  cd "$ROOT"
}

timestamp_run_name() {
  date +"run_%Y%m%d_%H%M%S"
}

consume_run_name_arg() {
  if [[ $# -gt 0 && -n "${1:-}" && "${1:0:1}" != "-" ]]; then
    printf '%s\n' "$1"
  else
    printf '\n'
  fi
}

resolve_train_run_dir() {
  local requested="$1"
  if [[ -n "$requested" ]]; then
    if [[ "$requested" == "latest" && -L "$RUNS_DIR/latest" ]]; then
      readlink -f "$RUNS_DIR/latest"
    else
      printf '%s\n' "$RUNS_DIR/$requested"
    fi
    return 0
  fi
  if [[ -L "$RUNS_DIR/latest" ]]; then
    readlink -f "$RUNS_DIR/latest"
  else
    printf '%s\n' "$RUNS_DIR/$(timestamp_run_name)"
  fi
}

resolve_consumer_run_dir() {
  local requested="$1"
  if [[ -n "$requested" ]]; then
    if [[ "$requested" == "latest" ]]; then
      printf '%s\n' "$RUNS_DIR/latest"
    else
      printf '%s\n' "$RUNS_DIR/$requested"
    fi
    return 0
  fi
  printf '%s\n' "$RUNS_DIR/latest"
}

prepare_run_dir() {
  local run_dir="$1"
  mkdir -p "$run_dir/watch/history" "$run_dir/analysis" "$run_dir/logs"
  ln -sfn "$(basename "$run_dir")" "$RUNS_DIR/latest"
}

ensure_run_exists() {
  local run_dir="$1"
  if [[ ! -e "$run_dir" ]]; then
    echo "Missing run directory: $run_dir" >&2
    echo "Start a run first with ./run_train.sh [run_name]" >&2
    exit 1
  fi
}

print_run_summary() {
  local run_dir="$1"
  echo "Run directory: $run_dir"
  echo "Latest symlink: $RUNS_DIR/latest"
}
