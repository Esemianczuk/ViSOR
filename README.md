# ViSOR

<p align="center">
  <img src="docs/demo.gif" alt="ViSOR demo" width="100%">
</p>

ViSOR is a compact view-synthesis project built around learned image-forming sheets instead of a full scene volume. The current version keeps the sheet-based idea, then strengthens it with tri-probe rear transport and a lightweight disagreement-aware Gaussian attenuation slab between the sheets.

## Current Model

```text
camera rays
   |
   v
front sheet
  - front color
  - front transmission / blocking
   |
   v
rear transport
  - 3 rear probes on the rear sheet
  - adaptive routing between rear heads
   |
   v
intra-slab Gaussian attenuation
  - sparse residual attenuation only where needed
   |
   v
final composite
```

This keeps the representation compact while still letting the renderer handle harder cases like partial occlusion, thin structure, and between-view ambiguity.

## Repo Contents

- `visor/train.py`: main training loop for the current sheet + transport + slab model
- `visor/viewer.py`: interactive viewer with adaptive lower-res motion rendering
- `visor/analyze_three_ray.py`: exact-pose train / heldout evaluator and ablation runner
- `visor/watch_training_progress.py`: live preview watcher for checkpoints
- `Blender/random.py`: Blender-side dataset generator
- `visor/utils/`: older helper tools for MiDaS depth and SH baking; still available but no longer required for the main training path

## Quick Start

### 1. Create the environment

Use the included environment file:

```bash
micromamba create -f environment.yml
```

or:

```bash
conda env create -f environment.yml
```

Then activate it:

```bash
export MAMBA_ROOT_PREFIX="$HOME/.local/share/visor-micromamba"
eval "$($HOME/.local/bin/micromamba shell hook -s bash)"
micromamba activate visor-cu118
export PYTHONNOUSERSITE=1
```

If you prefer a one-shot helper:

```bash
bash ./setup.sh
```

### 2. Install ViSOR

```bash
pip install -e .
```

### 3. Optional: native tiny-cuda-nn

ViSOR runs with the pure PyTorch fallback, but the native `tiny-cuda-nn` bindings are faster when your PyTorch and CUDA toolchain match:

```bash
pip install ninja cmake
pip install --no-build-isolation \
  "git+https://github.com/NVlabs/tiny-cuda-nn.git@v1.7#subdirectory=bindings/torch"
```

## Running the Project

### Train / watch / view

The simplest loop is:

```bash
bash ./run_train.sh
```

In another terminal:

```bash
bash ./watch_train.sh
```

When you want to inspect the latest checkpoint interactively:

```bash
bash ./view_latest.sh
```

These launchers write outputs into `runs/<run_name>/` and keep `runs/latest` pointed at the active run so you can train continuously and always inspect the newest checkpoint.

`view_latest.sh` expects a checkpoint under `runs/latest/`, so the most common flow is train first, then open the viewer.

The watcher updates:

```text
runs/latest/watch/latest.png
```

There is also a more detailed local command note at [RUN_TRAINING_WATCH_VIEW.txt](RUN_TRAINING_WATCH_VIEW.txt).

### Launch the viewer directly

```bash
python -s -m visor.viewer \
  --renders-dir ./renders \
  --checkpoint /path/to/checkpoint.pt \
  --sh-file-front '' \
  --gate-temperature 3 \
  --hard-gate-temperature 0.75 \
  --adaptive-router-strength 1.0
```

Controls:

- `W/S`: forward/back
- `A/D`: strafe left/right
- `Q/E`: down/up
- mouse drag: orbit
- wheel: dolly
- `R`: reset

## Data Layout

- `renders/`: small tracked demo dataset from the original repo
- `renders1/`: larger local dataset used during current development; if present, ViSOR prefers it automatically

You can override the default dataset location with either:

- `VISOR_RENDERS_DIR=/path/to/renders`
- `--renders-dir /path/to/renders`

## Current Training Direction

The newest repo state is centered on:

- real ray-to-plane intersections
- front-to-back attenuation compositing
- tri-probe rear transport on the rear sheet
- adaptive routing driven by probe disagreement
- a disagreement-aware sparse Gaussian slab between the sheets
- longer scheduled scratch / continuation runs through the launcher scripts

This is intentionally not a NeRF clone. The goal is to stay compact, interpretable, and fast while still improving fidelity.

## Packaging and CLI

Editable install exposes:

- `visor-train`
- `visor-view`
- `visor-watch`
- `visor-bake`
- `visor-depth`

## Notes

- `runs/`, `long_runs/`, checkpoints, and experiment dumps are intentionally ignored in git.
- The repo keeps the older SH / MiDaS tooling around for experimentation, but the current core training path does not depend on them.

