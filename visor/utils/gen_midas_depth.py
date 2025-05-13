#!/usr/bin/env python3
"""
batch-generate MiDaS depth maps (.npy) + debug PNGs
for the dual-billboard prism pipeline.

"""
from visor import PROJECT_ROOT, RENDERS_DIR
# ╭─── USER KNOBS ──────────────────────────────────────────────────────────╮

VIEWS_JSONL     = RENDERS_DIR / "views.jsonl"      # set to None to scan folder recursively
MODEL_TYPE      = "DPT_Large"        # "DPT_Large" | "DPT_Hybrid" | "MiDaS_small"
GPU_ID          = 0                  # -1 → force CPU
NUM_WORKERS     = 6                 # CPU workers that feed the GPU
WRITE_DEBUG_PNG = True               # colourised *_depth.png previews
# ╰─────────────────────────────────────────────────────────────────────────╯

# ───────────────────────── imports & global cache ─────────────────────────
import json, os, sys, multiprocessing as mp
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from tqdm import tqdm
import matplotlib.pyplot as plt

_MODEL_CACHE = {}            # (model_type, device) → (model, transform)


# ──────────────────────────── MiDaS utilities ────────────────────────────
def load_midas(device: torch.device, model_type: str):
    key = (model_type, device)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    model = torch.hub.load("isl-org/MiDaS", model_type, trust_repo=True)
    model.to(device).eval()

    midas_tfms = torch.hub.load("isl-org/MiDaS", "transforms", trust_repo=True)
    transform  = (
        midas_tfms.dpt_transform
        if "DPT" in model_type else midas_tfms.small_transform
    )
    _MODEL_CACHE[key] = (model, transform)
    return model, transform


def _process_one(job):
    """
    Worker process – runs once per image.
    """
    img_path, depth_dir, dbg_dir, gpu_id, model_type = job
    device = torch.device(f"cuda:{gpu_id}" if gpu_id >= 0 and torch.cuda.is_available()
                          else "cpu")
    model, transform = load_midas(device, model_type)

    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        print(f"⚠️  could not read {img_path}", file=sys.stderr)
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    inp     = transform(img_rgb).to(device)

    with torch.no_grad():
        pred = model(inp)                # 3-D  ➜  (B, H0, W0)   or   4-D ➜ (B,1,H0,W0)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),           # ALWAYS make it (B,1,H0,W0)
            size=img_rgb.shape[:2],      # target (H_img, W_img)
            mode="bicubic",
            align_corners=False,
        ).squeeze(1)                     # back to (B, H_img, W_img)

        depth = pred[0].cpu().numpy()    # → (H_img, W_img)


    # normalise [0,1] for storage / preview
    d_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    np.save(depth_dir / f"{img_path.stem}.npy", d_norm.astype(np.float32))

    if dbg_dir is not None:
        plt.imsave(dbg_dir / f"{img_path.stem}_depth.png", d_norm, cmap="inferno")


# ───────────────────────────────── main ───────────────────────────────────
def main(
    image_root: Path | None = None,
    views_jsonl: Path | None = None,
) -> None:
    """
    Parameters
    ----------
    image_root  : Path | None
        Folder that contains the rendered RGB images.  Defaults to `RENDERS_DIR`.
    views_jsonl : Path | None
        Path to `views.jsonl` that lists the images.  If None, the script
        will scan `image_root` recursively for *.png / *.jpg files.
    """

    # ──────────────────────────────────────────────────────────────
    # Resolve paths relative to the installed package
    # ──────────────────────────────────────────────────────────────
    root = Path(image_root or RENDERS_DIR).expanduser()
    if not root.is_dir():
        raise SystemExit(f"❌  renders directory does not exist: {root}")

    views_path = Path(views_jsonl or VIEWS_JSONL)

    # ──────────────────────────────────────────────────────────────
    # Pre-cache MiDaS weights (downloads if missing)
    # ──────────────────────────────────────────────────────────────
    print("🔄  Pre-caching MiDaS weights …")
    load_midas(torch.device("cpu"), MODEL_TYPE)

    # ──────────────────────────────────────────────────────────────
    # Discover RGB frames
    # ──────────────────────────────────────────────────────────────
    if views_path.is_file():
        with views_path.open() as f:
            img_paths = [root / json.loads(ln)["file"] for ln in f]
    else:
        # Fall back to globbing every PNG/JPG in the folder tree
        img_paths = sorted(root.rglob("*.png")) + sorted(root.rglob("*.jpg"))

    if not img_paths:
        raise SystemExit("❌  No images found to process.")

    # ──────────────────────────────────────────────────────────────
    # Prepare output directories
    # ──────────────────────────────────────────────────────────────
    depth_dir = root
    depth_dir.mkdir(parents=True, exist_ok=True) 


    dbg_dir = None
    if WRITE_DEBUG_PNG:
        dbg_dir = depth_dir / "debug"
        dbg_dir.mkdir(exist_ok=True)

    # ──────────────────────────────────────────────────────────────
    # Launch workers
    # ──────────────────────────────────────────────────────────────
    jobs = [(p, depth_dir, dbg_dir, GPU_ID, MODEL_TYPE) for p in img_paths]

    print(f"⚙️  Processing {len(jobs)} images "
          f"with {MODEL_TYPE} on GPU:{GPU_ID if GPU_ID >= 0 else 'CPU'} …")

    with mp.get_context("spawn").Pool(NUM_WORKERS) as pool:
        list(tqdm(pool.imap_unordered(_process_one, jobs), total=len(jobs)))

    print(f"✅  Depth maps saved to {depth_dir}")



if __name__ == "__main__":
    main()
