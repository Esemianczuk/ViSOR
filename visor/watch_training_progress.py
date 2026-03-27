#!/usr/bin/env python3

import argparse
import time
from pathlib import Path

import imageio.v2 as imageio
import torch

from visor import RENDERS_DIR
from visor.analyze_three_ray import _load_records, _render_view, _score, _split_indices
from visor.gaussian_slab import GaussianSlab
from visor.train import CameraEmbed, HARD_GATE_TEMPERATURE, OcclusionSheet, RefractionSheet


PRED_KEYS = {
    "full": "pred_full",
    "front_only": "pred_front_only",
    "no_refrac": "pred_no_refrac",
    "no_slab": "pred_no_slab",
    "head_mean": "pred_head_mean",
    "head0": "pred_head0",
}


def _load_model_bundle(checkpoint: Path):
    ck = torch.load(checkpoint.expanduser(), map_location="cpu")
    front = OcclusionSheet(512, 512, False).to("cuda" if torch.cuda.is_available() else "cpu")
    rear = RefractionSheet(512, 512, False).to("cuda" if torch.cuda.is_available() else "cpu")
    slab_splats = int(ck["slab"]["mean_raw"].shape[0]) if ("slab" in ck and "mean_raw" in ck["slab"]) else 32
    slab = GaussianSlab(num_splats=slab_splats).to("cuda" if torch.cuda.is_available() else "cpu")
    cam_mlp = CameraEmbed().to("cuda" if torch.cuda.is_available() else "cpu")
    front.load_state_dict(ck["front"], strict=False)
    rear.load_state_dict(ck["rear"], strict=False)
    if "slab" in ck:
        slab.load_state_dict(ck["slab"], strict=False)
    if "camera_embed" in ck:
        cam_mlp.load_state_dict(ck["camera_embed"], strict=False)
    front.eval()
    rear.eval()
    slab.eval()
    cam_mlp.eval()
    return front, rear, slab, cam_mlp, ck.get("z_offset", 0.3), int(ck.get("global_step", 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--renders-dir", type=Path, default=RENDERS_DIR)
    ap.add_argument("--split", choices=("all", "train", "heldout"), default="heldout")
    ap.add_argument("--holdout-mode", choices=("index", "phi_bucket"), default="phi_bucket")
    ap.add_argument("--holdout-every", type=int, default=8)
    ap.add_argument("--holdout-offset", type=int, default=0)
    ap.add_argument("--view-rank", type=int, default=-1,
                    help="Index inside the chosen split. -1 uses the middle view.")
    ap.add_argument("--pred", choices=tuple(PRED_KEYS.keys()), default="full")
    ap.add_argument("--gate-temperature", type=float, default=3.0)
    ap.add_argument("--hard-gate-temperature", type=float, default=HARD_GATE_TEMPERATURE)
    ap.add_argument("--adaptive-router-strength", type=float, default=1.0)
    ap.add_argument("--poll-seconds", type=float, default=15.0)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--history-dir", type=Path, default=None)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    renders_dir = args.renders_dir.expanduser()
    recs = _load_records(renders_dir)
    for rec in recs:
        rec["_img_path"] = str(renders_dir / rec["file"])
    split_idxs = _split_indices(
        recs,
        split=args.split,
        holdout_mode=args.holdout_mode,
        holdout_every=args.holdout_every,
        holdout_offset=args.holdout_offset,
    )
    if not split_idxs:
        raise SystemExit("No views matched the requested split.")
    view_rank = len(split_idxs) // 2 if args.view_rank < 0 else max(0, min(args.view_rank, len(split_idxs) - 1))
    rec = recs[split_idxs[view_rank]]

    out_path = args.output.expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    history_dir = args.history_dir.expanduser() if args.history_dir else None
    if history_dir is not None:
        history_dir.mkdir(parents=True, exist_ok=True)

    last_mtime_ns = None
    while True:
        ckpt = args.checkpoint.expanduser()
        if not ckpt.exists():
            if args.once:
                raise SystemExit(f"Missing checkpoint: {ckpt}")
            time.sleep(max(args.poll_seconds, 1.0))
            continue
        mtime_ns = ckpt.stat().st_mtime_ns
        if last_mtime_ns == mtime_ns and not args.once:
            time.sleep(max(args.poll_seconds, 1.0))
            continue

        front, rear, slab, cam_mlp, z_off, global_step = _load_model_bundle(ckpt)
        with torch.inference_mode():
            out = _render_view(
                front,
                rear,
                slab,
                cam_mlp,
                rec,
                z_off,
                gate_temperature=float(args.gate_temperature),
                hard_gate_temperature=float(args.hard_gate_temperature),
                adaptive_router_strength=float(args.adaptive_router_strength),
            )
        pred = out[PRED_KEYS[args.pred]]
        psnr, l1 = _score(pred, out["gt"])
        arr = pred.clamp(0.0, 1.0).mul(255.0).to(torch.uint8).cpu().numpy()
        imageio.imwrite(out_path, arr)
        if history_dir is not None:
            hist = history_dir / f"{args.pred}_step_{global_step:07d}.png"
            imageio.imwrite(hist, arr)
        print(
            f"updated {out_path} from step {global_step} "
            f"split={args.split} view_rank={view_rank} pred={args.pred} "
            f"psnr={psnr:.3f} l1={l1:.4f}"
        )
        last_mtime_ns = mtime_ns
        if args.once:
            break
        time.sleep(max(args.poll_seconds, 1.0))


if __name__ == "__main__":
    main()
