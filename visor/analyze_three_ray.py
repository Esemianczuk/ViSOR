#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

from visor import RENDERS_DIR
from visor.gaussian_slab import GaussianSlab, slab_ray_context
from visor.train import (
    DEVICE,
    RES,
    CameraEmbed,
    HARD_GATE_TEMPERATURE,
    OcclusionSheet,
    RefractionSheet,
    encode_dir,
    encode_vec,
    pixel_rays,
)
from visor.plane_geometry import composite_two_planes, look_at_rotation, plane_frame, project_rays_to_plane


def _load_records(renders_dir: Path):
    with (renders_dir / "views.jsonl").open() as f:
        return [json.loads(ln) for ln in f]


def _sample_indices(count: int, max_views: int) -> list[int]:
    if max_views <= 0 or max_views >= count:
        return list(range(count))
    if max_views == 1:
        return [count // 2]
    return sorted({round(i * (count - 1) / (max_views - 1)) for i in range(max_views)})


def _phi_bucket(rec: dict, bucket_count: int) -> int:
    phi = float(rec["phi"])
    phi_norm = ((phi + math.pi) / (2.0 * math.pi)) % 1.0
    return min(bucket_count - 1, int(phi_norm * bucket_count))


def _split_indices(
    recs: list[dict],
    split: str,
    holdout_mode: str,
    holdout_every: int,
    holdout_offset: int,
) -> list[int]:
    if split == "all":
        return list(range(len(recs)))

    holdout_every = max(1, holdout_every)
    holdout_offset = holdout_offset % holdout_every
    chosen = []
    for idx, rec in enumerate(recs):
        if holdout_mode == "phi_bucket":
            is_holdout = _phi_bucket(rec, holdout_every) == holdout_offset
        else:
            is_holdout = (idx % holdout_every) == holdout_offset
        if (split == "heldout" and is_holdout) or (split == "train" and not is_holdout):
            chosen.append(idx)
    return chosen


def _render_view(front, rear, slab, cam_mlp, rec, z_off, gate_temperature: float, hard_gate_temperature: float, adaptive_router_strength: float):
    phi, theta, rho = rec["phi"], rec["theta"], rec["rho"]
    loc = torch.tensor([
        rho * math.cos(theta) * math.cos(phi),
        rho * math.sin(theta),
        rho * math.cos(theta) * math.sin(phi),
    ], device=DEVICE, dtype=torch.float32)
    cam_feat = cam_mlp(
        torch.tensor(phi, device=DEVICE),
        torch.tensor(theta, device=DEVICE),
        torch.tensor(rho, device=DEVICE),
    )

    yy, xx = torch.meshgrid(
        torch.arange(RES, device=DEVICE, dtype=torch.float16),
        torch.arange(RES, device=DEVICE, dtype=torch.float16),
        indexing="ij",
    )
    yyf, xxf = yy.flatten(), xx.flatten()

    Rwc = look_at_rotation(loc, device=DEVICE)
    ray_o, ray_d = pixel_rays(yyf, xxf, Rwc.half(), loc.half(), z=0.0)
    front_center, rear_center, plane_normal, plane_u, plane_v = plane_frame(loc, z_off, device=DEVICE, dtype=ray_o.dtype)
    x_f, y_f, hit_f, t_f, valid_f = project_rays_to_plane(ray_o, ray_d, front_center, plane_normal, plane_u, plane_v, RES)
    x_r, y_r, hit_r, t_r, valid_r = project_rays_to_plane(ray_o, ray_d, rear_center, plane_normal, plane_u, plane_v, RES)

    with autocast(enabled=(DEVICE == "cuda")):
        pe_f = torch.cat([encode_dir(ray_d).half(), encode_vec(hit_f).half()], dim=-1)
        rgb_f, tau_f = front(y_f, x_f, pe_f, cam_feat)
        rgb_f = rgb_f.clamp(0, 1) * valid_f.unsqueeze(-1).to(dtype=rgb_f.dtype)
        tau_f = torch.where(valid_f, tau_f.clamp(0, 1), torch.ones_like(tau_f))
        alpha_f = 1.0 - tau_f

        rear_out, rear_alpha, _, rear_aux = rear(
            y_r, x_r, hit_r, ray_d, cam_feat,
            front_alpha=alpha_f,
            front_rgb=rgb_f,
            gate_temperature=gate_temperature,
            hard_gate_temperature=hard_gate_temperature,
            adaptive_router_strength=adaptive_router_strength,
            return_aux=True,
        )
        rear_full = rear_out.clamp(0, 1)
        rear_full = rear_full * valid_r.unsqueeze(-1).to(dtype=rear_full.dtype)
        rear_alpha = rear_alpha.clamp(0, 1) * valid_r.to(dtype=rear_alpha.dtype)
        slab_context = slab_ray_context(
            rear_aux["head_rgb"],
            rear_aux["gates"],
            rear_aux["route_strength"],
            alpha_f,
            rear_alpha,
        )
        slab_trans, slab_aux = slab(
            hit_f, hit_r,
            front_center, rear_center,
            plane_normal, plane_u, plane_v,
            valid_f, valid_r,
            ray_context=slab_context,
            return_aux=True,
        )
        rear_alpha_slab = rear_alpha * slab_trans.to(dtype=rear_alpha.dtype)

        pred_full, _, _ = composite_two_planes(rgb_f, alpha_f, t_f, valid_f, rear_full, rear_alpha_slab, t_r, valid_r)
        pred_no_slab, _, _ = composite_two_planes(rgb_f, alpha_f, t_f, valid_f, rear_full, rear_alpha, t_r, valid_r)

        rear_zero, rear_alpha_zero, _, rear_aux_zero = rear(
            y_r, x_r, hit_r, ray_d, cam_feat,
            front_alpha=alpha_f,
            front_rgb=rgb_f,
            offset_scale=0.0,
            gate_temperature=gate_temperature,
            hard_gate_temperature=hard_gate_temperature,
            adaptive_router_strength=adaptive_router_strength,
            return_aux=True,
        )
        rear_zero = rear_zero.clamp(0, 1)
        rear_zero = rear_zero * valid_r.unsqueeze(-1).to(dtype=rear_zero.dtype)
        rear_alpha_zero = rear_alpha_zero.clamp(0, 1) * valid_r.to(dtype=rear_alpha_zero.dtype)
        rear_alpha_zero_slab = rear_alpha_zero * slab_trans.to(dtype=rear_alpha_zero.dtype)
        pred_no_refrac, _, _ = composite_two_planes(
            rgb_f, alpha_f, t_f, valid_f, rear_zero, rear_alpha_zero_slab, t_r, valid_r
        )

        head_rgb = rear_aux["head_rgb"].float()
        gate_logits = rear_aux["gate_logits"].float()
        gates = rear_aux["gates"].float()
        offset_px = rear_aux["offset_px"].float()
        pred_head_mean, _, _ = composite_two_planes(
            rgb_f.float(), alpha_f.float(), t_f, valid_f,
            head_rgb.mean(dim=1), rear_alpha_slab.float(), t_r, valid_r,
        )
        pred_head0, _, _ = composite_two_planes(
            rgb_f.float(), alpha_f.float(), t_f, valid_f,
            head_rgb[:, 0], rear_alpha_slab.float(), t_r, valid_r,
        )

    gt = torch.from_numpy(imageio.imread(Path(rec["_img_path"]))[:, :, :3]).to(DEVICE).float() / 255.0
    gt = gt.view(RES, RES, 3)

    pair_spread = []
    for i, j in ((0, 1), (0, 2), (1, 2)):
        pair_spread.append((offset_px[:, i] - offset_px[:, j]).norm(dim=-1))
    pair_spread = torch.stack(pair_spread, dim=1)
    gate_entropy = -(gates.clamp(1e-6, 1.0) * gates.clamp(1e-6, 1.0).log()).sum(dim=-1)

    return {
        "gt": gt,
        "pred_full": pred_full.view(RES, RES, 3).float(),
        "pred_front_only": rgb_f.view(RES, RES, 3).float(),
        "pred_no_refrac": pred_no_refrac.view(RES, RES, 3).float(),
        "pred_no_slab": pred_no_slab.view(RES, RES, 3).float(),
        "pred_head_mean": pred_head_mean.view(RES, RES, 3).float(),
        "pred_head0": pred_head0.view(RES, RES, 3).float(),
        "head_rgb": head_rgb,
        "front_alpha": alpha_f.float(),
        "rear_alpha": rear_alpha.float(),
        "rear_alpha_zero": rear_alpha_zero.float(),
        "pair_spread": pair_spread,
        "offset_mag": offset_px.norm(dim=-1),
        "gates": gates,
        "gate_logits": gate_logits,
        "gate_entropy": gate_entropy,
        "route_strength": rear_aux["route_strength"].float(),
        "gate_temperature_ray": rear_aux["gate_temperature_ray"].float(),
        "rear_head_pair_l1": torch.stack([
            (head_rgb[:, 0] - head_rgb[:, 1]).abs().mean(dim=-1),
            (head_rgb[:, 0] - head_rgb[:, 2]).abs().mean(dim=-1),
            (head_rgb[:, 1] - head_rgb[:, 2]).abs().mean(dim=-1),
        ], dim=1),
        "rear_sample_valid": rear_aux["sample_valid"].float(),
        "slab_alpha": slab_aux["alpha"].view(-1).float(),
        "slab_tau": slab_aux["tau"].view(-1).float(),
        "slab_signal": slab_aux["signal"].view(-1).float(),
        "slab_ray_gain": slab_aux["ray_gain"].view(-1).float(),
        "slab_strength_mean": slab_aux["strength_mean"].float(),
        "slab_mass_peak": slab_aux["mass_peak"].view(-1).float(),
        "slab_context_mod_mean": slab_aux["context_mod_mean"].float(),
        "slab_disagreement_gain": slab_aux["disagreement_gain"].float(),
        "slab_context_scale": slab_aux["context_scale"].float(),
        "rear_aux_zero": rear_aux_zero,
    }


def _score(pred: torch.Tensor, gt: torch.Tensor) -> tuple[float, float]:
    mse = F.mse_loss(pred, gt)
    psnr = float(-10.0 * torch.log10(mse + 1e-8))
    l1 = float((pred - gt).abs().mean())
    return psnr, l1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--renders-dir", type=Path, default=RENDERS_DIR)
    ap.add_argument("--max-views", type=int, default=8)
    ap.add_argument("--split", choices=("all", "train", "heldout"), default="all")
    ap.add_argument("--holdout-mode", choices=("index", "phi_bucket"), default="phi_bucket")
    ap.add_argument("--holdout-every", type=int, default=8)
    ap.add_argument("--holdout-offset", type=int, default=0)
    ap.add_argument("--output-json", type=Path, default=None)
    ap.add_argument("--gate-temperature", type=float, default=1.0)
    ap.add_argument("--hard-gate-temperature", type=float, default=HARD_GATE_TEMPERATURE)
    ap.add_argument("--adaptive-router-strength", type=float, default=1.0)
    args = ap.parse_args()

    renders_dir = args.renders_dir.expanduser()
    recs = _load_records(renders_dir)
    for rec in recs:
        rec["_img_path"] = str(renders_dir / rec["file"])

    ck = torch.load(args.checkpoint.expanduser(), map_location="cpu")
    front = OcclusionSheet(RES, RES, False).to(DEVICE)
    rear = RefractionSheet(RES, RES, False).to(DEVICE)
    slab_splats = int(ck["slab"]["mean_raw"].shape[0]) if ("slab" in ck and "mean_raw" in ck["slab"]) else 32
    slab = GaussianSlab(num_splats=slab_splats).to(DEVICE)
    cam_mlp = CameraEmbed().to(DEVICE)
    front.load_state_dict(ck["front"], strict=False)
    rear.load_state_dict(ck["rear"], strict=False)
    if "slab" in ck:
        slab.load_state_dict(ck["slab"], strict=False)
    if "camera_embed" in ck:
        cam_mlp.load_state_dict(ck["camera_embed"], strict=False)
    z_off = ck.get("z_offset", 0.3)
    front.eval()
    rear.eval()
    slab.eval()
    cam_mlp.eval()

    split_idxs = _split_indices(
        recs,
        split=args.split,
        holdout_mode=args.holdout_mode,
        holdout_every=args.holdout_every,
        holdout_offset=args.holdout_offset,
    )
    if not split_idxs:
        raise SystemExit("No views matched the requested split.")
    sampled_rel = _sample_indices(len(split_idxs), args.max_views)
    idxs = [split_idxs[i] for i in sampled_rel]
    metrics = {
        "full": [],
        "front_only": [],
        "no_refrac": [],
        "no_slab": [],
        "head_mean": [],
        "head0": [],
    }
    diag = {
        "front_alpha_mean": [],
        "front_alpha_std": [],
        "rear_alpha_mean": [],
        "rear_alpha_std": [],
        "rear_pair_spread_px_mean": [],
        "rear_offset_mag_mean": [],
        "rear_head_pair_l1_mean": [],
        "gate_entropy_mean": [],
        "gate_mean_0": [],
        "gate_mean_1": [],
        "gate_mean_2": [],
        "route_strength_mean": [],
        "gate_temperature_ray_mean": [],
        "rear_sample_valid_frac": [],
        "rear_alpha_zero_mean": [],
        "slab_alpha_mean": [],
        "slab_tau_mean": [],
        "slab_signal_mean": [],
        "slab_ray_gain_mean": [],
        "slab_mass_peak_mean": [],
        "slab_strength_mean": [],
        "slab_context_mod_mean": [],
        "slab_disagreement_gain": [],
        "slab_context_scale": [],
    }

    with torch.inference_mode():
        for idx in idxs:
            out = _render_view(
                front,
                rear,
                slab,
                cam_mlp,
                recs[idx],
                z_off,
                gate_temperature=float(args.gate_temperature),
                hard_gate_temperature=float(args.hard_gate_temperature),
                adaptive_router_strength=float(args.adaptive_router_strength),
            )
            gt = out["gt"]
            for name, pred_key in (
                ("full", "pred_full"),
                ("front_only", "pred_front_only"),
                ("no_refrac", "pred_no_refrac"),
                ("no_slab", "pred_no_slab"),
                ("head_mean", "pred_head_mean"),
                ("head0", "pred_head0"),
            ):
                psnr, l1 = _score(out[pred_key], gt)
                metrics[name].append({"psnr": psnr, "l1": l1})

            diag["front_alpha_mean"].append(float(out["front_alpha"].mean().item()))
            diag["front_alpha_std"].append(float(out["front_alpha"].std().item()))
            diag["rear_alpha_mean"].append(float(out["rear_alpha"].mean().item()))
            diag["rear_alpha_std"].append(float(out["rear_alpha"].std().item()))
            diag["rear_pair_spread_px_mean"].append(float(out["pair_spread"].mean().item()))
            diag["rear_offset_mag_mean"].append(float(out["offset_mag"].mean().item()))
            diag["rear_head_pair_l1_mean"].append(float(out["rear_head_pair_l1"].mean().item()))
            diag["gate_entropy_mean"].append(float(out["gate_entropy"].mean().item()))
            diag["gate_mean_0"].append(float(out["gates"][:, 0].mean().item()))
            diag["gate_mean_1"].append(float(out["gates"][:, 1].mean().item()))
            diag["gate_mean_2"].append(float(out["gates"][:, 2].mean().item()))
            diag["route_strength_mean"].append(float(out["route_strength"].mean().item()))
            diag["gate_temperature_ray_mean"].append(float(out["gate_temperature_ray"].mean().item()))
            diag["rear_sample_valid_frac"].append(float(out["rear_sample_valid"].mean().item()))
            diag["rear_alpha_zero_mean"].append(float(out["rear_alpha_zero"].mean().item()))
            diag["slab_alpha_mean"].append(float(out["slab_alpha"].mean().item()))
            diag["slab_tau_mean"].append(float(out["slab_tau"].mean().item()))
            diag["slab_signal_mean"].append(float(out["slab_signal"].mean().item()))
            diag["slab_ray_gain_mean"].append(float(out["slab_ray_gain"].mean().item()))
            diag["slab_mass_peak_mean"].append(float(out["slab_mass_peak"].mean().item()))
            diag["slab_strength_mean"].append(float(out["slab_strength_mean"].item()))
            diag["slab_context_mod_mean"].append(float(out["slab_context_mod_mean"].item()))
            diag["slab_disagreement_gain"].append(float(out["slab_disagreement_gain"].item()))
            diag["slab_context_scale"].append(float(out["slab_context_scale"].item()))

    summary = {
        "checkpoint": str(args.checkpoint.expanduser()),
        "split": args.split,
        "holdout_mode": args.holdout_mode,
        "holdout_every": int(args.holdout_every),
        "holdout_offset": int(args.holdout_offset),
        "gate_temperature": float(args.gate_temperature),
        "hard_gate_temperature": float(args.hard_gate_temperature),
        "adaptive_router_strength": float(args.adaptive_router_strength),
        "views_available_in_split": len(split_idxs),
        "views_evaluated": idxs,
        "metrics": {
            name: {
                "mean_psnr": sum(m["psnr"] for m in vals) / len(vals),
                "mean_l1": sum(m["l1"] for m in vals) / len(vals),
            }
            for name, vals in metrics.items()
        },
        "diagnostics": {
            key: sum(vals) / len(vals)
            for key, vals in diag.items()
        },
    }

    for name, vals in summary["metrics"].items():
        print(f"{name:>10s}  psnr={vals['mean_psnr']:.3f}  l1={vals['mean_l1']:.4f}")
    print(
        f"split {args.split} mode={args.holdout_mode} every={args.holdout_every} offset={args.holdout_offset} "
        f"gate_t={args.gate_temperature:.2f}->{args.hard_gate_temperature:.2f} "
        f"route_mix={args.adaptive_router_strength:.2f} "
        f"count={len(split_idxs)} sampled={len(idxs)}"
    )
    print(
        "diagnostics "
        f"front_alpha={summary['diagnostics']['front_alpha_mean']:.3f}+/-{summary['diagnostics']['front_alpha_std']:.3f} "
        f"rear_alpha={summary['diagnostics']['rear_alpha_mean']:.3f}+/-{summary['diagnostics']['rear_alpha_std']:.3f} "
        f"gates=[{summary['diagnostics']['gate_mean_0']:.2f},{summary['diagnostics']['gate_mean_1']:.2f},{summary['diagnostics']['gate_mean_2']:.2f}] "
        f"route={summary['diagnostics']['route_strength_mean']:.2f} "
        f"gate_tr={summary['diagnostics']['gate_temperature_ray_mean']:.2f} "
        f"pair_spread={summary['diagnostics']['rear_pair_spread_px_mean']:.3f}px "
        f"offset_mag={summary['diagnostics']['rear_offset_mag_mean']:.3f}px "
        f"head_div={summary['diagnostics']['rear_head_pair_l1_mean']:.3f} "
        f"gate_H={summary['diagnostics']['gate_entropy_mean']:.3f} "
        f"slab_a={summary['diagnostics']['slab_alpha_mean']:.3f} "
        f"slab_tau={summary['diagnostics']['slab_tau_mean']:.3f} "
        f"slab_sig={summary['diagnostics']['slab_signal_mean']:.3f} "
        f"slab_rg={summary['diagnostics']['slab_ray_gain_mean']:.3f}"
    )

    if args.output_json is not None:
        args.output_json.expanduser().write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
