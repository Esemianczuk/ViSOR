#!/usr/bin/env python3
"""



Usage:
  python dual_billboard_with_sh.py [--sh_file=sh_billboard_L7.pt]
  (Same command-line usage as original, plus optional --sh_file argument.)

"""
from visor import PROJECT_ROOT, RENDERS_DIR
import json, math, time, collections, os, warnings, argparse
from pathlib import Path
from typing import Optional, Tuple
from math import factorial
import imageio.v2 as imageio
import numpy as np
import pygame

import torch
import torch.nn as nn
import torch.nn.functional as F
from visor.gaussian_slab import GaussianSlab, slab_head_disagreement, slab_ray_context
from visor.hashgrid import hashgrid
from visor.plane_geometry import composite_two_planes, look_at_rotation, plane_frame, project_rays_to_plane

# -- NEW IMPORTS FOR MIXED PRECISION --
from torch.cuda.amp import autocast, GradScaler

# ───────── hyper-params ───────── #
RES                = 512
ITERS_TOTAL        = 200000000
CHECKPOINT_PATH    = PROJECT_ROOT / f"dual_billboard_{RES:04d}_x2_cont_F7_sh.pt"

VIEW_DROPOUT       = 0.15          # 15 %
VAL_EPS            = 1e-4
LOG_EVERY          = 200
BATCH              = 16_384
LR_BASE            = 1e-4
WARMUP_STEPS       = 1000
CLIP_NORM          = 4.0
TV_WEIGHT          = 0 #1e-4
EDGE_WEIGHT        = 0.1
MAX_SAMPLE_OFFSET_PX = 8.0
MAX_RELATIVE_OFFSET_PX = 2.0
BASE_MIN_PAIR_SEPARATION_PX = 0.12
MAX_FAN_SHIFT_PX = 1.0
RESIDUAL_RGB_SCALE = 0.35
HARD_GATE_TEMPERATURE = 0.75
ROUTE_STRENGTH_ALIGN_WEIGHT = 2e-2
ADAPTIVE_GATE_ENTROPY_WEIGHT = 2e-2
HARD_GATE_ENTROPY_FRACTION = 0.72
RESIDUAL_GAIN_WEIGHT = 0.25
PHASE1_STEPS = 0
PHASE2_RESIDUAL_GAIN_MULT = 1.0
PHASE2_BASE_GRAD_SCALE = 0.25
PHASE2_HARD_FOCUS_QUANTILE = 0.6
PHASE2_HARD_FOCUS_FLOOR = 0.25
PHASE2_HARD_SPREAD_FLOOR_PX = 0.60
PHASE2_EDGE_SPREAD_MULT = 2.0
PHASE2_PAIR_REPULSION_MULT = 3.0
PHASE3_START_RATIO = 0.55
PHASE3_END_SPREAD_FLOOR_SCALE = 0.35
PHASE3_END_EDGE_SPREAD_MULT = 1.15
PHASE3_END_PAIR_REPULSION_MULT = 1.5
PHASE3_END_GATE_TEMPERATURE = 3.0
FREEZE_BASE_AFTER_PHASE1 = False
CKPT_EVERY         = 2_000
SAFETY_GB          = 20
BUNDLE_K      = 5          # rays per training pixel
BUNDLE_STD_PX = 0.35       # Gaussian jitter radius in *pixel* units
TRAIN_CHUNK        = 610
EXTRA_STEPS        = 0
DIAG_EVERY         = 0
DIAG_JSONL         = None

# depth-to-τ parameters
TAU_SIGMA_FRACTION = 0.15
TAU_WEIGHT         = 0.05

def safe_px(t: torch.Tensor, res: int = RES) -> torch.LongTensor:
    """
    Round to nearest int and clamp to [0, res-1] so we can safely use
    the result for tensor indexing.
    """
    return t.round().clamp_(0, res - 1).long()


DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True  # speed with TF32
torch.backends.cudnn.allow_tf32 = True        # speed with TF32 for conv
torch.backends.cudnn.benchmark = True

def sample_bundle(yc, xc, Rwc, cam_loc, k=BUNDLE_K):
    """Return origins (k,3) and dirs (k,3) for a cone centred at (yc,xc)."""
    offs = torch.randn(k,2, device=DEVICE) * BUNDLE_STD_PX
    ys   = yc + offs[:,0]
    xs   = xc + offs[:,1]
    return pixel_rays(ys, xs, Rwc, cam_loc, 0.0)   # (k,3), (k,3)

# Rear transport regularizers.
OFFSET_L2_WEIGHT    = 1e-3
GATE_USAGE_WEIGHT   = 1e-2
HEAD_DECORR_WEIGHT  = 2e-2
EDGE_SPREAD_WEIGHT  = 2e-2
EDGE_SPREAD_TARGET_PX = 4.0
PAIR_REPULSION_WEIGHT = 1e-1
SLAB_SPLATS = 32
SLAB_STRENGTH_WEIGHT = 5e-4
SLAB_GAIN_WEIGHT = 1.5e-1
SLAB_OPACITY_WEIGHT = 2e-3
NON_SLAB_LR_SCALE = 1.0
SLAB_LR_SCALE = 1.0
SLAB_RAMP_WARMUP_STEPS = 400
SLAB_RAMP_STEPS = 1200
SLAB_RAMP_START_SCALE = 0.10
SLAB_RAMP_HEAD_DIV_THRESHOLD = 0.08
SLAB_RAMP_SPREAD_THRESHOLD = 0.40

def rot_yx(phi, theta, device=DEVICE):
    cy, sy = math.cos(phi),   math.sin(phi)
    cp, sp = math.cos(theta), math.sin(theta)
    R_y = torch.tensor([[ cy, 0., sy],
                        [ 0., 1., 0.],
                        [-sy, 0., cy ]], dtype=torch.float32, device=device)
    R_x = torch.tensor([[1., 0., 0.],
                        [0., cp,-sp],
                        [0., sp, cp ]], dtype=torch.float32, device=device)
    return R_y @ R_x                         # yaw-then-pitch (matches viewer)


# ───────── positional encodings ───────── #
PE_BANDS = 8
def _psnr(pred, tgt):
    mse = F.mse_loss(pred, tgt)
    return -10.0 * torch.log10(mse + VAL_EPS)

def encode_dir(d):
    out = [d]
    for i in range(PE_BANDS):
        k = 2**i
        out += [torch.sin(k*d), torch.cos(k*d)]
    return torch.cat(out, -1)

def encode_vec(v):
    out = [v]
    for i in range(PE_BANDS):
        k = 2**i
        out += [torch.sin(k*v), torch.cos(k*v)]
    return torch.cat(out, -1)

# ───────── SinLayer with clamp ───────── #
class SinLayer(nn.Linear):
    def forward(self, x):
        x_32 = F.linear(x.float(), self.weight.float(), self.bias.float())
        x_32 = torch.clamp(x_32, -25.133, 25.133)
        y_32 = torch.sin(x_32)
        return y_32.to(x.dtype)

def make_siren(inp, outp):
    m = SinLayer(inp, outp)
    nn.init.uniform_(m.weight, -1/inp, 1/inp)
    nn.init.zeros_(m.bias)
    return m

def make_mlp(inp, hid, out, depth=3, siren=False):
    """
    Build an MLP that may use a SinLayer first layer for SIREN.
    """
    first = make_siren if siren else nn.Linear
    layers = [first(inp, hid), nn.ReLU(True)]
    for _ in range(depth - 1):
        layers += [nn.Linear(hid, hid), nn.ReLU(True)]
    layers += [nn.Linear(hid, out)]
    return nn.Sequential(*layers)

# ───────── Additional: CameraEmbed ───────── #
class CameraEmbed(nn.Module):
    """
    MLP from (phi,theta,rho) -> small embedding. 
    This is combined with the plane's latent code.
    """
    def __init__(self, embed_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(True),
            nn.Linear(64, 64), nn.ReLU(True),
            nn.Linear(64, embed_dim)
        )
    def forward(self, phi, theta, rho):
        x = torch.stack([phi, theta, rho], dim=-1).float()
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)
        out = self.net(x)
        return out[0] if squeeze else out

# ───────── Spherical Harmonics sampler ───────── #
def _legendre_p_batch(lmax: int, x: torch.Tensor):
    """
    Return P_l^m(x) for 0 <= l <= lmax, 0 <= m <= l, shape => (lmax+1, lmax+1, N).
    """
    N = x.shape[0]
    device = x.device
    out = torch.zeros((lmax+1, lmax+1, N), dtype=torch.float32, device=device)

    out[0,0] = 1.0
    if lmax == 0:
        return out

    out[1,0] = x
    y = torch.sqrt(torch.clamp(1 - x*x, min=1e-14))
    out[1,1] = y

    for l in range(2, lmax+1):
        out[l,l] = (2*l-1) * y * out[l-1,l-1]
        for m in range(l-1, -1, -1):
            if l==1 and m==0:
                continue
            a = (2*l - 1)*x*out[l-1,m]
            b = 0.0
            if (l-2) >= 0:
                b = (l + m - 1)*out[l-2,m]
            out[l,m] = (a - b)/(l - m)
    return out

def real_spherical_harmonics(dirs: torch.Tensor, L: int) -> torch.Tensor:
    """
    Compute Real SH basis up to band L for each direction in dirs (N,3).
    Return (N, (L+1)^2).
    """


    N = dirs.size(0)
    device = dirs.device
    out_dim = (L+1)*(L+1)

    # spherical angles
    eps = 1e-14
    r = dirs.norm(dim=-1) + eps
    x, y, z = dirs[:,0], dirs[:,1], dirs[:,2]
    theta = torch.acos(torch.clamp(z / r, -1, 1))
    phi   = torch.atan2(y, x)

    ctheta = torch.cos(theta)
    p_all = _legendre_p_batch(L, ctheta)  # (l+1, l+1, N)

    # Normalization K(l,m)
    norm_lm = torch.zeros((L+1, L+1), dtype=torch.float32, device=device)
    for l in range(L+1):
        for m in range(l+1):
            num = factorial(l-m)
            den = factorial(l+m)
            norm_lm[l,m] = math.sqrt((2*l+1)/(4*math.pi) * (num/den))

    m_grid = torch.arange(L+1, device=device, dtype=torch.float32).view(-1,1)
    mp = m_grid * phi.unsqueeze(0)
    cos_mphi = torch.cos(mp)
    sin_mphi = torch.sin(mp)

    out = dirs.new_zeros((N, out_dim), dtype=torch.float32)

    def sh_index(l, m):
        # canonical real SH ordering: index runs from l^2..(l+1)^2 -1 
        # with m offset by +l
        return l*l + l + m

    for l in range(L+1):
        for m in range(-l, l+1):
            idx = sh_index(l,m)
            if m == 0:
                out[:,idx] = norm_lm[l,0] * p_all[l,0]
            elif m>0:
                sign = (-1)**m
                out[:,idx] = math.sqrt(2.0)*sign * norm_lm[l,m] * p_all[l,m] * cos_mphi[m]
            else:
                mp_ = -m
                sign = (-1)**m
                out[:,idx] = math.sqrt(2.0)*sign * norm_lm[l,mp_] * p_all[l,mp_] * sin_mphi[mp_]
    return out

class SHEmbed(nn.Module):
    """
    Loads a pre-baked SH coefficient grid: shape [RES, RES, (L+1)^2, 3]
    Then, for each pixel and ray-dir, returns the SH color.
    """
    def __init__(self, sh_file: Path, device=DEVICE):
        super().__init__()
        data = torch.load(sh_file, map_location="cpu")   # expects {"sh":..., "L":...}
        sh_tensor = data["sh"]                           # (RES,RES, (L+1)^2,3)
        self.L = data["L"]
        self.register_buffer("sh_data", sh_tensor.to(device))

    def forward(self, y: torch.Tensor, x: torch.Tensor, ray_dir: torch.Tensor):
        """
        y,x: shape (B,) pixel indices
        ray_dir: shape (B,3)
        Return shape (B,3) color from the SH. 
        """
        # get the SH coefficients for each pixel
        # indexing => (B, (L+1)^2, 3)
        yi, xi = safe_px(y, self.sh_data.shape[0]), safe_px(x, self.sh_data.shape[1])
        coefs  = self.sh_data[yi, xi]

        B_mat = real_spherical_harmonics(ray_dir, self.L)

        color = torch.einsum("bi,bij->bj", B_mat, coefs)
        color = color.clamp(0,1)
        return color

# # ───────── billboard dims ───────── #
CODE_DIM       = 384
CAM_EMBED_DIM  = 32
POS_HID        = 512
POS_OUT        = 384
HEAD_HID       = 768

# # ───────── billboard (light) dims ───────── #
# CODE_DIM       = 192 
# CAM_EMBED_DIM  = 32
# POS_HID        = 256      
# POS_OUT        = 192 
# HEAD_HID       = 384


PE_DIM = (3 + 6*PE_BANDS)*2  # (dir + loc) => 78 when PE_BANDS=8 (default)

# ───────────────── Fourier helper ─────────────────
TAU = 2 * math.pi
def fourier_uv(x: torch.LongTensor, y: torch.LongTensor) -> torch.Tensor:
    """
    Return Fourier features (sin / cos 2πu,v) for each integer pixel.
    Jitters sub-pixel for training.
    """
    u = (x.float() + torch.rand_like(x, dtype=torch.float)) / RES
    v = (y.float() + torch.rand_like(y, dtype=torch.float)) / RES
    return torch.stack(
        [torch.sin(TAU*u), torch.cos(TAU*u),
         torch.sin(TAU*v), torch.cos(TAU*v)], dim=-1
    )  # (B,4)
FUV_DIM = 4


class Heads(nn.Module):
    """
    For the front billboard: R, G, B from latent + direction PE + optional SH color,
    plus alpha from latent+PE+FUV+SH.
    """
    def __init__(self, output_alpha: bool = True, with_sh: bool = False):
        super().__init__()
        self.with_sh = with_sh
        # If we feed the SH color (3 channels) into the heads:
        sh_dim = 3 if with_sh else 0

        # For the “diffuse” path => input is (POS_OUT + sh_dim)
        in_rgb_diff = POS_OUT + sh_dim

        # For the “specular” path => input is (POS_OUT + PE_DIM + sh_dim)
        in_rgb_spec = POS_OUT + PE_DIM + sh_dim

        # For “roughness” => (POS_OUT + FUV_DIM + sh_dim)
        in_rough    = POS_OUT + FUV_DIM + sh_dim

        # For “alpha” => (POS_OUT + FUV_DIM + PE_DIM + sh_dim)
        in_alpha    = POS_OUT + FUV_DIM + PE_DIM + sh_dim


        self.fuv_scale = nn.Parameter(torch.tensor(0.1))

        self.diff = make_mlp(in_rgb_diff, HEAD_HID, 3, depth=4, siren=True)
        self.spec = make_mlp(in_rgb_spec, HEAD_HID, 3, depth=4, siren=True)
        self.rough = nn.Sequential(
            make_mlp(in_rough, 64, 1, depth=2),
            nn.Sigmoid()
        )

        self.alpha = None
        if output_alpha:
            self.alpha = nn.Sequential(
                make_mlp(in_alpha, HEAD_HID, 1, depth=2),
                nn.Sigmoid()
            )

    def forward(self,
                feat: torch.Tensor,   # (B, POS_OUT)
                pe:   torch.Tensor,   # (B, PE_DIM)
                fuv:  torch.Tensor,   # (B,4)
                sh_col: Optional[torch.Tensor] = None
               ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        If with_sh=True, `sh_col` is shape (B,3), the precomputed SH color for that pixel+dir.
        """
        gamma_fuv = self.fuv_scale * fuv

        if self.with_sh and sh_col is not None:
            # cat the 3-ch SH color onto `feat`
            feat_diff = torch.cat([feat, sh_col], dim=-1)
            feat_spec = torch.cat([feat, pe, sh_col], dim=-1)
            feat_rough= torch.cat([feat, gamma_fuv, sh_col], dim=-1)
            alpha_in  = torch.cat([feat, pe, gamma_fuv, sh_col], dim=-1)
        else:
            feat_diff = feat
            feat_spec = torch.cat([feat, pe], dim=-1)
            feat_rough= torch.cat([feat, gamma_fuv], dim=-1)
            alpha_in  = torch.cat([feat, pe, gamma_fuv], dim=-1)

        rgb_d = torch.sigmoid(self.diff(feat_diff))
        rgb_s = torch.sigmoid(self.spec(feat_spec)) * 0.7

        rough = self.rough(feat_rough)
        rgb = rgb_d + (1.0 - rough) * rgb_s

        if self.alpha is None:
            return rgb, None

        a = self.alpha(alpha_in).squeeze(-1)
        return rgb, a


class SheetBase(nn.Module):
    def __init__(self, h: int, w: int):
        super().__init__()
        # 1. replace dense tensor
        self.codes = hashgrid.HashGrid(
            n_levels=16,
            n_features_per_level=2,
            log2_hashmap_size=19,
            base_resolution=32,
            per_level_scale=1.3819
        )
        # 2. adjust feature dim
        self.code_dim = 32               # 16 lvls × 2 feats
        self.pos = nn.Sequential(
            make_siren(self.code_dim + CAM_EMBED_DIM, POS_HID),
            make_siren(POS_HID, POS_HID),
            make_siren(POS_HID, POS_OUT)
        )

    def _feat(self, y, x, cam_feat):
        uv = torch.stack([
            x.float().clamp(0, RES - 1) / max(RES - 1, 1),
            y.float().clamp(0, RES - 1) / max(RES - 1, 1),
        ], -1)

        codes_xy = self.codes(uv)           # (B, code_dim)
        if cam_feat.dim() == 1:
            cam_feat_xy = cam_feat.unsqueeze(0).expand(codes_xy.shape[0], -1)
        elif cam_feat.dim() == 2 and cam_feat.shape[0] == codes_xy.shape[0]:
            cam_feat_xy = cam_feat
        elif cam_feat.dim() == 2 and cam_feat.shape[0] == 1:
            cam_feat_xy = cam_feat.expand(codes_xy.shape[0], -1)
        else:
            raise ValueError(f"Unsupported cam_feat shape for _feat: {tuple(cam_feat.shape)}")
        return self.pos(torch.cat([codes_xy, cam_feat_xy.to(dtype=codes_xy.dtype)], -1))



class OcclusionSheet(SheetBase):
    def __init__(self, h: int, w: int, with_sh: bool = False):
        super().__init__(h, w)
        self.heads = Heads(output_alpha=True, with_sh=with_sh)
        self.with_sh = with_sh
        self.sh_embed = None  # assigned externally if with_sh=True

    def forward(self,
                y, x,
                pe,
                cam_feat,
                sh_col=None
               ):
        feat = self._feat(y, x, cam_feat)
        fuv  = fourier_uv(x, y)
        rgb, alpha = self.heads(feat, pe, fuv, sh_col=sh_col)
        return rgb, alpha


class RefractionSheet(SheetBase):
    """
    Rear sheet as a tri-component transport model.
    We predict three nearby sample points on the rear plane, evaluate one head
    at each point, and route them with a learned softmax gate.
    """
    def __init__(self, h: int, w: int, with_sh: bool = False):
        super().__init__(h, w)
        self.with_sh = with_sh
        self.sh_embed = None
        router_dim = POS_OUT + PE_DIM + 4
        self.layout_mlp = nn.Sequential(
            make_mlp(router_dim, 256, 4, depth=4),
            nn.Tanh(),
        )
        self.gate_mlp = make_mlp(router_dim, 256, 3, depth=3)
        self.route_strength_mlp = nn.Sequential(
            make_mlp(router_dim, 256, 1, depth=2),
            nn.Sigmoid(),
        )
        route_last = self.route_strength_mlp[0][-1]
        nn.init.zeros_(route_last.weight)
        nn.init.constant_(route_last.bias, 4.0)
        base_in_spec = POS_OUT + PE_DIM + 2
        resid_in_spec = POS_OUT * 2 + PE_DIM + 2 + 4
        if self.with_sh:
            base_in_spec += 3
            resid_in_spec += 3
        self.component_heads = nn.ModuleList([
            make_mlp(base_in_spec, HEAD_HID, 3, siren=True),
            make_mlp(resid_in_spec, HEAD_HID, 3, siren=True),
            make_mlp(resid_in_spec, HEAD_HID, 3, siren=True),
        ])

        self.alpha_mlp = nn.Sequential(
            make_mlp(router_dim + 3, 256, 1, depth=2),
            nn.Sigmoid()
        )

    def forward(self,
                y, x,
                plane_hit:  torch.Tensor,
                ray_dir:    torch.Tensor,
                cam_feat:   torch.Tensor,
                front_alpha: Optional[torch.Tensor] = None,
                front_rgb: Optional[torch.Tensor] = None,
                offset_scale: float = 1.0,
                gate_temperature: float = 1.0,
                hard_gate_temperature: float = HARD_GATE_TEMPERATURE,
                adaptive_router_strength: float = 1.0,
                return_aux: bool = False,
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return (color, alpha, offset_raw)
          color shape => (B,3)
          alpha => (B,)
          offset_raw => (B,4)
        """
        base_feat = self._feat(y, x, cam_feat)
        if front_alpha is None:
            front_alpha = torch.zeros_like(y, dtype=base_feat.dtype)
        else:
            front_alpha = front_alpha.to(dtype=base_feat.dtype)
        if front_rgb is None:
            front_rgb = torch.zeros((y.shape[0], 3), device=base_feat.device, dtype=base_feat.dtype)
        else:
            front_rgb = front_rgb.to(dtype=base_feat.dtype)

        view_pe = torch.cat([encode_dir(ray_dir), encode_vec(plane_hit)], -1).to(dtype=base_feat.dtype)
        router_in = torch.cat([base_feat, view_pe, front_alpha.unsqueeze(-1), front_rgb], dim=-1)

        layout_raw = self.layout_mlp(router_in).clamp(-1.0, 1.0)
        if offset_scale <= 0.0:
            span_px = torch.zeros((y.shape[0],), device=base_feat.device, dtype=base_feat.dtype)
            fan_px = torch.zeros_like(span_px)
            relative_px = torch.zeros((y.shape[0], 3, 2), device=base_feat.device, dtype=base_feat.dtype)
        else:
            axis_raw = layout_raw[:, :2]
            axis = F.normalize(axis_raw + 1e-6, dim=-1)
            perp = torch.stack([-axis[:, 1], axis[:, 0]], dim=-1)
            min_span = BASE_MIN_PAIR_SEPARATION_PX * offset_scale
            span_px = min_span + 0.5 * (layout_raw[:, 2] + 1.0) * (MAX_RELATIVE_OFFSET_PX * offset_scale - min_span)
            fan_px = layout_raw[:, 3] * (MAX_FAN_SHIFT_PX * offset_scale)
            center = torch.zeros((y.shape[0], 2), device=base_feat.device, dtype=base_feat.dtype)
            pos = axis * span_px.unsqueeze(-1) + perp * fan_px.unsqueeze(-1)
            neg = -axis * span_px.unsqueeze(-1) + perp * fan_px.unsqueeze(-1)
            relative_px = torch.stack([center, pos, neg], dim=1)
        offset_px = relative_px
        offset_raw = layout_raw
        sample_x = x.unsqueeze(1).float() + offset_px[..., 0]
        sample_y = y.unsqueeze(1).float() + offset_px[..., 1]
        sample_count = sample_x.shape[0] * sample_x.shape[1]
        sample_valid = (
            (sample_x >= 0.0) & (sample_x <= (RES - 1)) &
            (sample_y >= 0.0) & (sample_y <= (RES - 1))
        )

        gate_logits = self.gate_mlp(router_in)
        route_strength_raw = self.route_strength_mlp(router_in).squeeze(-1)
        adaptive_mix = min(max(float(adaptive_router_strength), 0.0), 1.0)
        gate_temp_easy = max(float(gate_temperature), 1e-4)
        gate_temp_hard = min(gate_temp_easy, max(float(hard_gate_temperature), 1e-4))
        if adaptive_mix <= 0.0:
            route_strength = torch.ones_like(route_strength_raw)
            gate_temp_ray = torch.full_like(route_strength_raw, gate_temp_easy)
        else:
            route_strength = torch.lerp(
                torch.ones_like(route_strength_raw),
                route_strength_raw,
                adaptive_mix,
            )
            gate_temp_pred = gate_temp_easy + (gate_temp_hard - gate_temp_easy) * route_strength_raw.float()
            gate_temp_ray = torch.lerp(
                torch.full_like(route_strength_raw.float(), gate_temp_easy),
                gate_temp_pred,
                adaptive_mix,
            ).to(dtype=route_strength_raw.dtype)
        soft_gates = F.softmax(gate_logits.float() / gate_temp_ray.float().unsqueeze(-1), dim=-1)
        if adaptive_mix > 0.0:
            uniform_gates = torch.full_like(soft_gates, 1.0 / 3.0)
            gates = torch.lerp(uniform_gates, soft_gates, route_strength.float().unsqueeze(-1)).to(dtype=base_feat.dtype)
        else:
            gates = soft_gates.to(dtype=base_feat.dtype)

        cam_feat_samples = cam_feat.unsqueeze(0).expand(sample_count, -1)
        sample_feat = self._feat(
            sample_y.reshape(-1),
            sample_x.reshape(-1),
            cam_feat_samples,
        ).view(sample_y.shape[0], sample_y.shape[1], -1)
        center_feat = sample_feat[:, 0]
        zero_offset = torch.zeros_like(relative_px[:, 0])
        base_in = torch.cat([center_feat, view_pe, zero_offset], dim=-1)
        sample_sh = None
        if self.with_sh and self.sh_embed is not None:
            ray_dir_samples = ray_dir.unsqueeze(1).expand(-1, sample_y.shape[1], -1).reshape(-1, ray_dir.shape[-1])
            sample_sh = self.sh_embed(
                safe_px(sample_y.reshape(-1)),
                safe_px(sample_x.reshape(-1)),
                ray_dir_samples,
            ).view(sample_y.shape[0], sample_y.shape[1], -1)
            sh_center = sample_sh[:, 0]
            base_in = torch.cat([base_in, sh_center.to(dtype=center_feat.dtype)], dim=-1)
        base_rgb = torch.sigmoid(self.component_heads[0](base_in))

        flank_components = []
        residual_rgb = []
        for k, head in ((1, self.component_heads[1]), (2, self.component_heads[2])):
            feat_k = sample_feat[:, k]
            feat_delta = feat_k - center_feat
            resid_in = torch.cat([
                center_feat,
                feat_delta,
                view_pe,
                (relative_px[:, k] / max(MAX_RELATIVE_OFFSET_PX, 1e-6)).to(dtype=feat_k.dtype),
                front_alpha.unsqueeze(-1),
                front_rgb,
            ], dim=-1)
            if sample_sh is not None:
                sh_c = sample_sh[:, k]
                resid_in = torch.cat([resid_in, sh_c.to(dtype=feat_k.dtype)], dim=-1)
            resid_strength = (
                relative_px[:, k].norm(dim=-1, keepdim=True)
                / max(MAX_RELATIVE_OFFSET_PX, 1e-6)
            ).to(dtype=feat_k.dtype)
            resid_strength = resid_strength * sample_valid[:, k].to(dtype=feat_k.dtype).unsqueeze(-1)
            resid_delta = torch.tanh(head(resid_in)) * (RESIDUAL_RGB_SCALE * resid_strength)
            residual_rgb.append(resid_delta)
            flank_components.append((base_rgb + resid_delta).clamp(0.0, 1.0))

        head_rgb = torch.stack([base_rgb, flank_components[0], flank_components[1]], dim=1)
        transport_rgb = (gates.unsqueeze(-1) * head_rgb).sum(dim=1)

        alpha_in = torch.cat([router_in, gates], dim=-1)
        alpha_val = self.alpha_mlp(alpha_in).squeeze(-1)

        if return_aux:
            aux = {
                "head_rgb": head_rgb,
                "transport_rgb": transport_rgb,
                "gates": gates,
                "gate_logits": gate_logits,
                "offset_px": offset_px,
                "relative_px": relative_px,
                "span_px": span_px,
                "fan_px": fan_px,
                "base_rgb": base_rgb,
                "residual_rgb": torch.stack(residual_rgb, dim=1),
                "sample_x": sample_x,
                "sample_y": sample_y,
                "sample_valid": sample_valid,
                "soft_gates": soft_gates,
                "route_strength": route_strength,
                "route_strength_raw": route_strength_raw,
                "gate_temperature_ray": gate_temp_ray,
            }
            return transport_rgb, alpha_val, offset_raw, aux

        return transport_rgb, alpha_val, offset_raw


# ───────── sobel, pooling, etc. ───────── #
SOBEL_BASE = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32)
AVG2       = nn.AvgPool2d(2,2)
UP_VEC     = torch.tensor([0.,1.,0.], dtype=torch.float32)

def _read_image_u8(img_path: Path) -> torch.Tensor:
    arr = np.asarray(imageio.imread(img_path))
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def parse_manifest_and_center(img_dir: Path,
                              jsonl_name="views.jsonl",
                              depth_suffix="_depth.npy"):
    meta, locs = [], []
    running_sum  = torch.zeros(RES, RES, 3)
    running_ssum = torch.zeros(RES, RES, 3)
    img_cnt      = 0

    with open(img_dir / jsonl_name) as f:
        lines = f.readlines()

    for ln in sorted(lines):
        r = json.loads(ln)

        # ----- convert Blender Z-up → Y-up and compute rho -----
        phi   = r["phi"]
        theta = r["theta"]
        rho   = r["rho"]                          # already stored in manifest

        cam = torch.tensor([                     # Y-up world frame
                rho * math.cos(theta) * math.cos(phi),   # X
                rho * math.sin(theta),                   # Y  (was Z in Blender)
                rho * math.cos(theta) * math.sin(phi)    # Z  (was Y in Blender)
            ], dtype=torch.float32)

        # Use the true look-at basis from the camera position.
        R = look_at_rotation(cam, device="cpu")

        # -------------------------------------------------------

        img_path = img_dir / r["file"]
        stem     = img_path.stem
        cand1    = img_path.parent / f"{stem}{depth_suffix}"
        cand2    = img_path.parent / f"{stem}.npy"
        depth_path = cand1 if cand1.is_file() else (cand2 if cand2.is_file() else None)

        img_u8 = _read_image_u8(img_path)
        meta.append({
            "path":   img_path,
            "R":      R,
            "loc":    cam,
            "phi":    phi,
            "theta":  theta,
            "rho":    rho,
            "depth":  depth_path,
            "img_u8": img_u8,
        })
        locs.append(cam)

        img = img_u8.permute(1, 2, 0).float() / 255.0         # H,W,C
        running_sum  += img
        running_ssum += img * img
        img_cnt      += 1

    locs   = torch.stack(locs)

    mean_img = running_sum  / img_cnt
    var_img  = running_ssum / img_cnt - mean_img**2
    var_mag  = var_img.var(dim=-1).sqrt()
    w_var    = 0.5 + var_mag / var_mag.mean()

    vis_count = torch.ones(RES, RES, dtype=torch.float32, device=DEVICE)
    return meta, w_var.to(DEVICE), vis_count.to(DEVICE)


def _build_image_cache(meta: list[dict]) -> torch.Tensor:
    total_bytes = len(meta) * 3 * RES * RES
    print(f"building RAM image cache for {len(meta)} views (~{total_bytes / (1 << 30):.2f} GiB uint8 CPU)")
    cache = torch.empty((len(meta), 3, RES, RES), dtype=torch.uint8)
    for idx, rec in enumerate(meta):
        cache[idx].copy_(rec["img_u8"])
        if (idx + 1) % 128 == 0 or (idx + 1) == len(meta):
            print(f"  cached {idx + 1}/{len(meta)} images")
    return cache


FOV_DEG = 60.0
FOCAL   = RES / (2*math.tan(math.radians(FOV_DEG/2)))
def pixel_rays(y, x, Rwc, cam_loc, z=0.0):
    dx = (x + 0.5 - RES/2) / FOCAL
    dy = (y + 0.5 - RES/2) / FOCAL
    d_cam = torch.stack([dx, -dy, torch.ones_like(dx)], -1)
    v_w   = (Rwc.t() @ d_cam.t()).t()
    direction = F.normalize(v_w, dim=-1)
    origin    = cam_loc.unsqueeze(0).expand_as(direction) + direction*z
    return origin, direction


def _module_grad_norm(module: nn.Module) -> float:
    total = 0.0
    for p in module.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        total += float((g * g).sum().item())
    return math.sqrt(total) if total > 0.0 else 0.0


def _set_module_trainable(module: nn.Module, trainable: bool) -> None:
    for p in module.parameters():
        p.requires_grad_(trainable)


def _tensor_stats(prefix: str, tensor: torch.Tensor) -> dict[str, float]:
    t = tensor.detach().float()
    return {
        f"{prefix}_mean": float(t.mean().item()),
        f"{prefix}_std": float(t.std().item()),
        f"{prefix}_min": float(t.min().item()),
        f"{prefix}_max": float(t.max().item()),
    }


def _entropy_from_probs(probs: torch.Tensor, dim: int = -1) -> torch.Tensor:
    p = probs.clamp(1e-6, 1.0)
    return -(p * p.log()).sum(dim=dim)


def _pairwise_offset_dist_px(offset_px: torch.Tensor) -> torch.Tensor:
    pair_vals = _pairwise_offset_pair_dists_px(offset_px)
    return pair_vals.mean(dim=1)


def _pairwise_offset_pair_dists_px(offset_px: torch.Tensor) -> torch.Tensor:
    pair_vals = []
    for i, j in ((0, 1), (0, 2), (1, 2)):
        pair_vals.append((offset_px[:, i] - offset_px[:, j]).norm(dim=-1))
    return torch.stack(pair_vals, dim=1)


def _pairwise_offset_stats_px(offset_px: torch.Tensor) -> dict[str, float]:
    pair_vals = []
    for i, j in ((0, 1), (0, 2), (1, 2)):
        pair_vals.append((offset_px[:, i] - offset_px[:, j]).norm(dim=-1))
    all_pairs = torch.stack(pair_vals, dim=1)
    return {
        "rear_pair_spread_px_mean": float(all_pairs.mean().item()),
        "rear_pair_spread_px_std": float(all_pairs.std().item()),
        "rear_pair_spread_px_max": float(all_pairs.max().item()),
    }


def _enforce_min_pair_separation(relative_px: torch.Tensor, min_sep_px: torch.Tensor | float, iters: int = 2) -> torch.Tensor:
    rel = relative_px
    if torch.is_tensor(min_sep_px):
        min_sep = min_sep_px.to(device=rel.device, dtype=rel.dtype).view(-1, 1)
    else:
        min_sep = torch.full((rel.size(0), 1), float(min_sep_px), device=rel.device, dtype=rel.dtype)
    min_sep = min_sep.clamp_min(0.0)
    canonical = rel.new_tensor([
        [1.0, 0.0],
        [-0.5, 0.8660254],
        [-0.5, -0.8660254],
    ])
    pair_fallbacks = {
        (0, 1): canonical[0] - canonical[1],
        (0, 2): canonical[0] - canonical[2],
        (1, 2): canonical[1] - canonical[2],
    }
    for _ in range(iters):
        for i, j in ((0, 1), (0, 2), (1, 2)):
            diff = rel[:, i] - rel[:, j]
            dist = diff.norm(dim=-1, keepdim=True)
            fallback = pair_fallbacks[(i, j)].unsqueeze(0).expand_as(diff)
            fallback = fallback / fallback.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            dir_ij = torch.where(dist > 1e-4, diff / dist.clamp_min(1e-4), fallback)
            push = (min_sep - dist).clamp_min(0.0) * 0.5
            rel = rel.clone()
            rel[:, i] = rel[:, i] + push * dir_ij
            rel[:, j] = rel[:, j] - push * dir_ij
        rel = rel - rel.mean(dim=1, keepdim=True)
    return rel


def _luma_edge_strength_map(img: torch.Tensor, sobel_x: torch.Tensor, sobel_y: torch.Tensor) -> torch.Tensor:
    luma = (0.299 * img[0:1] + 0.587 * img[1:2] + 0.114 * img[2:3]).unsqueeze(0)
    gx = F.conv2d(luma, sobel_x, padding=1)
    gy = F.conv2d(luma, sobel_y, padding=1)
    mag = torch.sqrt(gx.square() + gy.square() + 1e-8)[0, 0]
    return mag / (mag.max() + 1e-6)


def _head_residual_decorrelation(head_rgb: torch.Tensor, gt_rgb: torch.Tensor) -> torch.Tensor:
    resid = head_rgb.float() - gt_rgb.float().unsqueeze(1)
    corrs = []
    for i, j in ((0, 1), (0, 2), (1, 2)):
        ri = resid[:, i].reshape(-1)
        rj = resid[:, j].reshape(-1)
        ri = ri - ri.mean()
        rj = rj - rj.mean()
        denom = ri.std(unbiased=False) * rj.std(unbiased=False) + 1e-6
        corrs.append(((ri * rj).mean() / denom).abs())
    return torch.stack(corrs).mean()


def _scale_module_grads(module: nn.Module, scale: float) -> None:
    for p in module.parameters():
        if p.grad is not None:
            p.grad.mul_(scale)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _normalize_hard_focus(focus: torch.Tensor) -> torch.Tensor:
    if PHASE2_HARD_FOCUS_FLOOR >= 1.0:
        return torch.ones_like(focus)
    denom = max(1e-6, 1.0 - PHASE2_HARD_FOCUS_FLOOR)
    return ((focus - PHASE2_HARD_FOCUS_FLOOR) / denom).clamp(0.0, 1.0)


def _normalize_batch_signal(signal: torch.Tensor, lo_q: float = 0.35, hi_q: float = 0.85) -> torch.Tensor:
    s = signal.detach().float()
    lo = torch.quantile(s, lo_q)
    hi = torch.quantile(s, hi_q)
    scale = (hi - lo).clamp_min(1e-6)
    return ((s - lo) / scale).clamp(0.0, 1.0)


def _slab_schedule_state(step: int, head_div_mean: float, spread_mean: float) -> dict[str, float]:
    if SLAB_RAMP_STEPS <= 0:
        step_ready = 1.0
    else:
        step_ready = min(max((float(step) - float(SLAB_RAMP_WARMUP_STEPS)) / float(max(1, SLAB_RAMP_STEPS)), 0.0), 1.0)
    if SLAB_RAMP_HEAD_DIV_THRESHOLD <= 0.0:
        head_ready = 1.0
    else:
        head_ready = min(max(float(head_div_mean) / float(SLAB_RAMP_HEAD_DIV_THRESHOLD), 0.0), 1.0)
    if SLAB_RAMP_SPREAD_THRESHOLD <= 0.0:
        spread_ready = 1.0
    else:
        spread_ready = min(max(float(spread_mean) / float(SLAB_RAMP_SPREAD_THRESHOLD), 0.0), 1.0)
    structure_ready = 0.5 * (max(head_ready, spread_ready) + math.sqrt(max(0.0, head_ready * spread_ready)))
    structure_ready = min(max(structure_ready, 0.0), 1.0)
    schedule_raw = step_ready * structure_ready
    scale = _lerp(SLAB_RAMP_START_SCALE, 1.0, schedule_raw)
    return {
        "scale": float(scale),
        "step_ready": float(step_ready),
        "head_ready": float(head_ready),
        "spread_ready": float(spread_ready),
        "structure_ready": float(structure_ready),
    }


def _schedule_state(global_step: int, target_step: int) -> dict[str, float | bool | str]:
    phase2_active = PHASE1_STEPS > 0 and global_step >= PHASE1_STEPS
    if not phase2_active:
        return {
            "stage_name": "phase1",
            "phase2_active": False,
            "phase3_active": False,
            "relax_mix": 0.0,
            "base_grad_scale": 1.0,
            "residual_gain_mult": 1.0,
            "spread_floor_scale": 0.0,
            "edge_spread_mult": 1.0,
            "pair_repulsion_mult": 1.0,
            "gate_temperature": 1.0,
        }

    post_phase1_span = max(1, target_step - PHASE1_STEPS)
    phase2_progress = min(max((global_step - PHASE1_STEPS) / post_phase1_span, 0.0), 1.0)
    phase3_start = min(max(PHASE3_START_RATIO, 0.0), 0.999)
    phase3_active = phase2_progress >= phase3_start
    relax_mix = 0.0
    if phase3_active:
        relax_mix = min(
            max((phase2_progress - phase3_start) / max(1e-6, 1.0 - phase3_start), 0.0),
            1.0,
        )

    return {
        "stage_name": "phase3" if phase3_active else "phase2",
        "phase2_active": True,
        "phase3_active": phase3_active,
        "relax_mix": relax_mix,
        "base_grad_scale": PHASE2_BASE_GRAD_SCALE,
        "residual_gain_mult": PHASE2_RESIDUAL_GAIN_MULT,
        "spread_floor_scale": _lerp(1.0, PHASE3_END_SPREAD_FLOOR_SCALE, relax_mix),
        "edge_spread_mult": _lerp(PHASE2_EDGE_SPREAD_MULT, PHASE3_END_EDGE_SPREAD_MULT, relax_mix),
        "pair_repulsion_mult": _lerp(PHASE2_PAIR_REPULSION_MULT, PHASE3_END_PAIR_REPULSION_MULT, relax_mix),
        "gate_temperature": _lerp(1.0, PHASE3_END_GATE_TEMPERATURE, relax_mix),
    }


def _three_ray_diag_payload(
    step: int,
    loss: torch.Tensor,
    l1: torch.Tensor,
    base_l1: torch.Tensor,
    no_slab_l1: torch.Tensor,
    residual_gain_reward: torch.Tensor,
    residual_gain_weight_curr: float,
    slab_gain_reward: torch.Tensor,
    slab_gain_weight_curr: float,
    slab_schedule_scale: float,
    slab_step_ready: float,
    slab_head_ready: float,
    slab_spread_ready: float,
    slab_structure_ready: float,
    residual_focus_mean: float,
    residual_focus_hi_frac: float,
    route_target_mean: float,
    route_disagreement_mean: float,
    base_grad_scale_curr: float,
    gate_temperature_curr: float,
    hard_gate_temperature_curr: float,
    schedule_relax_mix: float,
    spread_floor_scale_curr: float,
    hard_spread_floor_mean: float,
    edge_spread_weight_curr: float,
    pair_repulsion_weight_curr: float,
    base_frozen: bool,
    edge_l: torch.Tensor,
    tau_loss: torch.Tensor,
    offset_l2: torch.Tensor,
    slab_strength_loss: torch.Tensor,
    slab_opacity_loss: torch.Tensor,
    gate_usage_loss: torch.Tensor,
    head_decor_loss: torch.Tensor,
    route_strength_loss: torch.Tensor,
    route_entropy_loss: torch.Tensor,
    edge_spread_loss: torch.Tensor,
    pair_repulsion_loss: torch.Tensor,
    front: OcclusionSheet,
    rear: RefractionSheet,
    slab: GaussianSlab,
    camera_embed: CameraEmbed,
    alpha_front: torch.Tensor,
    tau0: torch.Tensor,
    rear_alpha: torch.Tensor,
    valid_f: torch.Tensor,
    valid_r: torch.Tensor,
    offset_raw: torch.Tensor,
    rear_aux: dict,
    slab_aux: dict,
    spread_target_px: torch.Tensor,
) -> dict[str, float]:
    alpha_front = alpha_front.detach().float()
    alpha_rear = rear_alpha.detach().float()
    slab_alpha = slab_aux["alpha"].detach().float()
    slab_tau = slab_aux["tau"].detach().float()
    slab_trans = slab_aux["trans"].detach().float()
    front_w = alpha_front
    rear_w = (1.0 - alpha_front) * slab_trans * alpha_rear

    head_rgb = rear_aux["head_rgb"].detach().float()
    transport_rgb = rear_aux["transport_rgb"].detach().float()
    gates = rear_aux["gates"].detach().float()
    offset_px = rear_aux["offset_px"].detach().float()
    relative_px = rear_aux["relative_px"].detach().float()
    span_px = rear_aux["span_px"].detach().float()
    fan_px = rear_aux["fan_px"].detach().float()
    base_rgb = rear_aux["base_rgb"].detach().float()
    residual_rgb = rear_aux["residual_rgb"].detach().float()
    sample_valid = rear_aux["sample_valid"].detach().float()
    route_strength = rear_aux["route_strength"].detach().float()
    route_strength_raw = rear_aux["route_strength_raw"].detach().float()
    gate_temperature_ray = rear_aux["gate_temperature_ray"].detach().float()
    gate_mean = gates.mean(dim=0)
    offset_mag = offset_px.norm(dim=-1)
    relative_mag = relative_px.norm(dim=-1)
    gate_entropy = _entropy_from_probs(gates, dim=-1)

    head_pair_l1 = torch.stack([
        (head_rgb[:, 0] - head_rgb[:, 1]).abs().mean(dim=-1),
        (head_rgb[:, 0] - head_rgb[:, 2]).abs().mean(dim=-1),
        (head_rgb[:, 1] - head_rgb[:, 2]).abs().mean(dim=-1),
    ], dim=1)

    diag = {
        "step": float(step),
        "loss": float(loss.item()),
        "l1": float(l1.item()),
        "base_l1": float(base_l1.item()),
        "no_slab_l1": float(no_slab_l1.item()),
        "residual_gain_reward": float(residual_gain_reward.item()),
        "residual_gain_weight_curr": float(residual_gain_weight_curr),
        "slab_gain_reward": float(slab_gain_reward.item()),
        "slab_gain_weight_curr": float(slab_gain_weight_curr),
        "slab_schedule_scale": float(slab_schedule_scale),
        "slab_step_ready": float(slab_step_ready),
        "slab_head_ready": float(slab_head_ready),
        "slab_spread_ready": float(slab_spread_ready),
        "slab_structure_ready": float(slab_structure_ready),
        "residual_focus_mean": float(residual_focus_mean),
        "residual_focus_hi_frac": float(residual_focus_hi_frac),
        "route_target_mean": float(route_target_mean),
        "route_disagreement_mean": float(route_disagreement_mean),
        "base_grad_scale_curr": float(base_grad_scale_curr),
        "gate_temperature_curr": float(gate_temperature_curr),
        "hard_gate_temperature_curr": float(hard_gate_temperature_curr),
        "schedule_relax_mix": float(schedule_relax_mix),
        "spread_floor_scale_curr": float(spread_floor_scale_curr),
        "hard_spread_floor_mean": float(hard_spread_floor_mean),
        "edge_spread_weight_curr": float(edge_spread_weight_curr),
        "pair_repulsion_weight_curr": float(pair_repulsion_weight_curr),
        "base_frozen": float(base_frozen),
        "edge_l": float(edge_l.item()),
        "tau_loss": float(tau_loss.item()),
        "offset_l2": float(offset_l2.item()),
        "slab_strength_loss": float(slab_strength_loss.item()),
        "slab_opacity_loss": float(slab_opacity_loss.item()),
        "gate_usage_loss": float(gate_usage_loss.item()),
        "head_decor_loss": float(head_decor_loss.item()),
        "route_strength_loss": float(route_strength_loss.item()),
        "route_entropy_loss": float(route_entropy_loss.item()),
        "edge_spread_loss": float(edge_spread_loss.item()),
        "pair_repulsion_loss": float(pair_repulsion_loss.item()),
        "valid_front_frac": float(valid_f.float().mean().item()),
        "valid_rear_frac": float(valid_r.float().mean().item()),
        "rear_sample_valid_frac": float(sample_valid.mean().item()),
        "front_contrib_mean": float(front_w.mean().item()),
        "rear_contrib_mean": float(rear_w.mean().item()),
        "rear_dominates_frac": float((rear_w > front_w).float().mean().item()),
        "front_alpha_entropy": float(_entropy_from_probs(torch.stack([alpha_front, 1.0 - alpha_front], dim=-1), dim=-1).mean().item()),
        "rear_alpha_entropy": float(_entropy_from_probs(torch.stack([alpha_rear, 1.0 - alpha_rear], dim=-1), dim=-1).mean().item()),
        "front_alpha_sat_frac": float(((alpha_front < 0.05) | (alpha_front > 0.95)).float().mean().item()),
        "rear_alpha_sat_frac": float(((alpha_rear < 0.05) | (alpha_rear > 0.95)).float().mean().item()),
        "gate_entropy_mean": float(gate_entropy.mean().item()),
        "gate_entropy_std": float(gate_entropy.std().item()),
        "gate_usage_entropy": float(_entropy_from_probs(gate_mean.unsqueeze(0), dim=-1).item()),
        "gate_peak_mean": float(gates.max(dim=-1).values.mean().item()),
        "gate_mean_0": float(gate_mean[0].item()),
        "gate_mean_1": float(gate_mean[1].item()),
        "gate_mean_2": float(gate_mean[2].item()),
        "route_strength_mean": float(route_strength.mean().item()),
        "route_strength_raw_mean": float(route_strength_raw.mean().item()),
        "route_strength_hi_frac": float((route_strength > 0.75).float().mean().item()),
        "gate_temperature_ray_mean": float(gate_temperature_ray.mean().item()),
        "gate_temperature_ray_std": float(gate_temperature_ray.std().item()),
        "rear_offset_mag_mean": float(offset_mag.mean().item()),
        "rear_offset_mag_std": float(offset_mag.std().item()),
        "rear_offset_mag_max": float(offset_mag.max().item()),
        "rear_relative_mag_mean": float(relative_mag.mean().item()),
        "rear_relative_mag_std": float(relative_mag.std().item()),
        "rear_span_px_mean": float(span_px.mean().item()),
        "rear_span_px_std": float(span_px.std().item()),
        "rear_fan_px_mean": float(fan_px.mean().item()),
        "rear_fan_px_std": float(fan_px.std().item()),
        "rear_head_pair_l1_mean": float(head_pair_l1.mean().item()),
        "rear_head_pair_l1_std": float(head_pair_l1.std().item()),
        "rear_transport_vs_headmean_l1": float((transport_rgb - head_rgb.mean(dim=1)).abs().mean().item()),
        "rear_base_rgb_mean": float(base_rgb.mean().item()),
        "rear_residual_abs_mean": float(residual_rgb.abs().mean().item()),
        "rear_residual_abs_max": float(residual_rgb.abs().max().item()),
        "rear_gain_frac": float((base_l1.detach() > l1.detach()).float().mean().item()) if base_l1.ndim > 0 else 0.0,
        "slab_alpha_mean": float(slab_alpha.mean().item()),
        "slab_alpha_std": float(slab_alpha.std().item()),
        "slab_tau_mean": float(slab_tau.mean().item()),
        "slab_tau_std": float(slab_tau.std().item()),
        "slab_signal_mean": float(slab_aux["signal"].detach().float().mean().item()),
        "slab_signal_std": float(slab_aux["signal"].detach().float().std().item()),
        "slab_ray_gain_mean": float(slab_aux["ray_gain"].detach().float().mean().item()),
        "slab_ray_gain_std": float(slab_aux["ray_gain"].detach().float().std().item()),
        "slab_mass_peak_mean": float(slab_aux["mass_peak"].detach().float().mean().item()),
        "slab_depth_mean": float(slab_aux["depth_mean"].detach().float().mean().item()),
        "slab_strength_mean": float(slab_aux["strength_mean"].detach().float().item()),
        "slab_strength_max": float(slab_aux["strength_max"].detach().float().item()),
        "slab_scale_mean": float(slab_aux["scale_mean"].detach().float().item()),
        "slab_context_mod_mean": float(slab_aux["context_mod_mean"].detach().float().item()),
        "slab_context_mod_std": float(slab_aux["context_mod_std"].detach().float().item()),
        "slab_disagreement_gain": float(slab_aux["disagreement_gain"].detach().float().item()),
        "slab_context_scale": float(slab_aux["context_scale"].detach().float().item()),
        "spread_target_px_mean": float(spread_target_px.detach().float().mean().item()),
        "grad_front_heads": _module_grad_norm(front.heads),
        "grad_front_codes": _module_grad_norm(front.codes),
        "grad_rear_layout": _module_grad_norm(rear.layout_mlp),
        "grad_rear_gate": _module_grad_norm(rear.gate_mlp),
        "grad_rear_head0": _module_grad_norm(rear.component_heads[0]),
        "grad_rear_head1": _module_grad_norm(rear.component_heads[1]),
        "grad_rear_head2": _module_grad_norm(rear.component_heads[2]),
        "grad_rear_alpha": _module_grad_norm(rear.alpha_mlp),
        "grad_rear_codes": _module_grad_norm(rear.codes),
        "grad_slab": _module_grad_norm(slab),
        "grad_camera_embed": _module_grad_norm(camera_embed),
    }
    diag.update(_tensor_stats("front_tau", tau0))
    diag.update(_tensor_stats("front_alpha", alpha_front))
    diag.update(_tensor_stats("rear_alpha", alpha_rear))
    diag.update(_tensor_stats("slab_alpha", slab_alpha))
    diag.update(_tensor_stats("rear_offset_raw", offset_raw))
    diag.update(_tensor_stats("rear_transport_rgb", transport_rgb))
    diag.update(_pairwise_offset_stats_px(offset_px))
    return diag


def _emit_three_ray_diag(payload: dict[str, float], diag_jsonl: Optional[Path]) -> None:
    print(
        "DIAG "
        f"step={int(payload['step'])} "
        f"front_a={payload['front_alpha_mean']:.3f}+/-{payload['front_alpha_std']:.3f} "
        f"rear_a={payload['rear_alpha_mean']:.3f}+/-{payload['rear_alpha_std']:.3f} "
        f"rear_w={payload['rear_contrib_mean']:.3f} "
        f"gate=[{payload['gate_mean_0']:.2f},{payload['gate_mean_1']:.2f},{payload['gate_mean_2']:.2f}] "
        f"spread={payload['rear_pair_spread_px_mean']:.3f}px "
        f"span={payload['rear_span_px_mean']:.3f}px "
        f"fan={payload['rear_fan_px_mean']:.3f}px "
        f"gain={payload['residual_gain_reward']:.4f}@{payload['residual_gain_weight_curr']:.2f} "
        f"slab={payload['slab_alpha_mean']:.3f}/{payload['slab_gain_reward']:.4f}@{payload['slab_gain_weight_curr']:.2f} "
        f"slab_sig={payload['slab_signal_mean']:.2f}/{payload['slab_ray_gain_mean']:.2f} "
        f"slab_sched={payload['slab_schedule_scale']:.2f}[{payload['slab_step_ready']:.2f},{payload['slab_structure_ready']:.2f}] "
        f"focus={payload['residual_focus_mean']:.2f}/{payload['residual_focus_hi_frac']:.2f} "
        f"route={payload['route_strength_mean']:.2f}/{payload['route_strength_hi_frac']:.2f}@{payload['route_target_mean']:.2f} "
        f"gate_t={payload['gate_temperature_curr']:.2f}->{payload['gate_temperature_ray_mean']:.2f} "
        f"relax={payload['schedule_relax_mix']:.2f} "
        f"spread_t={payload['spread_target_px_mean']:.2f}/{payload['hard_spread_floor_mean']:.2f} "
        f"spread_s={payload['spread_floor_scale_curr']:.2f} "
        f"base_lr={payload['base_grad_scale_curr']:.2f} "
        f"frozen={int(payload['base_frozen'])} "
        f"head_div={payload['rear_head_pair_l1_mean']:.3f} "
        f"gate_H={payload['gate_usage_entropy']:.3f}"
    )
    print(
        "     "
        f"grads lay={payload['grad_rear_layout']:.3e} "
        f"gate={payload['grad_rear_gate']:.3e} "
        f"h0={payload['grad_rear_head0']:.3e} "
        f"h1={payload['grad_rear_head1']:.3e} "
        f"h2={payload['grad_rear_head2']:.3e} "
        f"alpha={payload['grad_rear_alpha']:.3e} "
        f"route={payload['route_strength_loss']:.3e} "
        f"rent={payload['route_entropy_loss']:.3e} "
        f"repel={payload['pair_repulsion_loss']:.3e}@{payload['pair_repulsion_weight_curr']:.2f} "
        f"spread={payload['edge_spread_loss']:.3e}@{payload['edge_spread_weight_curr']:.2f}"
    )
    if diag_jsonl is not None:
        diag_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with diag_jsonl.open("a") as f:
            f.write(json.dumps(payload) + "\n")

def train(meta, w_var, vis_count,
          sh_embed_front: Optional[SHEmbed],
          sh_embed_rear:  Optional[SHEmbed]):

    ck = torch.load(CHECKPOINT_PATH, map_location="cpu") if os.path.exists(CHECKPOINT_PATH) else None
    slab_splats = SLAB_SPLATS
    if ck is not None and "slab" in ck and "mean_raw" in ck["slab"]:
        slab_splats = int(ck["slab"]["mean_raw"].shape[0])
    front = OcclusionSheet(RES, RES, with_sh=(sh_embed_front is not None)).to(DEVICE)
    rear  = RefractionSheet(RES, RES, with_sh=(sh_embed_rear  is not None)).to(DEVICE)
    slab = GaussianSlab(num_splats=slab_splats).to(DEVICE)
    z_rear_offset = nn.Parameter(torch.tensor(0.30, device=DEVICE))
    camera_embed  = CameraEmbed().to(DEVICE)

    # Assign the loaded SH modules
    if sh_embed_front is not None:
        front.sh_embed = sh_embed_front
    if sh_embed_rear is not None:
        rear.sh_embed = sh_embed_rear

    errmap = torch.full((RES, RES), 1e-3, device=DEVICE)
    EMA    = 0.80

    global_step = 0
    if ck is not None:
        front.load_state_dict(ck["front"], strict=False)
        rear.load_state_dict(ck["rear"], strict=False)
        if "slab" in ck:
            slab.load_state_dict(ck["slab"], strict=False)
        if "camera_embed" in ck:
            camera_embed.load_state_dict(ck["camera_embed"], strict=False)
        z_rear_offset.data.copy_(torch.tensor(ck.get("z_offset", 0.30)))
        global_step = ck.get("global_step", 0)
        print(f"✔ resumed from step {global_step}")

    # For the new heads, if the old checkpoint didn't have those extra weights, 
    # PyTorch will skip them or not find them. That's normal; no error.


    if sh_embed_front:
        for p in sh_embed_front.parameters():
            p.requires_grad = False
    if sh_embed_rear:
        for p in sh_embed_rear.parameters():
            p.requires_grad = False

    image_cache = _build_image_cache(meta)
    meta_R = torch.stack([rec["R"] for rec in meta]).to(DEVICE)
    meta_loc = torch.stack([rec["loc"] for rec in meta]).to(DEVICE)
    meta_phi = torch.tensor([rec["phi"] for rec in meta], device=DEVICE)
    meta_theta = torch.tensor([rec["theta"] for rec in meta], device=DEVICE)
    meta_rho = torch.tensor([rec["rho"] for rec in meta], device=DEVICE)
    meta_depths = [rec["depth"] for rec in meta]
    val_stride = 8
    val_ys = torch.arange(0, RES, val_stride, device=DEVICE)
    val_xs = torch.arange(0, RES, val_stride, device=DEVICE)
    val_yy, val_xx = torch.meshgrid(val_ys, val_xs, indexing="ij")
    val_yyf, val_xxf = val_yy.flatten(), val_xx.flatten()

    target_step = ITERS_TOTAL if EXTRA_STEPS <= 0 else min(ITERS_TOTAL, global_step + EXTRA_STEPS)
    schedule_state = _schedule_state(global_step, target_step)
    phase2_active = bool(schedule_state["phase2_active"])
    base_head_trainable = not (FREEZE_BASE_AFTER_PHASE1 and phase2_active)
    base_grad_scale_curr = float(schedule_state["base_grad_scale"])
    gate_temperature_curr = float(schedule_state["gate_temperature"])
    schedule_stage_name = str(schedule_state["stage_name"])
    _set_module_trainable(rear.component_heads[0], base_head_trainable)
    if PHASE1_STEPS > 0:
        print(
            f"schedule init: {schedule_stage_name} at step {global_step} "
            f"(phase1_steps={PHASE1_STEPS}, residual_gain_mult={float(schedule_state['residual_gain_mult']):.2f}, "
            f"spread_floor_scale={float(schedule_state['spread_floor_scale']):.2f}, "
            f"edge_mult={float(schedule_state['edge_spread_mult']):.2f}, "
            f"repel_mult={float(schedule_state['pair_repulsion_mult']):.2f}, "
            f"gate_t={gate_temperature_curr:.2f}, "
            f"relax={float(schedule_state['relax_mix']):.2f}, "
            f"base_grad_scale={base_grad_scale_curr:.2f}, base_trainable={int(base_head_trainable)})"
        )

    shared = {id(p) for p in front.heads.parameters()}
    nonslab_params = [
        z_rear_offset,
        *front.parameters(),
        *[p for p in rear.parameters() if id(p) not in shared],
        *camera_embed.parameters()
    ]
    slab_params = list(slab.parameters())
    # remove the (already-frozen) SH parameters from trainable set
    # (They won't appear in front/rear anyway if we do it as is, because they arn't registered as submodules.)
    opt = torch.optim.AdamW([
        {"params": nonslab_params, "lr": LR_BASE * NON_SLAB_LR_SCALE},
        {"params": slab_params, "lr": LR_BASE * SLAB_LR_SCALE},
    ], lr=LR_BASE)
    sched_span = max(1, ITERS_TOTAL - WARMUP_STEPS)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, sched_span)

    sobel_x = SOBEL_BASE.to(DEVICE).view(1,1,3,3).repeat(3,1,1,1)
    sobel_y = sobel_x.transpose(2,3)
    sobel_gray_x = SOBEL_BASE.to(DEVICE).view(1, 1, 3, 3)
    sobel_gray_y = sobel_gray_x.transpose(2, 3)

    free, _ = torch.cuda.mem_get_info() if DEVICE == "cuda" else (0,0)
    usable  = free - SAFETY_GB * (1<<30) if DEVICE=="cuda" else 0
    CHUNK   = TRAIN_CHUNK
    print(f"chunk size = {CHUNK}")

    rng_cpu  = torch.Generator()
    rng_cuda = torch.Generator(device=DEVICE)

    loss_hist = collections.deque(maxlen=100)
    dropout_hist = collections.deque(maxlen=1000)
    val_psnr_ema = None

    start_t = time.time()
    next_ck = ((global_step // CKPT_EVERY) + 1) * CKPT_EVERY
    torch.set_default_dtype(torch.float32)

    scaler = GradScaler()
    print(f"target step = {target_step}")

    while global_step < target_step:
        idxs = torch.randperm(len(meta), generator=rng_cpu)[:CHUNK]
        idxs_dev = idxs.to(device=DEVICE)
        imgs = image_cache.index_select(0, idxs).to(device=DEVICE, dtype=torch.float32).div_(255.0)
        Rcs = meta_R.index_select(0, idxs_dev)
        Lcs = meta_loc.index_select(0, idxs_dev)
        phis = meta_phi.index_select(0, idxs_dev)
        thetas = meta_theta.index_select(0, idxs_dev)
        rhos = meta_rho.index_select(0, idxs_dev)
        depths = [meta_depths[int(i)] for i in idxs.tolist()]

        steps_here = min(max(1, (len(idxs)*RES*RES)//BATCH),
                         target_step - global_step)

        for _ in range(steps_here):
            schedule_state = _schedule_state(global_step, target_step)
            phase2_active = bool(schedule_state["phase2_active"])
            base_grad_scale_curr = float(schedule_state["base_grad_scale"])
            gate_temperature_curr = float(schedule_state["gate_temperature"])
            hard_gate_temperature_curr = HARD_GATE_TEMPERATURE
            adaptive_router_strength_curr = float(schedule_state["relax_mix"])
            schedule_stage_curr = str(schedule_state["stage_name"])
            want_base_trainable = not (FREEZE_BASE_AFTER_PHASE1 and phase2_active)
            if want_base_trainable != base_head_trainable or schedule_stage_curr != schedule_stage_name:
                base_head_trainable = want_base_trainable
                schedule_stage_name = schedule_stage_curr
                _set_module_trainable(rear.component_heads[0], base_head_trainable)
                print(
                    f"schedule switch: {schedule_stage_name} at step {global_step} "
                    f"(residual_gain_mult={float(schedule_state['residual_gain_mult']):.2f}, "
                    f"spread_floor_scale={float(schedule_state['spread_floor_scale']):.2f}, "
                    f"edge_mult={float(schedule_state['edge_spread_mult']):.2f}, "
                    f"repel_mult={float(schedule_state['pair_repulsion_mult']):.2f}, "
                    f"gate_t={gate_temperature_curr:.2f}->{hard_gate_temperature_curr:.2f}, "
                    f"route_mix={adaptive_router_strength_curr:.2f}, "
                    f"relax={float(schedule_state['relax_mix']):.2f}, "
                    f"base_grad_scale={base_grad_scale_curr:.2f}, base_trainable={int(base_head_trainable)})"
                )

            fid = torch.randint(0, len(idxs), (), generator=rng_cpu).item()
            img, Rwc, loc = imgs[fid], Rcs[fid], Lcs[fid]
            phi_f, th_f, rh_f = phis[fid], thetas[fid], rhos[fid]
            depth_path = depths[fid]

            cam_feat = camera_embed(phi_f, th_f, rh_f)

            #  decide whether to DROP this view
            drop = (torch.rand((), generator=rng_cpu) < VIEW_DROPOUT)
            dropout_hist.append(int(drop.item()))
            if drop:
                # Evaluate quickly for val-PSNR
                with torch.no_grad():
                    ray_o, ray_d = pixel_rays(val_yyf, val_xxf, Rwc, loc, 0.0)
                    front_center, rear_center, plane_normal, plane_u, plane_v = plane_frame(
                        loc, z_rear_offset, device=DEVICE, dtype=ray_o.dtype
                    )
                    x_f, y_f, hit_f, t_f, valid_f = project_rays_to_plane(
                        ray_o, ray_d, front_center, plane_normal, plane_u, plane_v, RES
                    )
                    x_r, y_r, hit_r, t_r, valid_r = project_rays_to_plane(
                        ray_o, ray_d, rear_center, plane_normal, plane_u, plane_v, RES
                    )

                    pe_f = torch.cat([encode_dir(ray_d), encode_vec(hit_f)], -1)

                    sh_col_f = None
                    if sh_embed_front is not None:
                        sh_col_f = sh_embed_front(safe_px(y_f), safe_px(x_f), ray_d)

                    with autocast(enabled=(DEVICE == "cuda")):
                        rgb0, tau0 = front(y_f, x_f, pe_f,
                                           cam_feat, sh_col=sh_col_f)
                        rgb0 = rgb0.clamp(0,1) * valid_f.unsqueeze(-1).to(dtype=rgb0.dtype)
                        tau0 = torch.where(valid_f, tau0.clamp(0,1), torch.ones_like(tau0))
                        alpha0 = 1.0 - tau0

                        rear_rgb, rear_alpha, _, rear_aux = rear(
                            y_r, x_r, hit_r, ray_d, cam_feat,
                            front_alpha=alpha0.detach(),
                            front_rgb=rgb0.detach(),
                            gate_temperature=gate_temperature_curr,
                            hard_gate_temperature=hard_gate_temperature_curr,
                            adaptive_router_strength=adaptive_router_strength_curr,
                            return_aux=True,
                        )
                        rear_rgb = rear_rgb.clamp(0,1) * valid_r.unsqueeze(-1).to(dtype=rear_rgb.dtype)
                        rear_alpha = rear_alpha.clamp(0,1) * valid_r.to(dtype=rear_alpha.dtype)
                        slab_context = slab_ray_context(
                            rear_aux["head_rgb"].detach(),
                            rear_aux["gates"].detach(),
                            rear_aux["route_strength"].detach(),
                            alpha0.detach(),
                            rear_alpha.detach(),
                        )
                        slab_trans = slab(
                            hit_f, hit_r,
                            front_center, rear_center,
                            plane_normal, plane_u, plane_v,
                            valid_f, valid_r,
                            ray_context=slab_context,
                        )
                        rear_alpha_slab = rear_alpha * slab_trans.to(dtype=rear_alpha.dtype)

                    pred, _, _ = composite_two_planes(
                        rgb0, alpha0, t_f, valid_f,
                        rear_rgb, rear_alpha_slab, t_r, valid_r,
                    )
                    psnr_val = _psnr(pred, img[:, val_yyf, val_xxf].t())
                    val_psnr_ema = psnr_val.item() if val_psnr_ema is None else 0.99*val_psnr_ema + 0.01*psnr_val.item()
                continue

            # 50% top-k, 50% random
            topk_share = int(BATCH * 0.50)
            rand_share = BATCH - topk_share

            with torch.no_grad():
                w_map = errmap / vis_count
                flat_val, flat_idx = torch.topk(w_map.reshape(-1), topk_share, largest=True)
                yb1, xb1 = flat_idx // RES, flat_idx % RES
            yb2 = torch.randint(0, RES, (rand_share,), generator=rng_cuda, device=DEVICE)
            xb2 = torch.randint(0, RES, (rand_share,), generator=rng_cuda, device=DEVICE)

            yb = torch.cat([yb1, yb2])             # (B,)
            xb = torch.cat([xb1, xb2])             # (B,)
            B  = yb.shape[0]                       # number of centre-pixels
            K  = BUNDLE_K
            edge_strength_map = _luma_edge_strength_map(img, sobel_gray_x, sobel_gray_y)
            edge_target = edge_strength_map[yb, xb].unsqueeze(1).expand(-1, K).reshape(-1) * EDGE_SPREAD_TARGET_PX

            # ---------- update visit counter ---------------------------------------
            flat = yb * RES + xb
            vis_count.view(-1).index_add_(0, flat,
                                          torch.ones_like(flat,
                                                          dtype=vis_count.dtype))

            # ---------- bundle rays -------------------------------------------------
            offs = torch.randn(B, K, 2, device=DEVICE) * BUNDLE_STD_PX
            ys   = yb.unsqueeze(1).float() + offs[...,0]     # (B,K)
            xs   = xb.unsqueeze(1).float() + offs[...,1]     # (B,K)

            ray_o, ray_d = pixel_rays(ys.reshape(-1), xs.reshape(-1), Rwc, loc, 0.0)
            front_center, rear_center, plane_normal, plane_u, plane_v = plane_frame(
                loc, z_rear_offset, device=DEVICE, dtype=ray_o.dtype
            )
            x_f, y_f, hit_f, t_f, valid_f = project_rays_to_plane(
                ray_o, ray_d, front_center, plane_normal, plane_u, plane_v, RES
            )
            x_r, y_r, hit_r, t_r, valid_r = project_rays_to_plane(
                ray_o, ray_d, rear_center, plane_normal, plane_u, plane_v, RES
            )

            pe_f = torch.cat([encode_dir(ray_d),
                              encode_vec(hit_f)], -1)

            # ground-truth still one per centre pixel
            gt = img[:, yb, xb].t()

            # optional SH colour per sub-ray
            sh_col_f = None
            if sh_embed_front is not None:
                sh_col_f = sh_embed_front(
                    safe_px(y_f),
                    safe_px(x_f),
                    ray_d,
                )

            do_diag = DIAG_EVERY > 0 and ((global_step + 1) % DIAG_EVERY == 0)
            opt.zero_grad()
            with autocast():
                rgb0, tau0 = front(y_f,
                                   x_f,
                                   pe_f,
                                   cam_feat,
                                   sh_col=sh_col_f)
                rgb0 = rgb0.clamp(0,1) * valid_f.unsqueeze(-1).to(dtype=rgb0.dtype)
                tau0 = torch.where(valid_f, tau0.clamp(0,1), torch.ones_like(tau0))
                alpha0 = 1.0 - tau0
                tau0_subray = tau0
                alpha0_subray = alpha0

                # Rear
                rear_rgb, rear_alpha, offset_raw, rear_aux = rear(
                    y_r, x_r, hit_r, ray_d, cam_feat,
                    front_alpha=alpha0.detach(),
                    front_rgb=rgb0.detach(),
                    gate_temperature=gate_temperature_curr,
                    hard_gate_temperature=hard_gate_temperature_curr,
                    adaptive_router_strength=adaptive_router_strength_curr,
                    return_aux=True,
                )
                rear_rgb = rear_rgb.clamp(0,1) * valid_r.unsqueeze(-1).to(dtype=rear_rgb.dtype)
                rear_alpha = rear_alpha.clamp(0,1) * valid_r.to(dtype=rear_alpha.dtype)
                rear_base_rgb = rear_aux["base_rgb"].clamp(0,1) * valid_r.unsqueeze(-1).to(dtype=rear_rgb.dtype)
                slab_context = slab_ray_context(
                    rear_aux["head_rgb"].detach(),
                    rear_aux["gates"].detach(),
                    rear_aux["route_strength"].detach(),
                    alpha0.detach(),
                    rear_alpha.detach(),
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

                pred_subray, _, _ = composite_two_planes(
                    rgb0, alpha0, t_f, valid_f,
                    rear_rgb, rear_alpha_slab, t_r, valid_r,
                )
                base_pred_subray, _, _ = composite_two_planes(
                    rgb0, alpha0, t_f, valid_f,
                    rear_base_rgb, rear_alpha_slab, t_r, valid_r,
                )
                no_slab_pred_subray, _, _ = composite_two_planes(
                    rgb0, alpha0, t_f, valid_f,
                    rear_rgb, rear_alpha, t_r, valid_r,
                )

                # ---------- bundle mean --------------------------------------------
                pred      = pred_subray.view(B, K, 3).mean(1)               # (B,3)
                pred_base = base_pred_subray.view(B, K, 3).mean(1)          # (B,3)
                pred_no_slab = no_slab_pred_subray.view(B, K, 3).mean(1)    # (B,3)
                tau0      = tau0.view(B, K).mean(1)                         # (B,)
                gt_subray = gt.unsqueeze(1).expand(-1, K, -1).reshape(-1, 3)

                patch = int(math.isqrt(BATCH))
                if patch*patch == BATCH:
                    pr    = pred.view(patch, patch, 3).permute(2,0,1).unsqueeze(0)
                    gtimg = gt.view(patch, patch, 3).permute(2,0,1).unsqueeze(0)
                    low_p, low_g = AVG2(pr), AVG2(gtimg)
                    gpx = F.conv2d(low_p, sobel_x, padding=1, groups=3)
                    gpy = F.conv2d(low_p, sobel_y, padding=1, groups=3)
                    ggx = F.conv2d(low_g, sobel_x, padding=1, groups=3)
                    ggy = F.conv2d(low_g, sobel_y, padding=1, groups=3)
                    edge_l = (gpx-ggx).abs().mean() + (gpy-ggy).abs().mean()
                else:
                    edge_l = torch.tensor(0., device=DEVICE)

                # HashGrid encoders aren't dense H×W tensors, so we skip TV.
                tv_term = torch.tensor(0.0, device=DEVICE)

                per_ray_err = (pred - gt).abs().mean(dim=-1)
                per_ray_base_err = (pred_base - gt).abs().mean(dim=-1)
                per_ray_no_slab_err = (pred_no_slab - gt).abs().mean(dim=-1)
                per_ray_l1 = per_ray_err * w_var[yb, xb]
                per_ray_base_l1 = per_ray_base_err * w_var[yb, xb]
                per_ray_no_slab_l1 = per_ray_no_slab_err * w_var[yb, xb]
                l1 = per_ray_l1.mean()
                base_l1 = per_ray_base_l1.mean()
                no_slab_l1 = per_ray_no_slab_l1.mean()
                residual_gain_weight_curr = RESIDUAL_GAIN_WEIGHT * float(schedule_state["residual_gain_mult"])
                gain_margin = (per_ray_base_l1.detach() - per_ray_l1).clamp_min(0.0)
                if phase2_active and PHASE2_HARD_FOCUS_QUANTILE > 0.0:
                    focus_q = min(max(PHASE2_HARD_FOCUS_QUANTILE, 0.0), 0.999)
                    hard_cut = torch.quantile(per_ray_base_l1.detach(), focus_q)
                    focus_scale = per_ray_base_l1.detach().std(unbiased=False).clamp_min(1e-4)
                    hard_focus = torch.sigmoid((per_ray_base_l1.detach() - hard_cut) / focus_scale)
                    hard_focus = PHASE2_HARD_FOCUS_FLOOR + (1.0 - PHASE2_HARD_FOCUS_FLOOR) * hard_focus
                else:
                    hard_focus = torch.ones_like(gain_margin)
                residual_gain_reward = (
                    (gain_margin * hard_focus).mean()
                    * residual_gain_weight_curr
                )
                slab_gain_margin = (per_ray_no_slab_l1.detach() - per_ray_l1).clamp_min(0.0)
                residual_focus_mean = float(hard_focus.mean().item())
                residual_focus_hi_frac = float((hard_focus > 0.75).float().mean().item())
                hard_focus_subray = hard_focus.unsqueeze(1).expand(-1, K).reshape(-1)
                route_target_subray = torch.zeros_like(hard_focus_subray.float())
                route_disagreement_mean = 0.0
                hard_spread_floor = torch.zeros_like(hard_focus)
                spread_floor_scale_curr = float(schedule_state["spread_floor_scale"])
                edge_spread_weight_curr = EDGE_SPREAD_WEIGHT * float(schedule_state["edge_spread_mult"])
                pair_repulsion_weight_curr = PAIR_REPULSION_WEIGHT * float(schedule_state["pair_repulsion_mult"])
                schedule_relax_mix = float(schedule_state["relax_mix"])
                if phase2_active:
                    hard_spread_floor = PHASE2_HARD_SPREAD_FLOOR_PX * spread_floor_scale_curr * hard_focus
                hard_spread_floor_subray = hard_spread_floor.unsqueeze(1).expand(-1, K).reshape(-1)
                spread_target_curr = torch.maximum(edge_target.float(), hard_spread_floor_subray)
                hard_spread_floor_mean = float(hard_spread_floor.mean().item())

                # optional τ supervision
                if TAU_WEIGHT > 0 and depth_path is not None:
                    d_ = np.load(depth_path)
                    d_t = torch.from_numpy(d_).to(DEVICE).float()
                    d_min, d_max = d_t.min(), d_t.max()
                    d_split = 0.5*(d_min + d_max)
                    sigma   = (d_max - d_min) * TAU_SIGMA_FRACTION
                    tau_gt  = torch.sigmoid(-(d_t - d_split)/sigma)
                    with autocast(enabled=False):
                        tau_loss = F.binary_cross_entropy(
                            tau0.float(), 
                            tau_gt[yb, xb].float()
                        )
                else:
                    tau_loss = torch.tensor(0., device=DEVICE)

                gates = rear_aux["gates"]
                offset_px = rear_aux["offset_px"]
                head_rgb = rear_aux["head_rgb"]
                route_disagreement = slab_head_disagreement(head_rgb)
                route_target_subray = _normalize_batch_signal(route_disagreement)
                route_target_mean = float(route_target_subray.mean().item())
                route_disagreement_mean = float(route_disagreement.mean().item())
                head_div_mean_curr = float(torch.stack([
                    (head_rgb[:, 0] - head_rgb[:, 1]).abs().mean(dim=-1),
                    (head_rgb[:, 0] - head_rgb[:, 2]).abs().mean(dim=-1),
                    (head_rgb[:, 1] - head_rgb[:, 2]).abs().mean(dim=-1),
                ], dim=1).mean().item())
                route_strength_raw = rear_aux["route_strength_raw"].float()
                gate_usage_entropy = _entropy_from_probs(gates.float().mean(dim=0).unsqueeze(0), dim=-1).squeeze(0)
                gate_entropy = _entropy_from_probs(gates.float(), dim=-1)
                gate_usage_loss = (math.log(3.0) - gate_usage_entropy) * GATE_USAGE_WEIGHT
                head_decor_loss = _head_residual_decorrelation(head_rgb, gt_subray) * HEAD_DECORR_WEIGHT
                route_strength_loss = F.smooth_l1_loss(
                    route_strength_raw,
                    route_target_subray.detach(),
                    reduction="none",
                ).mean() * (ROUTE_STRENGTH_ALIGN_WEIGHT * adaptive_router_strength_curr)
                target_gate_entropy = (
                    1.0 - route_target_subray.detach() * (1.0 - HARD_GATE_ENTROPY_FRACTION)
                ) * math.log(3.0)
                route_entropy_loss = (
                    (gate_entropy - target_gate_entropy).abs().mean()
                    * (ADAPTIVE_GATE_ENTROPY_WEIGHT * adaptive_router_strength_curr)
                )
                pair_dists = _pairwise_offset_pair_dists_px(offset_px).float()
                pair_spread_mean_curr = float(pair_dists.mean().item())
                slab_schedule = _slab_schedule_state(global_step + 1, head_div_mean_curr, pair_spread_mean_curr)
                slab_schedule_scale = float(slab_schedule["scale"])
                spread_focus = hard_focus_subray.float()
                edge_spread_resid = (pair_dists.mean(dim=1) - spread_target_curr).abs()
                edge_spread_loss = (edge_spread_resid * spread_focus).mean() * edge_spread_weight_curr
                pair_repulsion_resid = F.relu(spread_target_curr.unsqueeze(-1) - pair_dists).mean(dim=1)
                pair_repulsion_loss = (pair_repulsion_resid * spread_focus).mean() * pair_repulsion_weight_curr
                offset_l2 = (offset_raw**2).mean() * OFFSET_L2_WEIGHT
                slab_gain_weight_curr = (
                    SLAB_GAIN_WEIGHT
                    * float(max(1.0, schedule_state["residual_gain_mult"]))
                    * slab_schedule_scale
                )
                slab_gain_reward = (
                    (slab_gain_margin * hard_focus).mean()
                    * slab_gain_weight_curr
                )
                slab_strength_loss = slab_aux["strength_mean"] * SLAB_STRENGTH_WEIGHT
                slab_opacity_loss = slab_aux["alpha"].mean() * SLAB_OPACITY_WEIGHT
                loss = (
                    l1
                    + EDGE_WEIGHT * edge_l
                    + TV_WEIGHT * tv_term
                    + TAU_WEIGHT * tau_loss
                    + offset_l2
                    + gate_usage_loss
                    + head_decor_loss
                    + route_strength_loss
                    + route_entropy_loss
                    + edge_spread_loss
                    + pair_repulsion_loss
                    + slab_strength_loss
                    + slab_opacity_loss
                    - residual_gain_reward
                    - slab_gain_reward
                )

            if torch.isnan(loss).any():
                warnings.warn("NaN encountered – aborting.")
                return front, rear, slab, z_rear_offset, camera_embed

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            if base_head_trainable and abs(base_grad_scale_curr - 1.0) > 1e-6:
                _scale_module_grads(rear.component_heads[0], base_grad_scale_curr)
            if abs(slab_schedule_scale - 1.0) > 1e-6:
                _scale_module_grads(slab, slab_schedule_scale)
            torch.nn.utils.clip_grad_norm_(nonslab_params + slab_params, CLIP_NORM)

            if do_diag:
                payload = _three_ray_diag_payload(
                    step=global_step + 1,
                    loss=loss,
                    l1=l1,
                    base_l1=base_l1,
                    no_slab_l1=no_slab_l1,
                    residual_gain_reward=residual_gain_reward,
                    residual_gain_weight_curr=residual_gain_weight_curr,
                    slab_gain_reward=slab_gain_reward,
                    slab_gain_weight_curr=slab_gain_weight_curr,
                    slab_schedule_scale=slab_schedule_scale,
                    slab_step_ready=slab_schedule["step_ready"],
                    slab_head_ready=slab_schedule["head_ready"],
                    slab_spread_ready=slab_schedule["spread_ready"],
                    slab_structure_ready=slab_schedule["structure_ready"],
                    residual_focus_mean=residual_focus_mean,
                    residual_focus_hi_frac=residual_focus_hi_frac,
                    route_target_mean=route_target_mean,
                    route_disagreement_mean=route_disagreement_mean,
                    base_grad_scale_curr=base_grad_scale_curr,
                    gate_temperature_curr=gate_temperature_curr,
                    hard_gate_temperature_curr=hard_gate_temperature_curr,
                    schedule_relax_mix=schedule_relax_mix,
                    spread_floor_scale_curr=spread_floor_scale_curr,
                    hard_spread_floor_mean=hard_spread_floor_mean,
                    edge_spread_weight_curr=edge_spread_weight_curr,
                    pair_repulsion_weight_curr=pair_repulsion_weight_curr,
                    base_frozen=not base_head_trainable,
                    edge_l=edge_l,
                    tau_loss=tau_loss,
                    offset_l2=offset_l2,
                    slab_strength_loss=slab_strength_loss,
                    slab_opacity_loss=slab_opacity_loss,
                    gate_usage_loss=gate_usage_loss,
                    head_decor_loss=head_decor_loss,
                    route_strength_loss=route_strength_loss,
                    route_entropy_loss=route_entropy_loss,
                    edge_spread_loss=edge_spread_loss,
                    pair_repulsion_loss=pair_repulsion_loss,
                    front=front,
                    rear=rear,
                    slab=slab,
                    camera_embed=camera_embed,
                    alpha_front=alpha0_subray,
                    tau0=tau0_subray,
                    rear_alpha=rear_alpha,
                    valid_f=valid_f,
                    valid_r=valid_r,
                    offset_raw=offset_raw,
                    rear_aux=rear_aux,
                    slab_aux=slab_aux,
                    spread_target_px=spread_target_curr,
                )
                _emit_three_ray_diag(payload, DIAG_JSONL)

            scaler.step(opt)
            scaler.update()

            global_step += 1

            if global_step < WARMUP_STEPS:
                lr = LR_BASE * (global_step+1) / WARMUP_STEPS
                if len(opt.param_groups) >= 1:
                    opt.param_groups[0]["lr"] = lr * NON_SLAB_LR_SCALE
                if len(opt.param_groups) >= 2:
                    opt.param_groups[1]["lr"] = lr * SLAB_LR_SCALE * slab_schedule_scale
            elif ITERS_TOTAL > WARMUP_STEPS:
                sched.step()
                if len(opt.param_groups) >= 2:
                    opt.param_groups[1]["lr"] = opt.param_groups[1]["lr"] * slab_schedule_scale

            with torch.no_grad():
                per_ray_err = (pred.detach() - gt).abs().mean(dim=-1)
                errmap.mul_(EMA)
                errmap.view(-1).index_add_(0, flat,
                                           per_ray_err * (1.0 - EMA))

            # track depth loss
            if depth_path is not None:
                running_tau = getattr(train, "_tau_hist", [])
                running_tau.append(tau_loss.item())
                if len(running_tau) > 100:
                    running_tau.pop(0)
                train._tau_hist = running_tau

            loss_hist.append(loss.item())
            if global_step % LOG_EVERY == 0:
                sps = global_step / (time.time() - start_t + 1e-9)
                eta = (ITERS_TOTAL - global_step) / (sps + 1e-9)
                mins, secs = divmod(int(eta), 60)
                peak = (torch.cuda.max_memory_allocated()>>20) if DEVICE=="cuda" else 0
                ema_val = sum(loss_hist)/len(loss_hist)
                vps = val_psnr_ema if val_psnr_ema is not None else 0
                drop_p = 100*sum(dropout_hist)/len(dropout_hist)
                tau_avg = sum(getattr(train, "_tau_hist", [0])) / max(len(getattr(train, "_tau_hist", [])),1)
                print(f"{global_step:>8d}/{ITERS_TOTAL}  "
                      f"L {loss.item():.4e}  EMA {ema_val:.4e}  "
                      f"valPSNR {vps:5.2f}  drop% {drop_p:4.1f}  "
                      f"{mins:02d}:{secs:02d} ETA  peak {peak:.0f} MB")
                print(f"        τL {tau_avg:.3e}")

            if global_step >= next_ck or global_step == target_step:
                torch.save({
                    "front": front.state_dict(),
                    "rear":  rear.state_dict(),
                    "slab": slab.state_dict(),
                    "camera_embed": camera_embed.state_dict(),
                    "z_offset": z_rear_offset.detach().cpu().item(),
                    "global_step": global_step
                }, CHECKPOINT_PATH)
                next_ck += CKPT_EVERY

            if global_step >= target_step:
                break

        torch.cuda.empty_cache()

    return front, rear, slab, z_rear_offset, camera_embed


def main():
    global CHECKPOINT_PATH, ITERS_TOTAL, BATCH, LOG_EVERY, CKPT_EVERY, TRAIN_CHUNK, EXTRA_STEPS
    global TAU_WEIGHT, DIAG_EVERY, DIAG_JSONL, OFFSET_L2_WEIGHT, GATE_USAGE_WEIGHT
    global HEAD_DECORR_WEIGHT, EDGE_SPREAD_WEIGHT, EDGE_SPREAD_TARGET_PX, PAIR_REPULSION_WEIGHT
    global MAX_RELATIVE_OFFSET_PX, BASE_MIN_PAIR_SEPARATION_PX, MAX_FAN_SHIFT_PX, RESIDUAL_GAIN_WEIGHT
    global SLAB_SPLATS, SLAB_STRENGTH_WEIGHT, SLAB_GAIN_WEIGHT, SLAB_OPACITY_WEIGHT
    global NON_SLAB_LR_SCALE, SLAB_LR_SCALE
    global SLAB_RAMP_WARMUP_STEPS, SLAB_RAMP_STEPS, SLAB_RAMP_START_SCALE
    global SLAB_RAMP_HEAD_DIV_THRESHOLD, SLAB_RAMP_SPREAD_THRESHOLD
    global PHASE1_STEPS, PHASE2_RESIDUAL_GAIN_MULT, PHASE2_BASE_GRAD_SCALE
    global PHASE2_HARD_FOCUS_QUANTILE, PHASE2_HARD_FOCUS_FLOOR
    global PHASE2_HARD_SPREAD_FLOOR_PX, PHASE2_EDGE_SPREAD_MULT, PHASE2_PAIR_REPULSION_MULT
    global PHASE3_START_RATIO, PHASE3_END_SPREAD_FLOOR_SCALE
    global PHASE3_END_EDGE_SPREAD_MULT, PHASE3_END_PAIR_REPULSION_MULT, PHASE3_END_GATE_TEMPERATURE
    global HARD_GATE_TEMPERATURE, ROUTE_STRENGTH_ALIGN_WEIGHT
    global ADAPTIVE_GATE_ENTROPY_WEIGHT, HARD_GATE_ENTROPY_FRACTION
    global FREEZE_BASE_AFTER_PHASE1
    parser = argparse.ArgumentParser()
    parser.add_argument("--renders-dir", type=Path, default=RENDERS_DIR,
                        help="Directory containing views.jsonl plus RGB/depth frames.")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH,
                        help="Checkpoint path to resume from and save to.")
    parser.add_argument("--steps", type=int, default=ITERS_TOTAL,
                        help="Absolute step ceiling for long training runs.")
    parser.add_argument("--extra-steps", type=int, default=0,
                        help="Run this many more steps from the loaded checkpoint.")
    parser.add_argument("--batch", type=int, default=BATCH,
                        help="Training batch size. Lower values are lighter but noisier.")
    parser.add_argument("--chunk", type=int, default=TRAIN_CHUNK,
                        help="How many camera views to stage per outer loop.")
    parser.add_argument("--log-every", type=int, default=LOG_EVERY,
                        help="Log cadence in optimizer steps.")
    parser.add_argument("--ckpt-every", type=int, default=CKPT_EVERY,
                        help="Checkpoint cadence in optimizer steps.")
    parser.add_argument("--tau-weight", type=float, default=TAU_WEIGHT,
                        help="Weight on the depth-derived front/rear supervision. Set to 0 to disable MiDaS-style tau supervision.")
    parser.add_argument("--diag-every", type=int, default=DIAG_EVERY,
                        help="Emit 3-ray diagnostics every N optimizer steps. 0 disables diagnostics.")
    parser.add_argument("--diag-jsonl", type=Path, default=None,
                        help="Optional JSONL path to append structured 3-ray diagnostics.")
    parser.add_argument("--offset-l2-weight", type=float, default=OFFSET_L2_WEIGHT,
                        help="L2 penalty on normalized rear-plane sample offsets.")
    parser.add_argument("--gate-usage-weight", type=float, default=GATE_USAGE_WEIGHT,
                        help="Batch-level entropy regularizer that keeps all three rear transport components active.")
    parser.add_argument("--head-decorr-weight", type=float, default=HEAD_DECORR_WEIGHT,
                        help="Residual decorrelation weight between the three rear transport heads.")
    parser.add_argument("--edge-spread-weight", type=float, default=EDGE_SPREAD_WEIGHT,
                        help="Weight for edge-biased spread of the three rear plane samples.")
    parser.add_argument("--edge-spread-target-px", type=float, default=EDGE_SPREAD_TARGET_PX,
                        help="Target maximum component spread in rear-plane pixels for strong edges.")
    parser.add_argument("--pair-repulsion-weight", type=float, default=PAIR_REPULSION_WEIGHT,
                        help="Additional hinge penalty that prevents the three rear plane samples from collapsing together.")
    parser.add_argument("--max-relative-offset-px", type=float, default=MAX_RELATIVE_OFFSET_PX,
                        help="Maximum learned component offset around the shared rear anchor, in plane pixels.")
    parser.add_argument("--base-min-separation-px", type=float, default=BASE_MIN_PAIR_SEPARATION_PX,
                        help="Minimum pairwise separation enforced between the three rear probes before loss terms.")
    parser.add_argument("--max-fan-shift-px", type=float, default=MAX_FAN_SHIFT_PX,
                        help="Maximum shared perpendicular fan shift for the two flank probes.")
    parser.add_argument("--residual-gain-weight", type=float, default=RESIDUAL_GAIN_WEIGHT,
                        help="Reward weight for error the flank residual transport removes beyond the base rear probe.")
    parser.add_argument("--slab-splats", type=int, default=SLAB_SPLATS,
                        help="Number of learned Gaussian attenuators inside the slab between the front and rear sheets.")
    parser.add_argument("--slab-strength-weight", type=float, default=SLAB_STRENGTH_WEIGHT,
                        help="L1-style regularizer on slab Gaussian strengths so the residual volume stays compact.")
    parser.add_argument("--slab-gain-weight", type=float, default=SLAB_GAIN_WEIGHT,
                        help="Reward weight for error removed by the intra-slab attenuation volume beyond the sheet-only render.")
    parser.add_argument("--slab-opacity-weight", type=float, default=SLAB_OPACITY_WEIGHT,
                        help="Penalty on mean slab opacity to keep the volume residual rather than dominant.")
    parser.add_argument("--slab-ramp-warmup-steps", type=int, default=SLAB_RAMP_WARMUP_STEPS,
                        help="Do not let slab reward/LR ramp significantly until at least this many scratch-training steps have passed.")
    parser.add_argument("--slab-ramp-steps", type=int, default=SLAB_RAMP_STEPS,
                        help="How many steps the disagreement-aware slab ramp should take once warmup has passed.")
    parser.add_argument("--slab-ramp-start-scale", type=float, default=SLAB_RAMP_START_SCALE,
                        help="Minimum slab reward/LR scale before the transport structure is ready.")
    parser.add_argument("--slab-ramp-head-div-threshold", type=float, default=SLAB_RAMP_HEAD_DIV_THRESHOLD,
                        help="Rear head-diversity level at which the slab is considered structurally ready.")
    parser.add_argument("--slab-ramp-spread-threshold", type=float, default=SLAB_RAMP_SPREAD_THRESHOLD,
                        help="Rear probe pair-spread level, in plane pixels, at which the slab is considered structurally ready.")
    parser.add_argument("--phase1-steps", type=int, default=PHASE1_STEPS,
                        help="Number of warmup steps before the phase-2 residual emphasis schedule activates. 0 disables the schedule.")
    parser.add_argument("--phase2-residual-gain-mult", type=float, default=PHASE2_RESIDUAL_GAIN_MULT,
                        help="Multiplier on residual-gain reward once phase 2 begins.")
    parser.add_argument("--phase2-base-grad-scale", type=float, default=PHASE2_BASE_GRAD_SCALE,
                        help="Gradient scale applied to the base rear head during phase 2. Values below 1 soften the base path without freezing it.")
    parser.add_argument("--phase2-hard-focus-quantile", type=float, default=PHASE2_HARD_FOCUS_QUANTILE,
                        help="In phase 2, emphasize residual gain on rays whose base-only error is above this quantile. 0 disables hard-ray focusing.")
    parser.add_argument("--phase2-hard-focus-floor", type=float, default=PHASE2_HARD_FOCUS_FLOOR,
                        help="Minimum residual-gain focus weight for easy rays during phase 2.")
    parser.add_argument("--phase2-hard-spread-floor-px", type=float, default=PHASE2_HARD_SPREAD_FLOOR_PX,
                        help="Minimum mean pair spread target, in plane pixels, for hard rays during phase 2.")
    parser.add_argument("--phase2-edge-spread-mult", type=float, default=PHASE2_EDGE_SPREAD_MULT,
                        help="Multiplier on the edge-spread loss during phase 2.")
    parser.add_argument("--phase2-pair-repulsion-mult", type=float, default=PHASE2_PAIR_REPULSION_MULT,
                        help="Multiplier on the pair-repulsion loss during phase 2.")
    parser.add_argument("--phase3-start-ratio", type=float, default=PHASE3_START_RATIO,
                        help="Fraction of phase-2 progress at which consolidation begins and spatial pressure starts relaxing.")
    parser.add_argument("--phase3-end-spread-floor-scale", type=float, default=PHASE3_END_SPREAD_FLOOR_SCALE,
                        help="Final scale on the phase-2 hard spread floor by the end of training.")
    parser.add_argument("--phase3-end-edge-spread-mult", type=float, default=PHASE3_END_EDGE_SPREAD_MULT,
                        help="Final phase-3 multiplier on edge-spread loss by the end of training.")
    parser.add_argument("--phase3-end-pair-repulsion-mult", type=float, default=PHASE3_END_PAIR_REPULSION_MULT,
                        help="Final phase-3 multiplier on pair-repulsion loss by the end of training.")
    parser.add_argument("--phase3-end-gate-temperature", type=float, default=PHASE3_END_GATE_TEMPERATURE,
                        help="Final phase-3 gate softening temperature by the end of training. Values above 1 make routing more uniform.")
    parser.add_argument("--hard-gate-temperature", type=float, default=HARD_GATE_TEMPERATURE,
                        help="Per-ray minimum gate temperature allowed on hard rays once adaptive routing is active.")
    parser.add_argument("--route-strength-align-weight", type=float, default=ROUTE_STRENGTH_ALIGN_WEIGHT,
                        help="How strongly phase-3 routing strength should track the hard-ray target derived from base-only error.")
    parser.add_argument("--adaptive-gate-entropy-weight", type=float, default=ADAPTIVE_GATE_ENTROPY_WEIGHT,
                        help="How strongly phase-3 routing entropy should stay high on easy rays and sharper on hard rays.")
    parser.add_argument("--hard-gate-entropy-fraction", type=float, default=HARD_GATE_ENTROPY_FRACTION,
                        help="Target gate entropy on hard rays as a fraction of log(3). Lower values allow sharper routing.")
    parser.add_argument("--freeze-base-after-phase1", action="store_true",
                        help="Freeze the base rear transport head once phase 2 begins so residual probes must explain the leftover error.")
    parser.add_argument("--non-slab-lr-scale", type=float, default=NON_SLAB_LR_SCALE,
                        help="Multiplier on the base learning rate for the existing sheet, transport, pose, and z-offset parameters.")
    parser.add_argument("--slab-lr-scale", type=float, default=SLAB_LR_SCALE,
                        help="Multiplier on the base learning rate for the intra-slab Gaussian attenuation field.")
    parser.add_argument("--quick", action="store_true",
                        help="Use a lightweight smoke-test preset.")
    parser.add_argument("--sh_file_front", type=str, default=str(PROJECT_ROOT / "sh_billboard_L7.pt"),
                        help="Path to a baked SH .pt file for the front billboard. "
                             "Leave empty to disable SH for front.")
    parser.add_argument("--sh_file_rear", type=str, default="",
                        help="Path to a baked SH .pt file for the rear billboard. "
                             "Leave empty to disable SH for rear.")

    args = parser.parse_args()
    if args.quick:
        args.batch = min(args.batch, 4096)
        args.chunk = min(args.chunk, 48)
        args.log_every = min(args.log_every, 5)
        args.ckpt_every = min(args.ckpt_every, 25)
        if args.extra_steps <= 0:
            args.extra_steps = 25

    CHECKPOINT_PATH = args.checkpoint.expanduser()
    ck_name = CHECKPOINT_PATH.name.strip()
    if ck_name in {"", ".pt"}:
        raise SystemExit(
            f"Invalid checkpoint path: {CHECKPOINT_PATH}. "
            "Use a real file name like /path/to/run_name.pt."
        )
    ITERS_TOTAL = args.steps
    EXTRA_STEPS = max(0, args.extra_steps)
    BATCH = args.batch
    TRAIN_CHUNK = max(1, args.chunk)
    LOG_EVERY = max(1, args.log_every)
    CKPT_EVERY = max(1, args.ckpt_every)
    TAU_WEIGHT = max(0.0, args.tau_weight)
    DIAG_EVERY = max(0, args.diag_every)
    DIAG_JSONL = args.diag_jsonl.expanduser() if args.diag_jsonl else None
    OFFSET_L2_WEIGHT = max(0.0, args.offset_l2_weight)
    GATE_USAGE_WEIGHT = max(0.0, args.gate_usage_weight)
    HEAD_DECORR_WEIGHT = max(0.0, args.head_decorr_weight)
    EDGE_SPREAD_WEIGHT = max(0.0, args.edge_spread_weight)
    EDGE_SPREAD_TARGET_PX = max(0.0, args.edge_spread_target_px)
    PAIR_REPULSION_WEIGHT = max(0.0, args.pair_repulsion_weight)
    MAX_RELATIVE_OFFSET_PX = max(0.0, args.max_relative_offset_px)
    BASE_MIN_PAIR_SEPARATION_PX = max(0.0, args.base_min_separation_px)
    MAX_FAN_SHIFT_PX = max(0.0, args.max_fan_shift_px)
    RESIDUAL_GAIN_WEIGHT = max(0.0, args.residual_gain_weight)
    SLAB_SPLATS = max(0, args.slab_splats)
    SLAB_STRENGTH_WEIGHT = max(0.0, args.slab_strength_weight)
    SLAB_GAIN_WEIGHT = max(0.0, args.slab_gain_weight)
    SLAB_OPACITY_WEIGHT = max(0.0, args.slab_opacity_weight)
    NON_SLAB_LR_SCALE = max(0.0, args.non_slab_lr_scale)
    SLAB_LR_SCALE = max(0.0, args.slab_lr_scale)
    SLAB_RAMP_WARMUP_STEPS = max(0, args.slab_ramp_warmup_steps)
    SLAB_RAMP_STEPS = max(0, args.slab_ramp_steps)
    SLAB_RAMP_START_SCALE = min(max(0.0, args.slab_ramp_start_scale), 1.0)
    SLAB_RAMP_HEAD_DIV_THRESHOLD = max(0.0, args.slab_ramp_head_div_threshold)
    SLAB_RAMP_SPREAD_THRESHOLD = max(0.0, args.slab_ramp_spread_threshold)
    PHASE1_STEPS = max(0, args.phase1_steps)
    PHASE2_RESIDUAL_GAIN_MULT = max(0.0, args.phase2_residual_gain_mult)
    PHASE2_BASE_GRAD_SCALE = max(0.0, args.phase2_base_grad_scale)
    PHASE2_HARD_FOCUS_QUANTILE = min(max(0.0, args.phase2_hard_focus_quantile), 0.999)
    PHASE2_HARD_FOCUS_FLOOR = min(max(0.0, args.phase2_hard_focus_floor), 1.0)
    PHASE2_HARD_SPREAD_FLOOR_PX = max(0.0, args.phase2_hard_spread_floor_px)
    PHASE2_EDGE_SPREAD_MULT = max(0.0, args.phase2_edge_spread_mult)
    PHASE2_PAIR_REPULSION_MULT = max(0.0, args.phase2_pair_repulsion_mult)
    PHASE3_START_RATIO = min(max(0.0, args.phase3_start_ratio), 0.999)
    PHASE3_END_SPREAD_FLOOR_SCALE = max(0.0, args.phase3_end_spread_floor_scale)
    PHASE3_END_EDGE_SPREAD_MULT = max(0.0, args.phase3_end_edge_spread_mult)
    PHASE3_END_PAIR_REPULSION_MULT = max(0.0, args.phase3_end_pair_repulsion_mult)
    PHASE3_END_GATE_TEMPERATURE = max(1.0, args.phase3_end_gate_temperature)
    HARD_GATE_TEMPERATURE = max(1e-4, args.hard_gate_temperature)
    ROUTE_STRENGTH_ALIGN_WEIGHT = max(0.0, args.route_strength_align_weight)
    ADAPTIVE_GATE_ENTROPY_WEIGHT = max(0.0, args.adaptive_gate_entropy_weight)
    HARD_GATE_ENTROPY_FRACTION = min(max(0.0, args.hard_gate_entropy_fraction), 1.0)
    FREEZE_BASE_AFTER_PHASE1 = args.freeze_base_after_phase1

    pygame.init()

    renders_dir = args.renders_dir.expanduser()
    if not renders_dir.is_dir():
        raise SystemExit(f"❌ missing directory {renders_dir}")

    print(f"Using renders dir: {renders_dir}")
    print(
        f"Training config: checkpoint={CHECKPOINT_PATH}, batch={BATCH}, "
        f"chunk={TRAIN_CHUNK}, log_every={LOG_EVERY}, extra_steps={EXTRA_STEPS}, "
        f"tau_weight={TAU_WEIGHT}, diag_every={DIAG_EVERY}, "
        f"phase1_steps={PHASE1_STEPS}, phase2_gain_mult={PHASE2_RESIDUAL_GAIN_MULT:.2f}, "
        f"phase2_base_grad_scale={PHASE2_BASE_GRAD_SCALE:.2f}, "
        f"phase2_focus_q={PHASE2_HARD_FOCUS_QUANTILE:.2f}, "
        f"phase2_focus_floor={PHASE2_HARD_FOCUS_FLOOR:.2f}, "
        f"phase2_spread_floor={PHASE2_HARD_SPREAD_FLOOR_PX:.2f}, "
        f"phase2_edge_mult={PHASE2_EDGE_SPREAD_MULT:.2f}, "
        f"phase2_repel_mult={PHASE2_PAIR_REPULSION_MULT:.2f}, "
        f"phase3_start={PHASE3_START_RATIO:.2f}, "
        f"phase3_end_spread_scale={PHASE3_END_SPREAD_FLOOR_SCALE:.2f}, "
        f"phase3_end_edge_mult={PHASE3_END_EDGE_SPREAD_MULT:.2f}, "
        f"phase3_end_repel_mult={PHASE3_END_PAIR_REPULSION_MULT:.2f}, "
        f"phase3_end_gate_t={PHASE3_END_GATE_TEMPERATURE:.2f}, "
        f"hard_gate_t={HARD_GATE_TEMPERATURE:.2f}, "
        f"route_align_w={ROUTE_STRENGTH_ALIGN_WEIGHT:.3f}, "
        f"route_entropy_w={ADAPTIVE_GATE_ENTROPY_WEIGHT:.3f}, "
        f"hard_gate_H={HARD_GATE_ENTROPY_FRACTION:.2f}, "
        f"slab_splats={SLAB_SPLATS}, slab_strength_w={SLAB_STRENGTH_WEIGHT:.4f}, "
        f"slab_gain_w={SLAB_GAIN_WEIGHT:.3f}, slab_opacity_w={SLAB_OPACITY_WEIGHT:.4f}, "
        f"non_slab_lr_scale={NON_SLAB_LR_SCALE:.2f}, slab_lr_scale={SLAB_LR_SCALE:.2f}, "
        f"slab_ramp={SLAB_RAMP_WARMUP_STEPS}+{SLAB_RAMP_STEPS} "
        f"start={SLAB_RAMP_START_SCALE:.2f} "
        f"div_th={SLAB_RAMP_HEAD_DIV_THRESHOLD:.3f} "
        f"spread_th={SLAB_RAMP_SPREAD_THRESHOLD:.3f}, "
        f"freeze_base={int(FREEZE_BASE_AFTER_PHASE1)}"
    )

    meta, w_var, vis_count = parse_manifest_and_center(renders_dir)

    # optional load of front & rear SH
    sh_embed_front = None
    if args.sh_file_front and Path(args.sh_file_front).is_file():
        sh_embed_front = SHEmbed(Path(args.sh_file_front))

    sh_embed_rear = None
    if args.sh_file_rear and Path(args.sh_file_rear).is_file():
        sh_embed_rear = SHEmbed(Path(args.sh_file_rear))

    front, rear, slab, z_off, cam_embed = train(meta, w_var, vis_count,
                                                sh_embed_front, sh_embed_rear)

    # Optionally, you could add a PyGame viewer here instead of the separate breakout
    print("Training complete.")

if __name__ == "__main__":
    main()
