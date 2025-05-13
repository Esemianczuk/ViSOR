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
try:
    # old API (built-from-source) ─ works if present
    from tinycudann import hashgrid 
except ImportError:
    import types, tinycudann as tcnn, torch.nn as nn
    class _HashGrid(nn.Module):
        def __init__(self,
                     n_levels,
                     n_features_per_level,
                     log2_hashmap_size,
                     base_resolution,
                     per_level_scale):
            super().__init__()
            self.enc = tcnn.Encoding(
                n_input_dims=2,
                encoding_config=dict(
                    otype="HashGrid",
                    n_levels=n_levels,
                    n_features_per_level=n_features_per_level,
                    log2_hashmap_size=log2_hashmap_size,
                    base_resolution=base_resolution,
                    per_level_scale=per_level_scale,
                )
            )
        def forward(self, xy):
            return self.enc(xy)
    # make something that looks like the old sub-module
    hashgrid = types.SimpleNamespace(HashGrid=_HashGrid)
import imageio.v2 as imageio
import numpy as np
import pygame

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TVF

# -- NEW IMPORTS FOR MIXED PRECISION --
from torch.cuda.amp import autocast, GradScaler

# ───────── hyper-params ───────── #
RES                = 512
ITERS_TOTAL        = 200000000
CHECKPOINT_PATH    = PROJECT_ROOT / f"dual_billboard_{RES:04d}_x2_cont_F7_sh_barn.pt"

VIEW_DROPOUT       = 0.15          # 15 %
VAL_EPS            = 1e-4
LOG_EVERY          = 200
BATCH              = 16_384
LR_BASE            = 1e-4
WARMUP_STEPS       = 1000
CLIP_NORM          = 4.0
TV_WEIGHT          = 0 #1e-4
EDGE_WEIGHT        = 0.1
MAX_ANGLE_DEG      = 2
MAX_ANGLE_RAD      = math.radians(MAX_ANGLE_DEG)
CKPT_EVERY         = 2_000
SAFETY_GB          = 20
BUNDLE_K      = 5          # rays per training pixel
BUNDLE_STD_PX = 0.35       # Gaussian jitter radius in *pixel* units

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

def sample_bundle(yc, xc, Rwc, cam_loc, k=BUNDLE_K):
    """Return origins (k,3) and dirs (k,3) for a cone centred at (yc,xc)."""
    offs = torch.randn(k,2, device=DEVICE) * BUNDLE_STD_PX
    ys   = yc + offs[:,0]
    xs   = xc + offs[:,1]
    return pixel_rays(ys, xs, Rwc, cam_loc, 0.0)   # (k,3), (k,3)

# small L2 penalty on prism delta
DELTA_L2_WEIGHT = 1e-3

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
        # shape (3,) in float => expand => (1,3)
        x = torch.stack([phi,theta,rho], dim=-1).float().unsqueeze(0)
        out = self.net(x)
        return out[0]

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
        # Normalise to [0,1] UV coordinates
        uv = torch.stack([safe_px(x).float(), safe_px(y).float()], -1) / RES

        codes_xy = self.codes(uv)           # (B, code_dim)
        return self.pos(torch.cat([codes_xy, cam_feat.expand_as(codes_xy)], -1))



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
    Rear sheet that refracts the ray by up to ±2 deg in 3 directions.
    Also has an alpha for partial coverage.
    If with_sh=True, we incorporate the SH color for each refracted direction's pixel,
    by blending them or feeding them into the MLP.
    """
    def __init__(self, h: int, w: int, with_sh: bool = False):
        super().__init__(h, w)
        self.with_sh = with_sh
        self.sh_embed = None
        self.delta_mlp = nn.Sequential(
            make_mlp(POS_OUT, 256, 9, depth=4), nn.Tanh()
        )
        in_spec = POS_OUT + PE_DIM
        if self.with_sh:
            in_spec += 3  
        self.colour_R = make_mlp(in_spec, HEAD_HID, 3, siren=True)
        self.colour_G = make_mlp(in_spec, HEAD_HID, 3, siren=True)
        self.colour_B = make_mlp(in_spec, HEAD_HID, 3, siren=True)

        self.mix = nn.Conv1d(3, 3, 1, bias=False)
        nn.init.eye_(self.mix.weight.squeeze(-1))

        self.alpha_mlp = nn.Sequential(
            make_mlp(POS_OUT + PE_DIM, 256, 1, depth=2),
            nn.Sigmoid()
        )

    def forward(self,
                y, x,
                ray_origin: torch.Tensor,
                ray_dir:    torch.Tensor,
                cam_feat:   torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return (color, alpha, delta_raw)
          color shape => (B,3,3)
          alpha => (B,)
          delta_raw => (B,9)
        """
        base_feat = self._feat(y, x, cam_feat)
        delta_raw = self.delta_mlp(base_feat)
        delta_raw = torch.clamp(delta_raw, -1.0, 1.0)
        delta = delta_raw.view(-1, 3, 3)

        dirs = F.normalize(ray_dir.unsqueeze(1) + MAX_ANGLE_RAD * delta, dim=-1)  # (B,3,3)

        # We build a 3-ray input to the color heads. 
        # If with_sh => compute SH color for each sub-ray, then pass them individually to each color path.
        # So for each k in [0..2]:
        #   we do PE for dirs[:,k] and optionally cat the SH color.
        # Then cR, cG, cB => shape (B,3).
        # Then pack them into col => (B,3,3).

        # Build the direction+origin PE
        B_ = y.size(0)
        all_pe = []
        all_sh = []
        for k in range(3):
            pe_dir_k = encode_dir(dirs[:,k])
            pe_loc_k = encode_vec(ray_origin)
            pe_k = torch.cat([pe_dir_k, pe_loc_k], -1)  # (B, PE_DIM)
            all_pe.append(pe_k)

            if self.with_sh and self.sh_embed is not None:
                sh_c = self.sh_embed(y, x, dirs[:,k])  # (B,3)
            else:
                sh_c = None
            all_sh.append(sh_c)

        pe_stack = torch.stack(all_pe, dim=1)  # (B,3,PE_DIM)
        # shape => (B,3,3) or None
        if self.with_sh and self.sh_embed is not None:
            sh_stack = torch.stack(all_sh, dim=1)  # (B,3,3)
        else:
            sh_stack = None


        feat_exp = base_feat.unsqueeze(1).expand(-1, 3, -1)
        if self.with_sh and (sh_stack is not None):
            inp = torch.cat([feat_exp, pe_stack, sh_stack], dim=-1)
        else:
            inp = torch.cat([feat_exp, pe_stack], dim=-1)

        cR = torch.sigmoid(self.colour_R(inp[:, 0]))  # sub-ray 0
        cG = torch.sigmoid(self.colour_G(inp[:, 1]))  # sub-ray 1
        cB = torch.sigmoid(self.colour_B(inp[:, 2]))  # sub-ray 2
        col = torch.stack([cR, cG, cB], dim=1)        # (B,3,3)
        col = self.mix(col)                           # (B,3,3)

        # Single alpha from (base_feat + pe of first sub-ray)
        alpha_in = torch.cat([base_feat, pe_stack[:,0]], dim=-1)
        alpha_val = self.alpha_mlp(alpha_in).squeeze(-1)  # (B,)

        return col, alpha_val, delta_raw


# ───────── sobel, pooling, etc. ───────── #
SOBEL_BASE = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32)
AVG2       = nn.AvgPool2d(2,2)
UP_VEC     = torch.tensor([0.,1.,0.], dtype=torch.float32)

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

        # rotation that matches the analytic cam above
        R = rot_yx(phi, theta, device="cpu")      # keep meta on CPU

        # -------------------------------------------------------

        img_path = img_dir / r["file"]
        stem     = img_path.stem
        cand1    = img_path.parent / f"{stem}{depth_suffix}"
        cand2    = img_path.parent / f"{stem}.npy"
        depth_path = cand1 if cand1.is_file() else (cand2 if cand2.is_file() else None)

        meta.append({
            "path":   img_path,
            "R":      R,
            "loc":    cam,
            "phi":    phi,
            "theta":  theta,
            "rho":    rho,
            "depth":  depth_path
        })
        locs.append(cam)

        img = TVF.to_tensor(imageio.imread(img_path))[:3]     # C,H,W
        img = img.permute(1,2,0)                              # H,W,C
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

def train(meta, w_var, vis_count,
          sh_embed_front: Optional[SHEmbed],
          sh_embed_rear:  Optional[SHEmbed]):

    front = OcclusionSheet(RES, RES, with_sh=(sh_embed_front is not None)).to(DEVICE)
    rear  = RefractionSheet(RES, RES, with_sh=(sh_embed_rear  is not None)).to(DEVICE)
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
    if os.path.exists(CHECKPOINT_PATH):
        ck = torch.load(CHECKPOINT_PATH, map_location="cpu")
        front.load_state_dict(ck["front"], strict=False)
        rear.load_state_dict(ck["rear"], strict=False)
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

    shared = {id(p) for p in front.heads.parameters()}
    params = [
        z_rear_offset,
        *front.parameters(),
        *[p for p in rear.parameters() if id(p) not in shared],
        *camera_embed.parameters()
    ]
    # remove the (already-frozen) SH parameters from trainable set
    # (They won't appear in front/rear anyway if we do it as is, because they arn't registered as submodules.)
    opt   = torch.optim.AdamW(params, lr=LR_BASE)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, ITERS_TOTAL - WARMUP_STEPS)

    sobel_x = SOBEL_BASE.to(DEVICE).view(1,1,3,3).repeat(3,1,1,1)
    sobel_y = sobel_x.transpose(2,3)

    free, _ = torch.cuda.mem_get_info() if DEVICE == "cuda" else (0,0)
    usable  = free - SAFETY_GB * (1<<30) if DEVICE=="cuda" else 0
    CHUNK   = 610
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

    while global_step < ITERS_TOTAL:
        idxs = torch.randperm(len(meta), generator=rng_cpu)[:CHUNK]
        imgs, Rcs, Lcs, phis, thetas, rhos, depths = [], [], [], [], [], [], []
        for i in idxs:
            rec = meta[i]
            imgs.append(TVF.to_tensor(imageio.imread(rec["path"]))[:3].to(DEVICE))
            Rcs.append(rec["R"].to(DEVICE))
            Lcs.append(rec["loc"].to(DEVICE))
            phis.append(rec["phi"])
            thetas.append(rec["theta"])
            rhos.append(rec["rho"])
            depths.append(rec["depth"])

        imgs   = torch.stack(imgs)
        Rcs    = torch.stack(Rcs)
        Lcs    = torch.stack(Lcs)
        phis   = torch.tensor(phis,   device=DEVICE)
        thetas = torch.tensor(thetas, device=DEVICE)
        rhos   = torch.tensor(rhos,   device=DEVICE)

        steps_here = min(max(1, (len(idxs)*RES*RES)//BATCH),
                          ITERS_TOTAL - global_step)

        for _ in range(steps_here):
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
                    ys, xs = torch.arange(0, RES, 8, device=DEVICE), torch.arange(0, RES, 8, device=DEVICE)
                    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
                    yyf, xxf = yy.flatten(), xx.flatten()

                    # --- fast 1-ray eval (keep lightweight) -------------------------
                    _, d_f_val = pixel_rays(yyf, xxf, Rwc, loc, 0.0)
                    o_r, d_r     = pixel_rays(yyf, xxf, Rwc, loc,
                                              z_rear_offset)

                    pe_f = torch.cat([encode_dir(d_f_val),
                                      encode_vec(loc.unsqueeze(0).expand_as(d_f_val))], -1)

                    sh_col_f = None
                    if sh_embed_front is not None:
                        sh_col_f = sh_embed_front(yyf, xxf, d_f_val)

                    rgb0, tau0 = front(yyf, xxf, pe_f,
                                       cam_feat, sh_col=sh_col_f)
                    rgb0 = rgb0.clamp(0,1)
                    tau0 = tau0.clamp(0,1)

                    rear_out, rear_alpha, _ = rear(yyf, xxf, o_r, d_r, cam_feat)
                    rgb1_diag = rear_out[:, [0,1,2], [0,1,2]].clamp(0,1)
                    rear_alpha = rear_alpha.clamp(0,1)

                    pred = rgb0 + (1 - tau0).unsqueeze(-1)*(rear_alpha.unsqueeze(-1)*rgb1_diag)
                    psnr_val = _psnr(pred, img[:, yyf, xxf].t())
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

            # ---------- update visit counter ---------------------------------------
            flat = yb * RES + xb
            vis_count.view(-1).index_add_(0, flat,
                                          torch.ones_like(flat,
                                                          dtype=vis_count.dtype))

            # ---------- bundle rays -------------------------------------------------
            offs = torch.randn(B, K, 2, device=DEVICE) * BUNDLE_STD_PX
            ys   = yb.unsqueeze(1).float() + offs[...,0]     # (B,K)
            xs   = xb.unsqueeze(1).float() + offs[...,1]     # (B,K)

            # flat (B*K,) → origins / dirs for front & rear
            orig_f, dir_f = pixel_rays(ys.reshape(-1), xs.reshape(-1), Rwc, loc, 0.0)
            orig_r, dir_r = pixel_rays(ys.reshape(-1), xs.reshape(-1), Rwc, loc, z_rear_offset)

            # PE for every sub-ray
            pe_f = torch.cat([encode_dir(dir_f),
                              encode_vec(loc.unsqueeze(0).expand_as(dir_f))], -1)

            # ground-truth still one per centre pixel
            gt = img[:, yb, xb].t()

            # optional SH colour per sub-ray
            sh_col_f = None
            if sh_embed_front is not None:
                sh_col_f = sh_embed_front(  safe_px(ys.reshape(-1)),
                                            safe_px(xs.reshape(-1)),
                                            dir_f)

            opt.zero_grad()
            with autocast():
                rgb0, tau0 = front(ys.reshape(-1).long(),
                                   xs.reshape(-1).long(),
                                   pe_f,
                                   cam_feat,
                                   sh_col=sh_col_f)
                rgb0 = rgb0.clamp(0,1)
                tau0 = tau0.clamp(0,1)

                # Rear
                # If we also have a rear SH, the `rear` forward call 
                # automatically does sh_embed_rear(...) internally. 
                # We already stored that in `rear.sh_embed`. 
                # So no separate param needed in the call if we want each sub‐ray's color.
                rear_out, rear_alpha, delta_raw = rear(ys.reshape(-1).long(),
                                                        xs.reshape(-1).long(),
                                                        orig_r, dir_r, cam_feat)

                # ---------- bundle mean --------------------------------------------
                rgb0      = rgb0.view(B, K, 3).mean(1)                     # (B,3)
                tau0      = tau0.view(B, K).mean(1)                        # (B,)
                rgb1_diag = rear_out[:, [0,1,2], [0,1,2]] \
                            .view(B, K, 3).mean(1).clamp(0,1)              # (B,3)
                rear_alpha= rear_alpha.view(B, K).mean(1).clamp(0,1)       # (B,)

                pred = rgb0 + (1 - tau0).unsqueeze(-1) * \
                             (rear_alpha.unsqueeze(-1) * rgb1_diag)

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

                per_ray_l1 = (pred - gt).abs().mean(dim=-1) * w_var[yb, xb]
                l1 = per_ray_l1.mean()

                # optional τ supervision
                if depth_path is not None:
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

                delta_l2 = (delta_raw**2).mean() * DELTA_L2_WEIGHT
                loss = l1 + EDGE_WEIGHT*edge_l + TV_WEIGHT*tv_term + TAU_WEIGHT*tau_loss + delta_l2

            if torch.isnan(loss).any():
                warnings.warn("NaN encountered – aborting.")
                return front, rear, z_rear_offset, camera_embed

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, CLIP_NORM)
            scaler.step(opt)
            scaler.update()

            global_step += 1

            if global_step < WARMUP_STEPS:
                lr = LR_BASE * (global_step+1) / WARMUP_STEPS
                for g in opt.param_groups:
                    g["lr"] = lr
            else:
                sched.step()

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

            if global_step >= next_ck or global_step == ITERS_TOTAL:
                torch.save({
                    "front": front.state_dict(),
                    "rear":  rear.state_dict(),
                    "camera_embed": camera_embed.state_dict(),
                    "z_offset": z_rear_offset.detach().cpu().item(),
                    "global_step": global_step
                }, CHECKPOINT_PATH)
                next_ck += CKPT_EVERY

            if global_step >= ITERS_TOTAL:
                break

        torch.cuda.empty_cache()

    return front, rear, z_rear_offset, camera_embed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sh_file_front", type=str, default= str(PROJECT_ROOT / "sh_billboard_L7.pt"),
                        help="Path to a baked SH .pt file for the front billboard. "
                             "Leave empty to disable SH for front.")
    parser.add_argument("--sh_file_rear", type=str, default="",
                        help="Path to a baked SH .pt file for the rear billboard. "
                             "Leave empty to disable SH for rear.")

    args = parser.parse_args()

    pygame.init()

    renders_dir = RENDERS_DIR
    if not renders_dir.is_dir():
        raise SystemExit(f"❌ missing directory {renders_dir}")

    meta, w_var, vis_count = parse_manifest_and_center(renders_dir)

    # optional load of front & rear SH
    sh_embed_front = None
    if args.sh_file_front and Path(args.sh_file_front).is_file():
        sh_embed_front = SHEmbed(Path(args.sh_file_front))

    sh_embed_rear = None
    if args.sh_file_rear and Path(args.sh_file_rear).is_file():
        sh_embed_rear = SHEmbed(Path(args.sh_file_rear))

    front, rear, z_off, cam_embed = train(meta, w_var, vis_count,
                                          sh_embed_front, sh_embed_rear)

    # Optionally, you could add a PyGame viewer here instead of the separate breakout
    print("Training complete.")

if __name__ == "__main__":
    main()
