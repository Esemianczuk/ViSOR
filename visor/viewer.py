#!/usr/bin/env python3
"""


Controls:
  - W / S:  move forward/back
  - A / D:  move left/right
  - Q:      move down
  - E:      move up
  - R:      reset to the initial camera position
  - close window: quit

Debug pane controls (right half):
  - LMB drag in debug half: orbit the debug camera
  - Mouse wheel in debug half: zoom in/out the debug camera
  - Left-click a camera point (blue dot) to "jump" to that camera in the main view

Author: Eric Semianczuk
"""

import argparse
import json, math, os, warnings

import numpy as np
from pathlib import Path
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from math import factorial, pi
from visor import PROJECT_ROOT, RENDERS_DIR
from visor.gaussian_slab import GaussianSlab, slab_ray_context
from visor.hashgrid import hashgrid
from visor.plane_geometry import composite_two_planes, look_at_rotation, plane_frame, project_rays_to_plane

# ────────── USER KNOBS ──────────
RES            = 512
CKPT_PATH      = PROJECT_ROOT / "dual_billboard_0512_x2_cont_F7_sh.pt"
SH_FILE_FRONT  = PROJECT_ROOT / "sh_billboard_L7.pt"
SH_FILE_REAR   = None
K_NEIGH        = 15
PE_BANDS       = 8
FOV_DEG        = 60.0
MAX_SAMPLE_OFFSET_PX = 8.0
MAX_RELATIVE_OFFSET_PX = 2.0
BASE_MIN_PAIR_SEPARATION_PX = 0.12
MAX_FAN_SHIFT_PX = 1.0
RESIDUAL_RGB_SCALE = 0.35
HARD_GATE_TEMPERATURE = 0.75

ORIENT_SMOOTH  = 1.0
FLY_SPEED      = 1.0
SCALE          = 1
SHOW_3D_DEBUG  = True
SHOW_DEBUG_LOGS= False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# ────────── UTILS ──────────
def safe_px(t: torch.Tensor, res: int = RES) -> torch.LongTensor:
    """Round, clamp and cast to LONG for safe indexing."""
    return t.round().clamp_(0, res - 1).long()


def enforce_min_pair_separation(relative_px: torch.Tensor, min_sep_px, iters: int = 2) -> torch.Tensor:
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

def load_views(renders_dir: Path, fname: str = "views.jsonl") -> tuple[np.ndarray, ...]:
    json_path = renders_dir / fname
    if not json_path.is_file():
        raise SystemExit(f"❌ missing {json_path}")
    pos, phi, theta, rho = [], [], [], []
    data = json_path.read_text().splitlines()
    for ln in data:
        try:
            r = json.loads(ln)
            phi.append(r["phi"])
            theta.append(r["theta"])
            rho.append(r["rho"])
            x = r["rho"] * math.cos(r["theta"]) * math.cos(r["phi"])
            y = r["rho"] * math.sin(r["theta"])
            z = r["rho"] * math.cos(r["theta"]) * math.sin(r["phi"])
            pos.append((x, y, z))
        except:
            pass
    pos = np.asarray(pos, dtype=np.float32)
    return pos, np.asarray(phi), np.asarray(theta), np.asarray(rho)

POS_ALL = np.empty((0, 3), dtype=np.float32)
PHI_ALL = np.empty((0,), dtype=np.float32)
THETA_ALL = np.empty((0,), dtype=np.float32)
RHO_ALL = np.empty((0,), dtype=np.float32)
N_CAM = 0

def nearest_k(p, k=K_NEIGH):
    if POS_ALL.shape[0] == 0:
        raise RuntimeError("No camera views have been loaded.")
    k = max(1, min(k, POS_ALL.shape[0]))
    d2 = np.sum((POS_ALL - p)**2, 1)
    if k == POS_ALL.shape[0]:
        idx = np.arange(POS_ALL.shape[0])
    else:
        idx = np.argpartition(d2, k - 1)[:k]
    return idx, np.sqrt(d2[idx])


def quantize_render_res(scale: float) -> int:
    scale = float(min(max(scale, 1.0 / RES), 1.0))
    return max(32, min(RES, int(round(RES * scale))))


def build_render_grid(out_res: int, device=DEVICE, dtype=torch.float16):
    coords = torch.linspace(0, RES - 1, out_res, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    return yy.flatten(), xx.flatten()

def bound_cluster(idx):
    sub = POS_ALL[idx]
    return sub.min(0), sub.max(0)

def yaw_pitch_from_phi_theta(phi, theta):
    return float(phi), float(theta)

def slerp_angle(old, new, fac):
    diff = (new - old + math.pi) % (2 * math.pi) - math.pi
    return old + diff * fac


# ────────── Modules ──────────

class SinLayer(nn.Linear):
    """Sine activation with no forced float32 inside."""
    def forward(self, x):
        # forward pass in x's dtype
        w = self.weight.to(dtype=x.dtype)
        b = self.bias.to(dtype=x.dtype)
        x_pre = F.linear(x, w, b)
        # clamp a bit
        x_pre = torch.clamp(x_pre, -25.133, 25.133)
        return torch.sin(x_pre)

def make_siren(in_dim, out_dim):
    m = SinLayer(in_dim, out_dim)
    nn.init.uniform_(m.weight, -1/in_dim, 1/in_dim)
    nn.init.zeros_(m.bias)
    return m

def make_mlp(i, h, o, depth=3, siren=False):
    """
    Build an MLP. By default uses standard Linear + ReLU, or uses SIREN for first layer.
    """
    first = make_siren if siren else nn.Linear
    layers = [first(i, h), nn.ReLU(True)]
    for _ in range(depth - 1):
        layers += [nn.Linear(h, h), nn.ReLU(True)]
    layers += [nn.Linear(h, o)]
    return nn.Sequential(*layers)


# ───── camera embed (kept in float32) ─────
CAM_EMBED_DIM = 32
class CameraEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(True),
            nn.Linear(64, 64), nn.ReLU(True),
            nn.Linear(64, CAM_EMBED_DIM)
        )

    def forward(self, φ, θ, ρ):
        inp = torch.stack([φ.float(), θ.float(), ρ.float()], -1)
        squeeze = inp.dim() == 1
        if squeeze:
            inp = inp.unsqueeze(0)
        out = self.net(inp)
        return out[0] if squeeze else out


# ───── positional encodings ─────
PE_BANDS = 8
def encode_dir(d):
    """Encode direction vectors (B,3) -> (B, 3 + 2*3*PE_BANDS) * 2 => sin/cos expansions."""
    out = [d]
    for i in range(PE_BANDS):
        k = 2**i
        out.append(torch.sin(k*d))
        out.append(torch.cos(k*d))
    return torch.cat(out, -1)

def encode_vec(v):
    """Same idea, for positions or single vectors repeated."""
    out = [v]
    for i in range(PE_BANDS):
        k = 2**i
        out.append(torch.sin(k*v))
        out.append(torch.cos(k*v))
    return torch.cat(out, -1)

PE_DIM = (3 + 6 * PE_BANDS) * 2  # direction + position expansions

# for sheet
CODE_DIM  = 384
POS_HID   = 512
POS_OUT   = 384
HEAD_HID  = 768


#  Spherical Harmonics  
def _legendre_p_batch(lmax, x):
    """Compute associated Legendre polynomials in a batched manner."""
    N = x.shape[0]
    x32 = x.float()
    out = torch.zeros((lmax+1, lmax+1, N), dtype=torch.float32, device=x.device)
    out[0,0] = 1.0
    if lmax == 0:
        return out
    out[1,0] = x32
    y = torch.sqrt(torch.clamp(1 - x32*x32, min=1e-14))
    out[1,1] = y
    for l in range(2, lmax+1):
        out[l,l] = (2*l - 1)*y * out[l-1,l-1]
        for m in range(l-1, -1, -1):
            if l == 1 and m == 0:
                continue
            a = (2*l - 1)*x32*out[l-1,m]
            b = 0.
            if (l - 2) >= 0:
                b = (l + m - 1)*out[l-2,m]
            out[l,m] = (a - b)/(l - m)
    return out

def real_spherical_harmonics(dirs: torch.Tensor, L: int):
    """
    Returns Y_l^m(dirs) up to l = L, shape = [N, (L+1)*(L+1)], in float32.
    """

    N = dirs.size(0)
    device = dirs.device
    out_dim = (L+1)*(L+1)

    eps = 1e-14
    # cast to float32 for the trig below
    d32 = dirs.float()
    r = d32.norm(dim=-1) + eps
    x, y, z = d32[:,0], d32[:,1], d32[:,2]
    theta = torch.acos(torch.clamp(z/r, -1, 1))
    phi   = torch.atan2(y, x)

    ctheta = torch.cos(theta)
    p_all  = _legendre_p_batch(L, ctheta)

    norm_lm = torch.zeros((L+1, L+1), dtype=torch.float32, device=device)
    for l in range(L+1):
        for m in range(l+1):
            num = factorial(l - m)
            den = factorial(l + m)
            norm_lm[l,m] = math.sqrt((2*l + 1)/(4*math.pi)*(num/den))

    m_grid = torch.arange(L+1, device=device, dtype=torch.float32).view(-1,1)
    mp = m_grid * phi.unsqueeze(0)
    cos_mphi = torch.cos(mp)
    sin_mphi = torch.sin(mp)

    out = torch.zeros((N, out_dim), dtype=torch.float32, device=device)

    def sh_index(ll, mm):
        return ll*ll + ll + mm

    for l in range(L+1):
        for m in range(-l, l+1):
            idx = sh_index(l, m)
            if m == 0:
                out[:, idx] = norm_lm[l,0]*p_all[l,0]
            elif m > 0:
                sign = (-1)**m
                out[:, idx] = (math.sqrt(2.0)*sign*norm_lm[l,m]*p_all[l,m]*cos_mphi[m])
            else:
                mp_ = -m
                sign = (-1)**m
                out[:, idx] = (math.sqrt(2.0)*sign*norm_lm[l, mp_]*p_all[l, mp_]*sin_mphi[mp_])
    return out

class SHEmbed(nn.Module):
    """
    Spherical Harmonics embedding, stored in float32. The final color can be cast to half.
    """
    def __init__(self, sh_file: str):
        super().__init__()
        data = torch.load(sh_file, map_location="cpu")
        self.L = data["L"]
        sh_data = data["sh"]  # shape => (RES,RES,(L+1)^2,3)

        self.register_buffer("sh_data", sh_data.float().to(DEVICE))

    def forward(self, y: torch.Tensor, x: torch.Tensor, ray_dir: torch.Tensor):
        """
        Single-direction version for backward compatibility.
        Indexing in [B, (L+1)^2, 3], all float32.
        """
        coefs = self.sh_data[y, x]  # shape [B, (L+1)^2, 3], float32
        B_mat = real_spherical_harmonics(ray_dir, self.L)  # shape [B, (L+1)^2], float32
        c = torch.einsum("bi, bij->bj", B_mat, coefs).clamp(0, 1)  # shape [B,3], float32
        return c

    def forward_all(self, y, x, dirs_all: torch.Tensor):
        """
        For B pixels, 3 directions each => shape(dirs_all) = [B*3, 3].
        We'll expand sh_data [B, (L+1)^2, 3] -> [B*3, (L+1)^2, 3].
        Return shape [B*3, 3], float32.
        """
        B = y.size(0)
        B3 = dirs_all.size(0)
        coefs = self.sh_data[y, x]  # shape [B, (L+1)^2, 3], float32
        coefs = coefs.unsqueeze(1).expand(-1,3,-1,-1).reshape(B3, coefs.shape[1], coefs.shape[2])
        B_mat = real_spherical_harmonics(dirs_all, self.L)  # shape [B*3, (L+1)^2], float32
        c = torch.einsum("bi, bij->bj", B_mat, coefs).clamp(0,1)  # shape [B*3,3], float32
        return c


# ───── Fourier uv ─────
def fourier_uv(x: torch.LongTensor, y: torch.LongTensor) -> torch.Tensor:
    u = (x.float() + 0.5) / RES
    v = (y.float() + 0.5) / RES
    return torch.stack([
        torch.sin(2*math.pi*u),
        torch.cos(2*math.pi*u),
        torch.sin(2*math.pi*v),
        torch.cos(2*math.pi*v),
    ], dim=-1)


# ────────── HEADS ──────────

class Heads(nn.Module):
    """
    The "material heads" that produce rgb & alpha from final features.
    """
    def __init__(self, output_alpha=True, with_sh=False):
        super().__init__()
        self.with_sh = with_sh
        sh_dim = 3 if with_sh else 0

        in_diff  = POS_OUT + sh_dim
        in_spec  = POS_OUT + PE_DIM + sh_dim
        in_rough = POS_OUT + 4 + sh_dim
        in_alpha = POS_OUT + 4 + PE_DIM + sh_dim

        self.fuv_scale = nn.Parameter(torch.tensor(0.1))
        self.diff = make_mlp(in_diff, HEAD_HID, 3, depth=4, siren=True)
        self.spec = make_mlp(in_spec, HEAD_HID, 3, depth=4, siren=True)
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

    def forward(self, feat, pe, fuv, sh_col=None):
        """
        feat, pe, fuv, sh_col are all in half precision (the big MLP logic).
        """
        # cast everything to the same dtype as feat
        dtype_ = feat.dtype

        pe    = pe.to(dtype=dtype_)
        fuv   = fuv.to(dtype=dtype_)
        if sh_col is not None:
            sh_col = sh_col.to(dtype=dtype_)

        gamma = self.fuv_scale.to(dtype=dtype_) * fuv
        if self.with_sh and sh_col is not None:
            feat_d   = torch.cat([feat, sh_col], dim=-1)
            feat_s   = torch.cat([feat, pe, sh_col], dim=-1)
            feat_r   = torch.cat([feat, gamma, sh_col], dim=-1)
            alpha_in = torch.cat([feat, pe, gamma, sh_col], dim=-1)
        else:
            feat_d   = feat
            feat_s   = torch.cat([feat, pe], dim=-1)
            feat_r   = torch.cat([feat, gamma], dim=-1)
            alpha_in = torch.cat([feat, pe, gamma], dim=-1)

        rgb_d = torch.sigmoid(self.diff(feat_d))
        rgb_s = torch.sigmoid(self.spec(feat_s)) * 0.7
        rough = self.rough(feat_r)
        rgb   = rgb_d + (1.0 - rough) * rgb_s

        if self.alpha is None:
            return rgb, None
        a = self.alpha(alpha_in).squeeze(-1)
        return rgb, a


# ────────── SHEETS ──────────

class SheetBase(nn.Module):
    """
    This part runs in half precision.  The hashgrid is half, the final MLP is half.
    """
    def __init__(self, h, w):
        super().__init__()
        # tinycudann HashGrid can run in half
        self.codes = hashgrid.HashGrid(
            n_levels=16,
            n_features_per_level=2,
            log2_hashmap_size=19,
            base_resolution=32,
            per_level_scale=1.3819,
        )
        self.code_dim = 32
        self.pos = nn.Sequential(
            make_siren(self.code_dim + CAM_EMBED_DIM, POS_HID),
            make_siren(POS_HID, POS_HID),
            make_siren(POS_HID, POS_OUT),
        )

    def _feat(self, y, x, cam_feat):
        """
        y, x, cam_feat can come in float16 or float32. We'll unify them to half.
        """
        uv = torch.stack([
            x.float().clamp(0, RES - 1) / max(RES - 1, 1),
            y.float().clamp(0, RES - 1) / max(RES - 1, 1),
        ], dim=-1).to(device=DEVICE)
        uv = uv.half()

        codes_xy = self.codes(uv)  # returns half, shape [B, code_dim]

        cam_feat_ = cam_feat.to(dtype=torch.half)
        if cam_feat_.dim() == 1:
            cam_exp = cam_feat_.unsqueeze(0).expand_as(codes_xy)
        elif cam_feat_.dim() == 2 and cam_feat_.shape[0] == codes_xy.shape[0]:
            cam_exp = cam_feat_
        elif cam_feat_.dim() == 2 and cam_feat_.shape[0] == 1:
            cam_exp = cam_feat_.expand_as(codes_xy)
        else:
            raise ValueError(f"Unsupported cam_feat shape for _feat: {tuple(cam_feat_.shape)}")
        feat_input = torch.cat([codes_xy, cam_exp], dim=-1)  # half
        return self.pos(feat_input)  # returns half


class OcclusionSheet(SheetBase):
    def __init__(self, h, w, with_sh=False):
        super().__init__(h,w)
        self.heads = Heads(True, with_sh=with_sh)
        self.with_sh = with_sh
        self.sh_embed = None
        # dir_cache is used to store directions

    def forward(self, y, x, pe, cam_feat):
        """
        y, x, pe, cam_feat => half
        """
        base_feat = self._feat(y, x, cam_feat)  # half
        pe_  = pe.to(dtype=torch.half)
        # build fuv from x,y
        xL = safe_px(x).to(device=DEVICE)
        yL = safe_px(y).to(device=DEVICE)
        fuv  = fourier_uv(xL, yL).half()  # shape [B,4], half

        sh_col = None
        if self.with_sh and self.sh_embed is not None and hasattr(self, "_dir_cache"):
            yL = safe_px(y)
            xL = safe_px(x)
            dir32 = self._dir_cache.float() 
            raw_col = self.sh_embed(yL, xL, dir32)  # returns float32
            sh_col  = raw_col.half()

        rgb, alpha = self.heads(base_feat, pe_, fuv, sh_col=sh_col)
        return rgb, alpha


class RefractionSheet(SheetBase):
    def __init__(self, h, w, with_sh=False):
        super().__init__(h,w)
        self.with_sh = with_sh
        self.sh_embed = None
        router_dim = POS_OUT + PE_DIM + 4
        self.layout_mlp = nn.Sequential(
            make_mlp(router_dim, 256, 4, depth=4),
            nn.Tanh()
        )
        self.gate_mlp = make_mlp(router_dim, 256, 3, depth=3)
        self.route_strength_mlp = nn.Sequential(
            make_mlp(router_dim, 256, 1, depth=2),
            nn.Sigmoid()
        )
        route_last = self.route_strength_mlp[0][-1]
        nn.init.zeros_(route_last.weight)
        nn.init.constant_(route_last.bias, 4.0)
        base_in_spec = POS_OUT + PE_DIM + 2 + (3 if with_sh else 0)
        resid_in_spec = POS_OUT * 2 + PE_DIM + 2 + 4 + (3 if with_sh else 0)
        self.component_heads = nn.ModuleList([
            make_mlp(base_in_spec, HEAD_HID, 3, siren=True),
            make_mlp(resid_in_spec, HEAD_HID, 3, siren=True),
            make_mlp(resid_in_spec, HEAD_HID, 3, siren=True),
        ])

        self.alpha_mlp = nn.Sequential(
            make_mlp(router_dim + 3, 256, 1, depth=2),
            nn.Sigmoid()
        )

    def forward(self, y, x, plane_hit, ray_d, cam_feat, front_alpha=None, front_rgb=None, offset_scale=1.0, gate_temperature=1.0, hard_gate_temperature=HARD_GATE_TEMPERATURE, adaptive_router_strength=1.0, return_aux=False):
        base_feat = self._feat(y, x, cam_feat)
        if front_alpha is None:
            front_alpha = torch.zeros_like(y, dtype=base_feat.dtype)
        else:
            front_alpha = front_alpha.to(dtype=base_feat.dtype)
        if front_rgb is None:
            front_rgb = torch.zeros((y.shape[0], 3), device=base_feat.device, dtype=base_feat.dtype)
        else:
            front_rgb = front_rgb.to(dtype=base_feat.dtype)

        ray_d_ = ray_d.to(dtype=base_feat.dtype)
        plane_hit_ = plane_hit.to(dtype=base_feat.dtype)
        view_pe = torch.cat([encode_dir(ray_d_).half(), encode_vec(plane_hit_).half()], dim=-1)
        router_in = torch.cat([base_feat, view_pe, front_alpha.unsqueeze(-1), front_rgb], dim=-1)

        layout_raw = self.layout_mlp(router_in).clamp(-1, 1)
        if offset_scale <= 0.0:
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
            ray_dir_samples = ray_d.unsqueeze(1).expand(-1, sample_y.shape[1], -1).reshape(-1, ray_d.shape[-1])
            sample_sh = self.sh_embed(
                safe_px(sample_y.reshape(-1)),
                safe_px(sample_x.reshape(-1)),
                ray_dir_samples.float(),
            ).view(sample_y.shape[0], sample_y.shape[1], -1)
            sh_center = sample_sh[:, 0].half()
            base_in = torch.cat([base_in, sh_center], dim=-1)
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
                sh_c = sample_sh[:, k].half()
                resid_in = torch.cat([resid_in, sh_c], dim=-1)
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


# ────────── RAY UTILS ──────────
def pixel_rays(y, x, Rwc, cam, z=0.0):
    """
    Build rays in half precision if Rwc is half, or float if Rwc is float.
    """
    f = RES / (2 * math.tan(math.radians(FOV_DEG / 2)))
    dx = (x + 0.5 - RES/2) / f
    dy = (y + 0.5 - RES/2) / f
    d_cam = torch.stack([dx, -dy, torch.ones_like(dx)], dim=-1)
    # multiply
    dirs = (Rwc.t() @ d_cam.t()).t()
    dirs = F.normalize(dirs, dim=-1)
    origin = cam.unsqueeze(0) + dirs * z
    return origin, dirs

def rot_yx(yaw, pitch):
    """Return a rotation matrix"""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    R_y = torch.tensor([
        [ cy, 0,  sy],
        [  0, 1,   0],
        [-sy, 0,  cy]], dtype=torch.float32, device=DEVICE)
    R_x = torch.tensor([
        [1,   0,   0],
        [0,  cp, -sp],
        [0,  sp,  cp]], dtype=torch.float32, device=DEVICE)
    return R_y @ R_x

def project_3D_to_2D(pts_3d, cam_pos, cam_yaw, cam_pitch, cam_dist, scr_w, scr_h):
    """
    Simple perspective for debug. Just CPU numpy math.
    """
    cy, sy = math.cos(cam_yaw), math.sin(cam_yaw)
    cp, sp = math.cos(cam_pitch), math.sin(cam_pitch)
    camX = cam_dist * cp*cy
    camY = cam_dist * sp
    camZ = cam_dist * cp*sy
    camera_world = np.array([camX, camY, camZ], dtype=np.float32)
    fwd = -camera_world
    fwd_len = np.linalg.norm(fwd) + 1e-9
    fwd /= fwd_len
    up0 = np.array([0,1,0], dtype=np.float32)
    right = np.cross(up0, fwd)
    right_len = np.linalg.norm(right) + 1e-9
    right /= right_len
    up = np.cross(fwd, right)

    out_2d = []
    for p in pts_3d:
        p_ = p - camera_world
        x_ = np.dot(p_, right)
        y_ = np.dot(p_, up)
        z_ = np.dot(p_, fwd)
        if z_ < 1e-3:
            out_2d.append((None,None))
            continue
        focal_ = 1.0
        sx = (x_ / z_) * focal_
        sy = (y_ / z_) * focal_
        s = min(scr_w, scr_h) * 0.4
        cx, cy = scr_w*0.5, scr_h*0.5
        px = cx + s*sx
        py = cy - s*sy
        out_2d.append((px, py))
    return out_2d


# ────────── MAIN ──────────
def main():
    global POS_ALL, PHI_ALL, THETA_ALL, RHO_ALL, N_CAM
    global CKPT_PATH, SH_FILE_FRONT, SH_FILE_REAR, K_NEIGH, SCALE, SHOW_3D_DEBUG
    parser = argparse.ArgumentParser()
    parser.add_argument("--renders-dir", type=Path, default=RENDERS_DIR,
                        help="Directory containing views.jsonl plus RGB/depth frames.")
    parser.add_argument("--checkpoint", type=Path, default=CKPT_PATH,
                        help="Checkpoint file to load.")
    parser.add_argument("--sh-file-front", type=Path, default=SH_FILE_FRONT,
                        help="Optional baked SH file for the front billboard.")
    parser.add_argument("--sh-file-rear", type=Path, default=None,
                        help="Optional baked SH file for the rear billboard.")
    parser.add_argument("--k-neigh", type=int, default=K_NEIGH,
                        help="How many nearby cameras to blend for pose interpolation.")
    parser.add_argument("--scale", type=int, default=SCALE,
                        help="Window upscaling factor.")
    parser.add_argument("--no-debug", action="store_true",
                        help="Hide the right-side debug viewport.")
    parser.add_argument("--headless", action="store_true",
                        help="Use SDL dummy mode for non-interactive smoke tests.")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Render this many frames and exit. 0 means interactive until closed.")
    parser.add_argument("--save-frame", type=Path, default=None,
                        help="Optional PNG path for the final on-screen frame.")
    parser.add_argument("--gate-temperature", type=float, default=1.0,
                        help="Rear transport gate temperature. Values above 1 soften routing across the three probes.")
    parser.add_argument("--hard-gate-temperature", type=float, default=HARD_GATE_TEMPERATURE,
                        help="Minimum per-ray gate temperature on hard rays when adaptive routing is active.")
    parser.add_argument("--adaptive-router-strength", type=float, default=1.0,
                        help="Blend from legacy learned routing (0) to adaptive easy-vs-hard routing (1).")
    parser.add_argument("--move-render-scale", type=float, default=0.5,
                        help="Internal render scale while the main camera is moving. 1 keeps full resolution at all times.")
    parser.add_argument("--idle-fullres-frames", type=int, default=6,
                        help="How many still frames to wait before snapping back to full internal resolution.")
    parser.add_argument("--force-render-scale", type=float, default=0.0,
                        help="Override internal render scale for every frame. 0 keeps adaptive movement scaling.")
    args = parser.parse_args()

    if args.headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    CKPT_PATH = args.checkpoint.expanduser()
    SH_FILE_FRONT = args.sh_file_front.expanduser() if args.sh_file_front else None
    SH_FILE_REAR = args.sh_file_rear.expanduser() if args.sh_file_rear else None
    K_NEIGH = max(1, args.k_neigh)
    SCALE = max(1, args.scale)
    SHOW_3D_DEBUG = not args.no_debug

    renders_dir = args.renders_dir.expanduser()
    POS_ALL, PHI_ALL, THETA_ALL, RHO_ALL = load_views(renders_dir)
    N_CAM = POS_ALL.shape[0]
    if N_CAM == 0:
        raise SystemExit(f"❌ no camera records found in {renders_dir}")

    print(f"Using renders dir: {renders_dir}")
    print(
        f"Viewer config: checkpoint={CKPT_PATH}, k_neigh={K_NEIGH}, debug={SHOW_3D_DEBUG}, "
        f"gate_temperature={args.gate_temperature:.2f}, hard_gate_temperature={args.hard_gate_temperature:.2f}, "
        f"adaptive_router_strength={args.adaptive_router_strength:.2f}, "
        f"move_render_scale={args.move_render_scale:.2f}, idle_fullres_frames={max(0, args.idle_fullres_frames)}, "
        f"force_render_scale={args.force_render_scale:.2f}"
    )

    centre_idx = N_CAM // 2
    cam_pos_np = POS_ALL[centre_idx].copy()

    neigh_idx,_ = nearest_k(cam_pos_np, K_NEIGH)
    box_lo, box_hi = bound_cluster(neigh_idx)

    front_has_sh = SH_FILE_FRONT is not None and Path(SH_FILE_FRONT).is_file()
    rear_has_sh = SH_FILE_REAR is not None and Path(SH_FILE_REAR).is_file()

    # load models
    ck = torch.load(CKPT_PATH, map_location="cpu")
    slab_splats = int(ck["slab"]["mean_raw"].shape[0]) if ("slab" in ck and "mean_raw" in ck["slab"]) else 32
    front = OcclusionSheet(RES, RES, front_has_sh).to(DEVICE)
    rear  = RefractionSheet(RES, RES, rear_has_sh).to(DEVICE)
    slab = GaussianSlab(num_splats=slab_splats).to(DEVICE)
    cam_mlp = CameraEmbed().to(DEVICE)  # float32

    front.load_state_dict(ck["front"], strict=False)
    rear.load_state_dict(ck["rear"], strict=False)
    if "slab" in ck:
        slab.load_state_dict(ck["slab"], strict=False)
    if "camera_embed" in ck:
        cam_mlp.load_state_dict(ck["camera_embed"], strict=False)

    z_off = ck.get("z_offset", 0.3)

    # now cast only front/rear to half
    front.half()
    rear.half()
    # keep cam_mlp in float32

    front.eval()
    rear.eval()
    slab.eval()
    cam_mlp.eval()

    phi_all_t = torch.from_numpy(PHI_ALL).to(DEVICE, dtype=torch.float32)
    theta_all_t = torch.from_numpy(THETA_ALL).to(DEVICE, dtype=torch.float32)
    rho_all_t = torch.from_numpy(RHO_ALL).to(DEVICE, dtype=torch.float32)
    all_cam_feats = cam_mlp(phi_all_t, theta_all_t, rho_all_t)

    if front_has_sh:
        front.sh_embed = SHEmbed(str(SH_FILE_FRONT))
    if rear_has_sh:
        rear.sh_embed = SHEmbed(str(SH_FILE_REAR))

    # init orientation
    yaw0, pitch0 = yaw_pitch_from_phi_theta(PHI_ALL[centre_idx], THETA_ALL[centre_idx])
    yaw, pitch = yaw0, pitch0

    # pygame
    if SHOW_3D_DEBUG:
        winW = RES*SCALE*2
        winH = RES*SCALE
    else:
        winW = RES*SCALE
        winH = RES*SCALE

    pygame.init()
    screen = pygame.display.set_mode((winW, winH))
    clock  = pygame.time.Clock()

    debug_yaw   = 0.0
    debug_pitch = 0.0
    debug_dist  = max( (box_hi - box_lo).max(), 0.001 ) * 3.0
    debug_dragActive = False
    last_mouseDbg = (0,0)
    projected_points_2d = []
    cached_main_surface = None
    cached_debug_surface = None
    last_debug_signature = None

    render_grid_cache = {}

    def get_render_grid(render_res: int):
        cached = render_grid_cache.get(render_res)
        if cached is None:
            cached = build_render_grid(render_res, device=DEVICE, dtype=torch.float16)
            render_grid_cache[render_res] = cached
        return cached

    last_idxs = None
    last_yaw, last_pitch = yaw, pitch
    last_render_res = None
    frames_since_motion = max(0, args.idle_fullres_frames)

    running = True
    frame_idx = 0

    with torch.inference_mode():
        while running:
            dt = clock.tick(60)/1000.0
            debug_changed = False
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_r:
                        cam_pos_np[:] = POS_ALL[centre_idx]
                        yaw, pitch = yaw0, pitch0
                        debug_changed = True

                if e.type == pygame.MOUSEBUTTONDOWN:
                    if e.button == 1:
                        mx, my = e.pos
                        if SHOW_3D_DEBUG:
                            debugSurfaceRect = pygame.Rect(RES*SCALE, 0, RES*SCALE, RES*SCALE)
                            if debugSurfaceRect.collidepoint(mx,my):
                                local_x = mx - debugSurfaceRect.left
                                local_y = my - debugSurfaceRect.top
                                radius_sq = 5*5
                                clicked_any = False
                                for i_cam, (sx, sy) in enumerate(projected_points_2d):
                                    if sx is None or sy is None:
                                        continue
                                    dx = sx - local_x
                                    dy = sy - local_y
                                    dist_sq = dx*dx + dy*dy
                                    if dist_sq <= radius_sq:
                                        cam_pos_np[:] = POS_ALL[i_cam]
                                        clicked_any = True
                                        debug_changed = True
                                        break
                                if not clicked_any:
                                    debug_dragActive = True
                                    last_mouseDbg= (mx,my)
                elif e.type == pygame.MOUSEBUTTONUP:
                    if e.button == 1:
                        debug_dragActive = False
                elif e.type == pygame.MOUSEWHEEL:
                    mx,my = pygame.mouse.get_pos()
                    if SHOW_3D_DEBUG:
                        debugSurfaceRect= pygame.Rect(RES*SCALE, 0, RES*SCALE, RES*SCALE)
                        if debugSurfaceRect.collidepoint(mx,my):
                            step= (0.5 if e.y>0 else -0.5)
                            debug_dist = max(debug_dist + step, 0.1)
                            debug_changed = True

            keys = pygame.key.get_pressed()
            move_speed = FLY_SPEED * dt
            Rwc_f32 = look_at_rotation(
                torch.tensor(cam_pos_np, device=DEVICE, dtype=torch.float32),
                device=DEVICE,
            )
            right_vec = Rwc_f32[0,:].cpu().numpy()
            up_vec    = Rwc_f32[1,:].cpu().numpy()
            fwd_vec   = Rwc_f32[2,:].cpu().numpy()

            move = np.array([0,0,0], dtype=np.float32)
            if keys[pygame.K_w]:
                move += fwd_vec
            if keys[pygame.K_s]:
                move -= fwd_vec
            if keys[pygame.K_a]:
                move -= right_vec
            if keys[pygame.K_d]:
                move += right_vec
            if keys[pygame.K_e]:
                move += up_vec
            if keys[pygame.K_q]:
                move -= up_vec

            cam_pos_prev = cam_pos_np.copy()
            cam_pos_np += move * move_speed

            idxs, dists = nearest_k(cam_pos_np, K_NEIGH)
            box_lo_, box_hi_ = bound_cluster(idxs)
            box_lo, box_hi   = box_lo_, box_hi_
            cam_pos_np = np.minimum(np.maximum(cam_pos_np, box_lo), box_hi)
            cam_pos_changed = not np.allclose(cam_pos_np, cam_pos_prev, atol=1e-6)
            if cam_pos_changed:
                frames_since_motion = 0
            else:
                frames_since_motion += 1
            if args.force_render_scale > 0.0:
                render_scale = args.force_render_scale
            elif frames_since_motion < max(0, args.idle_fullres_frames):
                render_scale = args.move_render_scale
            else:
                render_scale = 1.0
            render_res = quantize_render_res(render_scale)
            main_changed = cached_main_surface is None or cam_pos_changed or render_res != last_render_res

            if debug_dragActive and SHOW_3D_DEBUG:
                mx, my = pygame.mouse.get_pos()
                dx, dy = mx - last_mouseDbg[0], my - last_mouseDbg[1]
                last_mouseDbg = (mx,my)
                debug_yaw   += dx * 0.01
                debug_pitch += -dy * 0.01
                debug_pitch = np.clip(debug_pitch, -math.pi*0.49, math.pi*0.49)
                debug_changed = True

            weights = 1/(dists+1e-8)
            weights /= weights.sum()
            idxs_t = torch.as_tensor(idxs, device=DEVICE, dtype=torch.long)
            w_t = torch.as_tensor(weights, device=DEVICE, dtype=torch.float32).unsqueeze(1)
            cam_feat_interpolated = (all_cam_feats.index_select(0, idxs_t) * w_t).sum(dim=0)

            # orientation
            nearest = idxs[np.argmin(dists)]
            desired_yaw, desired_pitch = yaw_pitch_from_phi_theta(
                PHI_ALL[nearest], THETA_ALL[nearest]
            )
            yaw   = slerp_angle(yaw, desired_yaw, ORIENT_SMOOTH)
            pitch = slerp_angle(pitch, desired_pitch, ORIENT_SMOOTH)

            something_changed = (
                last_idxs is None or
                not np.array_equal(last_idxs, idxs) or
                abs(last_yaw - yaw) > 1e-5 or
                abs(last_pitch - pitch) > 1e-5
            )
            last_idxs = idxs
            last_yaw, last_pitch = yaw, pitch
            if SHOW_DEBUG_LOGS and something_changed:
                print(f"Nearest cluster changed. yaw={yaw:.3f} pitch={pitch:.3f}")

            if main_changed:
                yyf, xxf = get_render_grid(render_res)
                cam_pos_16 = torch.tensor(cam_pos_np, device=DEVICE, dtype=torch.float16)
                Rwc_half = Rwc_f32.half()

                ray_o, ray_d = pixel_rays(yyf, xxf, Rwc_half, cam_pos_16, z=0.0)
                front_center, rear_center, plane_normal, plane_u, plane_v = plane_frame(
                    cam_pos_16, z_off, device=DEVICE, dtype=ray_o.dtype
                )
                xF, yF, hitF, tF, validF = project_rays_to_plane(
                    ray_o, ray_d, front_center, plane_normal, plane_u, plane_v, RES
                )
                xR, yR, hitR, tR, validR = project_rays_to_plane(
                    ray_o, ray_d, rear_center, plane_normal, plane_u, plane_v, RES
                )

                if front.with_sh and front.sh_embed is not None:
                    front._dir_cache = ray_d

                pe_dirF = encode_dir(ray_d).half()
                pe_posF = encode_vec(hitF).half()
                peF = torch.cat([pe_dirF, pe_posF], dim=-1)
                rgbF, tauF = front(yF, xF, peF, cam_feat_interpolated)
                rgbF = rgbF.clamp(0,1) * validF.unsqueeze(-1).to(dtype=rgbF.dtype)
                tauF = torch.where(validF, tauF.clamp(0,1), torch.ones_like(tauF))
                alphaF = 1.0 - tauF

                rgbR, alphaR, _, rear_aux = rear(
                    yR, xR, hitR, ray_d, cam_feat_interpolated,
                    front_alpha=alphaF.detach(),
                    front_rgb=rgbF.detach(),
                    gate_temperature=args.gate_temperature,
                    hard_gate_temperature=args.hard_gate_temperature,
                    adaptive_router_strength=args.adaptive_router_strength,
                    return_aux=True,
                )
                rgbR   = rgbR.clamp(0,1) * validR.unsqueeze(-1).to(dtype=rgbR.dtype)
                alphaR = alphaR.clamp(0,1) * validR.to(dtype=alphaR.dtype)
                slab_context = slab_ray_context(
                    rear_aux["head_rgb"],
                    rear_aux["gates"],
                    rear_aux["route_strength"],
                    alphaF,
                    alphaR,
                )
                slab_trans = slab(
                    hitF, hitR,
                    front_center, rear_center,
                    plane_normal, plane_u, plane_v,
                    validF, validR,
                    ray_context=slab_context,
                )
                alphaR_eff = alphaR * slab_trans.to(dtype=alphaR.dtype)

                final, _, _ = composite_two_planes(
                    rgbF, alphaF, tF, validF,
                    rgbR, alphaR_eff, tR, validR,
                )
                frame_u8 = (
                    final.view(render_res, render_res, 3)
                    .clamp(0.0, 1.0)
                    .mul(255.0)
                    .to(torch.uint8)
                    .cpu()
                    .numpy()
                )
                cached_main_surface = pygame.surfarray.make_surface(frame_u8.swapaxes(0,1))
                if render_res != RES or SCALE != 1:
                    cached_main_surface = pygame.transform.smoothscale(cached_main_surface, (RES*SCALE, RES*SCALE))
                last_render_res = render_res

            screen.blit(cached_main_surface, (0,0))

            if SHOW_3D_DEBUG:
                debug_signature = (
                    round(float(debug_yaw), 5),
                    round(float(debug_pitch), 5),
                    round(float(debug_dist), 5),
                    tuple(np.round(cam_pos_np.astype(np.float32), 5)),
                )
                if debug_changed or main_changed or cached_debug_surface is None or debug_signature != last_debug_signature:
                    debugSurf= pygame.Surface((RES*SCALE, RES*SCALE))
                    debugSurf.fill((15,15,15))

                    pts_3d = POS_ALL.astype(np.float32)
                    proj2d = project_3D_to_2D(pts_3d, None, debug_yaw, debug_pitch, debug_dist,
                                             scr_w=RES*SCALE, scr_h=RES*SCALE)
                    projected_points_2d = proj2d
                    for i,(sx,sy) in enumerate(proj2d):
                        if sx is None:
                            continue
                        c= (100,100,255)
                        pygame.draw.circle(debugSurf, c, (int(sx), int(sy)), 3)

                    me_3d= cam_pos_np.astype(np.float32).reshape(1,3)
                    me2d= project_3D_to_2D(me_3d, None, debug_yaw, debug_pitch, debug_dist,
                                           RES*SCALE, RES*SCALE)
                    if me2d[0][0] is not None:
                        pygame.draw.circle(debugSurf, (255,50,50),
                                           (int(me2d[0][0]), int(me2d[0][1])), 5)
                    cached_debug_surface = debugSurf
                    last_debug_signature = debug_signature

                screen.blit(cached_debug_surface, (RES*SCALE, 0))

            pygame.display.flip()
            frame_idx += 1

            if args.max_frames and frame_idx >= args.max_frames:
                running = False

    if args.save_frame is not None and frame_idx > 0:
        pygame.image.save(screen, str(args.save_frame.expanduser()))
        print(f"Saved frame to {args.save_frame.expanduser()}")
    pygame.quit()


if __name__ == "__main__":
    main()
