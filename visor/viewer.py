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

import json, math, os, warnings

try:
    from tinycudann import hashgrid
except ImportError:
    import types, tinycudann as tcnn, torch.nn as nn
    class _HashGrid(nn.Module):
        def __init__(self,
                     n_levels, n_features_per_level,
                     log2_hashmap_size, base_resolution,
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
        def forward(self, uv):
            return self.enc(uv)
    hashgrid = types.SimpleNamespace(HashGrid=_HashGrid)

import numpy as np
from pathlib import Path
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from math import factorial, pi
from visor import PROJECT_ROOT, RENDERS_DIR

# ────────── USER KNOBS ──────────
RES            = 512
CKPT_PATH      = PROJECT_ROOT / "dual_billboard_0512_x2_cont_F7_sh.pt"
SH_FILE_FRONT  = PROJECT_ROOT / "sh_billboard_L7.pt"
SH_FILE_REAR   = ""
K_NEIGH        = 15
PE_BANDS       = 8
FOV_DEG        = 60.0

ORIENT_SMOOTH  = 1.0
FLY_SPEED      = 1.0
SCALE          = 1
SHOW_3D_DEBUG  = True
SHOW_DEBUG_LOGS= False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ────────── UTILS ──────────
def safe_px(t: torch.Tensor, res: int = RES) -> torch.LongTensor:
    """Round, clamp and cast to LONG for safe indexing."""
    return t.round().clamp_(0, res - 1).long()

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

POS_ALL, PHI_ALL, THETA_ALL, RHO_ALL = load_views(RENDERS_DIR)
N_CAM = POS_ALL.shape[0]

def nearest_k(p, k=K_NEIGH):
    d2 = np.sum((POS_ALL - p)**2, 1)
    idx = np.argpartition(d2, k)[:k]
    return idx, np.sqrt(d2[idx])

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
        # keep it in float32
        φf = φ.float()
        θf = θ.float()
        ρf = ρ.float()
        inp = torch.stack([φf, θf, ρf], -1).unsqueeze(0)  # shape [1,3]
        out = self.net(inp)  # shape [1, 32]
        return out[0]        # shape [32]


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
        yL = safe_px(y).to(device=DEVICE)
        xL = safe_px(x).to(device=DEVICE)

        # convert to half AFTER safe_px
        uv = torch.stack([xL, yL], dim=-1).float() / RES  # uv is float
        uv = uv.half()

        codes_xy = self.codes(uv)  # returns half, shape [B, code_dim]

        cam_feat_ = cam_feat.to(dtype=torch.half)  # ensure half
        # expand shape
        cam_exp = cam_feat_.unsqueeze(0).expand_as(codes_xy)
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
        self.delta_mlp = nn.Sequential(
            make_mlp(POS_OUT, 256, 9, depth=4),
            nn.Tanh()
        )
        in_spec = POS_OUT + PE_DIM + (3 if with_sh else 0)
        self.cR = make_mlp(in_spec, HEAD_HID, 3, siren=True)
        self.cG = make_mlp(in_spec, HEAD_HID, 3, siren=True)
        self.cB = make_mlp(in_spec, HEAD_HID, 3, siren=True)
        self.mix= nn.Conv1d(3,3,1,bias=False)
        nn.init.eye_(self.mix.weight.squeeze(-1))

        self.alpha_mlp = nn.Sequential(
            make_mlp(POS_OUT + PE_DIM, 256, 1, depth=2),
            nn.Sigmoid()
        )

    def forward(self, y, x, ray_o, ray_d, cam_feat):
        """
        y, x => indexing
        ray_o, ray_d => half or float => we unify to half or float as needed
        cam_feat => we unify to half
        """
        base_feat = self._feat(y, x, cam_feat)  # [B,POS_OUT] half

        delta_raw = self.delta_mlp(base_feat)  # half
        delta_raw = delta_raw.clamp(-1,1)
        # reshape => (B, 3, 3)
        B = y.size(0)
        delta = delta_raw.view(B, 3, 3)
        # unify ray_d => half
        ray_d_ = ray_d.to(dtype=torch.half)
        # add delta
        dirs = ray_d_.unsqueeze(1) + delta * math.radians(2.0)
        dirs = F.normalize(dirs, dim=-1)  # shape [B,3,3]

        # Flatten => shape [B*3, 3]
        dirs_all = dirs.reshape(-1,3)

        # optional SH
        sh_col_all = None
        if self.with_sh and self.sh_embed is not None:
            yL = safe_px(y)
            xL = safe_px(x)
            # cast dirs_all to float32 for SH
            dirs_all32 = dirs_all.float()
            raw_col_all = self.sh_embed.forward_all(yL, xL, dirs_all32)  # float32 => [B*3,3]
            sh_col_all  = raw_col_all.half()

        # expand base_feat => shape [B, 3, POS_OUT], then flatten => [B*3, POS_OUT]
        base_feat_all = base_feat.unsqueeze(1).expand(B,3, base_feat.size(-1)).reshape(-1, base_feat.size(-1))  # half

        # build direction + position encoding (pe_dir, pe_loc)
        pe_dir_all = encode_dir(dirs_all)  # float
        pe_dir_all = pe_dir_all.half()

        ray_o_ = ray_o.to(dtype=torch.half)
        pe_loc_o = encode_vec(ray_o_).unsqueeze(1)  # shape [B,1,pe_dim]
        pe_loc_o = pe_loc_o.expand(B,3, pe_loc_o.size(-1)).reshape(B*3, -1)  # half

        if self.with_sh and sh_col_all is not None:
            feat_all = torch.cat([base_feat_all, pe_dir_all, pe_loc_o, sh_col_all], dim=-1)
        else:
            feat_all = torch.cat([base_feat_all, pe_dir_all, pe_loc_o], dim=-1)

        # chunk into 3
        in_cR = feat_all[0::3]
        in_cG = feat_all[1::3]
        in_cB = feat_all[2::3]

        cR = torch.sigmoid(self.cR(in_cR))
        cG = torch.sigmoid(self.cG(in_cG))
        cB = torch.sigmoid(self.cB(in_cB))
        # shape => [B, 3, 3]
        col = torch.stack([cR,cG,cB], dim=1)
        col = self.mix(col)  # => [B, 3, 3]

        idx3 = torch.arange(3, device=dirs.device)
        rgbR = col[:, idx3, idx3]  # shape [B, 3], half

        # alpha from direction 0 only
        pe_dir_0 = pe_dir_all[0::3]
        # base_feat => [B,POS_OUT], so cat => shape [B, (POS_OUT+pe_dim)]
        alpha_in = torch.cat([base_feat, pe_dir_0, encode_vec(ray_o_).half()], dim=-1)
        alpha_val = self.alpha_mlp(alpha_in).squeeze(-1)

        return rgbR, alpha_val, delta_raw


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
    centre_idx = N_CAM // 2
    cam_pos_np = POS_ALL[centre_idx].copy()

    neigh_idx,_ = nearest_k(cam_pos_np, K_NEIGH)
    box_lo, box_hi = bound_cluster(neigh_idx)

    # load models
    ck = torch.load(CKPT_PATH, map_location="cpu")
    front = OcclusionSheet(RES, RES, bool(SH_FILE_FRONT)).to(DEVICE)
    rear  = RefractionSheet(RES, RES, bool(SH_FILE_REAR)).to(DEVICE)
    cam_mlp = CameraEmbed().to(DEVICE)  # float32

    front.load_state_dict(ck["front"], strict=False)
    rear.load_state_dict(ck["rear"], strict=False)
    if "camera_embed" in ck:
        cam_mlp.load_state_dict(ck["camera_embed"], strict=False)

    z_off = ck.get("z_offset", 0.3)

    # now cast only front/rear to half
    front.half()
    rear.half()
    # keep cam_mlp in float32

    front.eval()
    rear.eval()
    cam_mlp.eval()

    if SH_FILE_FRONT and os.path.isfile(SH_FILE_FRONT):
        front.sh_embed = SHEmbed(SH_FILE_FRONT)
    if SH_FILE_REAR and os.path.isfile(SH_FILE_REAR):
        rear.sh_embed = SHEmbed(SH_FILE_REAR)

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

    # meshgrid in half
    yy, xx = torch.meshgrid(
        torch.arange(RES, device=DEVICE, dtype=torch.float16),
        torch.arange(RES, device=DEVICE, dtype=torch.float16),
        indexing="ij"
    )
    yyf, xxf = yy.flatten(), xx.flatten()

    last_idxs = None
    last_yaw, last_pitch = yaw, pitch

    running = True

    with torch.inference_mode():
        while running:
            dt = clock.tick(60)/1000.0
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_r:
                        cam_pos_np[:] = POS_ALL[centre_idx]
                        yaw, pitch = yaw0, pitch0

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

            keys = pygame.key.get_pressed()
            move_speed = FLY_SPEED * dt
            # orientation from yaw/pitch
            Rwc_f32 = rot_yx(yaw, pitch)  # float32
            right_vec = Rwc_f32[0,:].cpu().numpy()
            up_vec    = Rwc_f32[1,:].cpu().numpy()
            fwd_vec   = Rwc_f32[2,:].cpu().numpy()

            move = np.array([0,0,0], dtype=np.float32)
            if keys[pygame.K_a]:
                move += fwd_vec
            if keys[pygame.K_d]:
                move -= fwd_vec
            if keys[pygame.K_w]:
                move -= right_vec
            if keys[pygame.K_s]:
                move += right_vec
            if keys[pygame.K_e]:
                move += up_vec
            if keys[pygame.K_q]:
                move -= up_vec

            cam_pos_np += move * move_speed

            idxs, dists = nearest_k(cam_pos_np, K_NEIGH)
            box_lo_, box_hi_ = bound_cluster(idxs)
            box_lo, box_hi   = box_lo_, box_hi_
            cam_pos_np = np.minimum(np.maximum(cam_pos_np, box_lo), box_hi)

            if debug_dragActive and SHOW_3D_DEBUG:
                mx, my = pygame.mouse.get_pos()
                dx, dy = mx - last_mouseDbg[0], my - last_mouseDbg[1]
                last_mouseDbg = (mx,my)
                debug_yaw   += dx * 0.01
                debug_pitch += -dy * 0.01
                debug_pitch = np.clip(debug_pitch, -math.pi*0.49, math.pi*0.49)

            weights = 1/(dists+1e-8)
            weights /= weights.sum()

            phi_neighbors   = PHI_ALL[idxs]
            theta_neighbors = THETA_ALL[idxs]
            rho_neighbors   = RHO_ALL[idxs]

            # build neighbor_embeds in float32
            phi_t   = torch.tensor(phi_neighbors, device=DEVICE, dtype=torch.float32)
            theta_t = torch.tensor(theta_neighbors, device=DEVICE, dtype=torch.float32)
            rho_t   = torch.tensor(rho_neighbors, device=DEVICE, dtype=torch.float32)

            neighbor_embeds = []
            for iN in range(K_NEIGH):
                neighbor_embeds.append(cam_mlp(phi_t[iN], theta_t[iN], rho_t[iN]))  # [32], float32

            neighbor_embeds = torch.stack(neighbor_embeds, dim=0)  # shape [K_NEIGH,32], float32

            w_t = torch.tensor(weights, device=DEVICE, dtype=torch.float32).unsqueeze(1)  # [K_NEIGH,1]
            cam_feat_interpolated = (neighbor_embeds * w_t).sum(dim=0)  # [32], float32

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

            # convert camera pos to float16 only for the rendering pass
            cam_pos_16 = torch.tensor(cam_pos_np, device=DEVICE, dtype=torch.float16)
            # orientation matrix in half
            Rwc_half = Rwc_f32.half()

            # billboard pass
            oF, dF = pixel_rays(yyf, xxf, Rwc_half, cam_pos_16, z=0.0)
            # store directions for SH
            if front.with_sh and front.sh_embed is not None:
                front._dir_cache = dF  # shape [B,3], half

            # build the positional encodes in half
            pe_dirF = encode_dir(dF)     # float => cast to half
            pe_dirF = pe_dirF.half()
            # expand cam_pos for each pixel
            cpos_exp = cam_pos_16.unsqueeze(0).expand_as(dF)
            pe_posF  = encode_vec(cpos_exp)  # float => cast
            pe_posF  = pe_posF.half()

            peF = torch.cat([pe_dirF, pe_posF], dim=-1)  # half
            # forward
            rgbF, tauF = front(yyf, xxf, peF, cam_feat_interpolated)  # half
            rgbF = rgbF.clamp(0,1)
            tauF = tauF.clamp(0,1)

            # refraction
            oR, dR = pixel_rays(yyf, xxf, Rwc_half, cam_pos_16, z_off)
            rgbR, alphaR, _ = rear(yyf, xxf, oR, dR, cam_feat_interpolated)
            rgbR   = rgbR.clamp(0,1)
            alphaR = alphaR.clamp(0,1)

            final = rgbF + (1 - tauF).unsqueeze(-1) * (alphaR.unsqueeze(-1) * rgbR)
            # shape => [B,3], B=RES*RES => reshape => [RES,RES,3]

            out_np = (final.view(RES,RES,3).float().cpu().numpy() * 255).astype(np.uint8)

            mainSurf = pygame.surfarray.make_surface(out_np.swapaxes(0,1))
            if SCALE != 1:
                mainSurf = pygame.transform.smoothscale(mainSurf, (RES*SCALE, RES*SCALE))
            screen.blit(mainSurf, (0,0))

            if SHOW_3D_DEBUG:
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

                screen.blit(debugSurf, (RES*SCALE, 0))

            pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
