import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from visor.plane_geometry import PLANE_HALF_EXTENT


SLAB_EPS = 1.0e-6


def _inv_sigmoid(x: float) -> float:
    x = min(max(float(x), 1.0e-5), 1.0 - 1.0e-5)
    return math.log(x / (1.0 - x))


def _inv_softplus(x: float) -> float:
    x = max(float(x), 1.0e-6)
    return math.log(math.expm1(x))


def _grid_means(num_splats: int, device=None, dtype=torch.float32) -> torch.Tensor:
    if num_splats <= 0:
        return torch.zeros((0, 3), device=device, dtype=dtype)

    side = math.ceil(num_splats ** (1.0 / 3.0))
    xs = torch.linspace(-0.7, 0.7, side, device=device, dtype=dtype)
    ys = torch.linspace(-0.7, 0.7, side, device=device, dtype=dtype)
    zs = torch.linspace(-0.45, 0.45, side, device=device, dtype=dtype)
    zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing="ij")
    pts = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
    return pts[:num_splats]


def slab_local_coords(
    points: torch.Tensor,
    slab_center: torch.Tensor,
    plane_normal: torch.Tensor,
    plane_u: torch.Tensor,
    plane_v: torch.Tensor,
    half_gap: torch.Tensor,
    half_extent: float = PLANE_HALF_EXTENT,
) -> torch.Tensor:
    pts = points.float()
    center = slab_center.to(device=pts.device, dtype=torch.float32)
    normal = plane_normal.to(device=pts.device, dtype=torch.float32)
    u_axis = plane_u.to(device=pts.device, dtype=torch.float32)
    v_axis = plane_v.to(device=pts.device, dtype=torch.float32)
    gap = torch.as_tensor(half_gap, device=pts.device, dtype=torch.float32).clamp_min(SLAB_EPS)

    rel = pts - center
    u = (rel * u_axis).sum(dim=-1) / max(float(half_extent), SLAB_EPS)
    v = -(rel * v_axis).sum(dim=-1) / max(float(half_extent), SLAB_EPS)
    w = (rel * normal).sum(dim=-1) / gap
    return torch.stack([u, v, w], dim=-1)


def _entropy_from_probs(probs: torch.Tensor, dim: int = -1) -> torch.Tensor:
    p = probs.clamp(1.0e-6, 1.0)
    return -(p * p.log()).sum(dim=dim)


def _normalize_batch_signal(signal: torch.Tensor, lo_q: float = 0.35, hi_q: float = 0.85) -> torch.Tensor:
    s = signal.float()
    if s.numel() == 0:
        return s
    lo = torch.quantile(s, lo_q)
    hi = torch.quantile(s, hi_q)
    scale = (hi - lo).clamp_min(1.0e-6)
    return ((s - lo) / scale).clamp(0.0, 1.0)


def slab_head_disagreement(head_rgb: torch.Tensor) -> torch.Tensor:
    heads = head_rgb.float()
    if heads.dim() != 3 or heads.shape[1] < 2:
        return torch.zeros(heads.shape[0], device=heads.device, dtype=torch.float32)
    pair_vals = [
        (heads[:, 0] - heads[:, 1]).abs().mean(dim=-1),
        (heads[:, 0] - heads[:, 2]).abs().mean(dim=-1),
        (heads[:, 1] - heads[:, 2]).abs().mean(dim=-1),
    ]
    return torch.stack(pair_vals, dim=1).mean(dim=1)


def slab_ray_context(
    head_rgb: torch.Tensor,
    gates: torch.Tensor,
    route_strength: torch.Tensor,
    front_alpha: torch.Tensor,
    rear_alpha: torch.Tensor,
) -> torch.Tensor:
    disagreement_raw = slab_head_disagreement(head_rgb)
    disagreement_norm = _normalize_batch_signal(disagreement_raw)
    gate_entropy = _entropy_from_probs(gates.float(), dim=-1)
    gate_entropy = gate_entropy / max(math.log(float(max(gates.shape[-1], 2))), 1.0e-6)
    return torch.stack([
        disagreement_norm,
        disagreement_raw.clamp(0.0, 1.0),
        route_strength.float().clamp(0.0, 1.0),
        gate_entropy.clamp(0.0, 1.0),
        front_alpha.float().clamp(0.0, 1.0),
        rear_alpha.float().clamp(0.0, 1.0),
    ], dim=-1)


class GaussianSlab(nn.Module):
    def __init__(
        self,
        num_splats: int = 32,
        min_scale: float = 0.05,
        max_scale: float = 0.65,
        init_scale_xy: float = 0.22,
        init_scale_z: float = 0.30,
        init_strength_bias: float = -5.0,
        mean_limit: float = 1.15,
        context_dim: int = 6,
        context_hidden: int = 16,
        context_latent: int = 8,
        init_disagreement_gain: float = 0.25,
        init_context_scale: float = 0.45,
    ):
        super().__init__()
        self.num_splats = int(num_splats)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.mean_limit = float(mean_limit)
        self.context_dim = int(context_dim)
        self.context_latent = int(context_latent)

        init_means = _grid_means(self.num_splats)
        self.mean_raw = nn.Parameter(torch.atanh((init_means / self.mean_limit).clamp(-0.999, 0.999)))

        init_scales = torch.tensor([init_scale_xy, init_scale_xy, init_scale_z], dtype=torch.float32)
        init_scales = ((init_scales - self.min_scale) / max(self.max_scale - self.min_scale, SLAB_EPS)).clamp(1.0e-4, 1.0 - 1.0e-4)
        self.scale_raw = nn.Parameter(torch.full((self.num_splats, 3), _inv_sigmoid(float(init_scales[0]))))
        self.scale_raw.data[:, 2] = _inv_sigmoid(float(init_scales[2]))

        self.strength_raw = nn.Parameter(torch.full((self.num_splats,), float(init_strength_bias)))
        self.context_mlp = nn.Sequential(
            nn.Linear(self.context_dim, context_hidden),
            nn.SiLU(),
            nn.Linear(context_hidden, self.context_latent),
        )
        nn.init.zeros_(self.context_mlp[-1].weight)
        nn.init.zeros_(self.context_mlp[-1].bias)
        self.context_keys = nn.Parameter(torch.randn(self.num_splats, self.context_latent) * 0.05)
        self.disagreement_gain_raw = nn.Parameter(torch.tensor(_inv_softplus(init_disagreement_gain), dtype=torch.float32))
        self.context_scale_raw = nn.Parameter(torch.tensor(_inv_softplus(init_context_scale), dtype=torch.float32))

    def means(self) -> torch.Tensor:
        return self.mean_limit * torch.tanh(self.mean_raw)

    def scales(self) -> torch.Tensor:
        return self.min_scale + torch.sigmoid(self.scale_raw) * (self.max_scale - self.min_scale)

    def strengths(self) -> torch.Tensor:
        return F.softplus(self.strength_raw)

    def disagreement_gain(self) -> torch.Tensor:
        return F.softplus(self.disagreement_gain_raw)

    def context_scale(self) -> torch.Tensor:
        return F.softplus(self.context_scale_raw)

    def forward(
        self,
        hit_f: torch.Tensor,
        hit_r: torch.Tensor,
        front_center: torch.Tensor,
        rear_center: torch.Tensor,
        plane_normal: torch.Tensor,
        plane_u: torch.Tensor,
        plane_v: torch.Tensor,
        valid_f: torch.Tensor,
        valid_r: torch.Tensor,
        ray_context: torch.Tensor | None = None,
        return_aux: bool = False,
    ):
        device = hit_f.device
        out_dtype = hit_f.dtype
        ray_count = hit_f.shape[0]
        trans = torch.ones(ray_count, device=device, dtype=out_dtype)
        tau = torch.zeros(ray_count, device=device, dtype=torch.float32)
        signal = torch.zeros(ray_count, device=device, dtype=torch.float32)
        ray_gain = torch.ones(ray_count, device=device, dtype=torch.float32)

        means = self.means()
        scales = self.scales()
        strengths = self.strengths()
        context_mod_mean = torch.tensor(1.0, device=device, dtype=torch.float32)
        context_mod_std = torch.tensor(0.0, device=device, dtype=torch.float32)
        if self.num_splats <= 0 or means.numel() == 0:
            aux = {
                "tau": tau,
                "alpha": 1.0 - trans.float(),
                "trans": trans.float(),
                "signal": signal,
                "ray_gain": ray_gain,
                "strength_mean": torch.tensor(0.0, device=device),
                "strength_max": torch.tensor(0.0, device=device),
                "scale_mean": torch.tensor(0.0, device=device),
                "mass_peak": torch.tensor(0.0, device=device),
                "depth_mean": torch.tensor(0.0, device=device),
                "context_mod_mean": context_mod_mean,
                "context_mod_std": context_mod_std,
                "disagreement_gain": torch.tensor(0.0, device=device),
                "context_scale": torch.tensor(0.0, device=device),
            }
            return (trans, aux) if return_aux else trans

        context_mod = torch.ones((ray_count, self.num_splats), device=device, dtype=torch.float32)
        if ray_context is not None and ray_context.numel() > 0:
            ctx = ray_context.to(device=device, dtype=torch.float32)
            signal = ctx[:, 0].clamp(0.0, 1.0)
            ctx_latent = self.context_mlp(ctx)
            if self.num_splats > 0:
                logits = (ctx_latent @ self.context_keys.t()) / math.sqrt(max(self.context_latent, 1))
                context_mod = 1.0 + signal.unsqueeze(-1) * self.context_scale() * torch.tanh(logits)
                context_mod = context_mod.clamp_min(0.25)
            ray_gain = (1.0 + signal * self.disagreement_gain()).clamp(0.25, 4.0)
            context_mod_mean = context_mod.mean()
            context_mod_std = context_mod.std(unbiased=False)

        slab_center = 0.5 * (front_center + rear_center)
        half_gap = 0.5 * (rear_center - front_center).norm().clamp_min(SLAB_EPS)
        segment_valid = (valid_f & valid_r).reshape(-1)
        if segment_valid.any():
            start_local = slab_local_coords(hit_f, slab_center, plane_normal, plane_u, plane_v, half_gap)
            end_local = slab_local_coords(hit_r, slab_center, plane_normal, plane_u, plane_v, half_gap)
            delta = (end_local - start_local).unsqueeze(1)
            diff0 = start_local.unsqueeze(1) - means.unsqueeze(0)
            inv_var = scales.reciprocal().square().unsqueeze(0)

            A = (delta.square() * inv_var).sum(dim=-1).clamp_min(SLAB_EPS)
            B = (diff0 * delta * inv_var).sum(dim=-1)
            C = (diff0.square() * inv_var).sum(dim=-1)

            denom = torch.sqrt(2.0 * A)
            erf_hi = torch.erf((A + B) / denom)
            erf_lo = torch.erf(B / denom)
            quad_term = (C - (B.square() / A)).clamp_min(0.0)
            integral = torch.exp(-0.5 * quad_term) * torch.sqrt((math.pi / 2.0) / A) * (erf_hi - erf_lo).clamp_min(0.0)
            mass = integral * strengths.unsqueeze(0) * context_mod * ray_gain.unsqueeze(-1)
            mass = mass * segment_valid.float().unsqueeze(-1)
            tau = mass.sum(dim=-1)
            trans = torch.exp(-tau).to(dtype=out_dtype)
            peak_mass = mass.max(dim=-1).values
            depth_mean = (mass * means[:, 2].unsqueeze(0)).sum(dim=-1) / tau.clamp_min(SLAB_EPS)
        else:
            mass = torch.zeros((ray_count, self.num_splats), device=device, dtype=torch.float32)
            peak_mass = torch.zeros(ray_count, device=device, dtype=torch.float32)
            depth_mean = torch.zeros(ray_count, device=device, dtype=torch.float32)

        if not return_aux:
            return trans

        aux = {
            "tau": tau,
            "alpha": 1.0 - trans.float(),
            "trans": trans.float(),
            "signal": signal,
            "ray_gain": ray_gain,
            "means": means.detach(),
            "scales": scales.detach(),
            "strengths": strengths.detach(),
            "mass": mass,
            "mass_peak": peak_mass,
            "depth_mean": depth_mean,
            "strength_mean": strengths.mean(),
            "strength_max": strengths.max(),
            "scale_mean": scales.mean(),
            "context_mod_mean": context_mod_mean,
            "context_mod_std": context_mod_std,
            "disagreement_gain": self.disagreement_gain(),
            "context_scale": self.context_scale(),
        }
        return trans, aux
