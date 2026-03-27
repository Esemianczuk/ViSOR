import torch


WORLD_UP = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
SCENE_CENTER = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
PLANE_HALF_EXTENT = 2.0
PLANE_EPS = 1.0e-6


def _normalize(vec: torch.Tensor) -> torch.Tensor:
    return vec / vec.norm(dim=-1, keepdim=True).clamp_min(PLANE_EPS)


def look_at_rotation(cam_loc, target=None, device=None, dtype=torch.float32):
    cam = torch.as_tensor(cam_loc, device=device, dtype=dtype)
    tgt = SCENE_CENTER.to(device=cam.device, dtype=dtype) if target is None else torch.as_tensor(target, device=device, dtype=dtype)
    up = WORLD_UP.to(device=cam.device, dtype=dtype)

    fwd = _normalize(tgt - cam)
    right = _normalize(torch.cross(up, fwd, dim=0))
    up2 = _normalize(torch.cross(fwd, right, dim=0))
    return torch.stack([right, up2, fwd], dim=0)


def plane_frame(cam_loc, z_offset, device, dtype=torch.float32):
    cam = torch.as_tensor(cam_loc, device=device, dtype=dtype)
    center = SCENE_CENTER.to(device=device, dtype=dtype)
    up = WORLD_UP.to(device=device, dtype=dtype)

    forward = _normalize(center - cam)
    right = _normalize(torch.cross(up, forward, dim=0))
    up2 = _normalize(torch.cross(forward, right, dim=0))

    half_gap = torch.as_tensor(z_offset, device=device, dtype=dtype) * 0.5
    front_center = center - forward * half_gap
    rear_center = center + forward * half_gap
    return front_center, rear_center, forward, right, up2


def project_rays_to_plane(
    ray_o,
    ray_d,
    plane_center,
    plane_normal,
    u_axis,
    v_axis,
    res: int,
    half_extent: float = PLANE_HALF_EXTENT,
):
    ray_o32 = ray_o.float()
    ray_d32 = ray_d.float()
    plane_center32 = plane_center.to(device=ray_o32.device, dtype=torch.float32)
    normal = plane_normal.to(device=ray_o32.device, dtype=torch.float32)
    u_axis = u_axis.to(device=ray_o32.device, dtype=torch.float32)
    v_axis = v_axis.to(device=ray_o32.device, dtype=torch.float32)

    denom = (ray_d32 * normal).sum(dim=-1)
    numer = ((plane_center32 - ray_o32) * normal).sum(dim=-1)
    safe_denom = torch.where(denom.abs() > PLANE_EPS, denom, torch.ones_like(denom))
    t = numer / safe_denom
    hit = ray_o32 + t.unsqueeze(-1) * ray_d32

    rel = hit - plane_center32
    u = 0.5 + (rel * u_axis).sum(dim=-1) / (2.0 * half_extent)
    v = 0.5 - (rel * v_axis).sum(dim=-1) / (2.0 * half_extent)

    x = u * (res - 1)
    y = v * (res - 1)
    valid = (
        (denom.abs() > PLANE_EPS) &
        (t > PLANE_EPS) &
        (u >= 0.0) & (u <= 1.0) &
        (v >= 0.0) & (v <= 1.0)
    )
    return x, y, hit, t, valid


def composite_two_planes(rgb_a, alpha_a, t_a, valid_a, rgb_b, alpha_b, t_b, valid_b):
    inf = torch.full_like(t_a, float("inf"))
    t_stack = torch.stack([
        torch.where(valid_a, t_a, inf),
        torch.where(valid_b, t_b, inf),
    ], dim=1)
    alpha_stack = torch.stack([
        alpha_a * valid_a.to(dtype=alpha_a.dtype),
        alpha_b * valid_b.to(dtype=alpha_b.dtype),
    ], dim=1)
    rgb_stack = torch.stack([rgb_a, rgb_b], dim=1)

    order = torch.argsort(t_stack, dim=1)
    alpha_sorted = torch.gather(alpha_stack, 1, order)
    rgb_sorted = torch.gather(rgb_stack, 1, order.unsqueeze(-1).expand(-1, -1, rgb_stack.size(-1)))

    near_alpha = alpha_sorted[:, 0:1]
    far_alpha = alpha_sorted[:, 1:2]
    pred = near_alpha * rgb_sorted[:, 0] + (1.0 - near_alpha) * far_alpha * rgb_sorted[:, 1]
    return pred, alpha_sorted[:, 0], alpha_sorted[:, 1]
