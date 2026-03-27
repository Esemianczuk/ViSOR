import math
import types

import torch
import torch.nn as nn


def _aligned_level_size(resolution: int, log2_hashmap_size: int) -> int:
    params_in_level = resolution * resolution
    params_in_level = ((params_in_level + 7) // 8) * 8
    return min(params_in_level, 1 << log2_hashmap_size)


class _TorchHashEncoding(nn.Module):
    def __init__(
        self,
        n_levels: int,
        n_features_per_level: int,
        log2_hashmap_size: int,
        base_resolution: int,
        per_level_scale: float,
    ):
        super().__init__()
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.per_level_scale = per_level_scale

        self.resolutions = []
        self.level_offsets = []

        offset = 0
        log2_scale = math.log2(per_level_scale)
        for level in range(n_levels):
            scale = (2.0 ** (level * log2_scale)) * base_resolution - 1.0
            resolution = int(math.ceil(scale) + 1)
            self.resolutions.append(resolution)
            self.level_offsets.append(offset)
            offset += _aligned_level_size(resolution, log2_hashmap_size)

        self.total_entries = offset
        self.params = nn.Parameter(torch.empty(offset * n_features_per_level))
        nn.init.uniform_(self.params, -1.0e-4, 1.0e-4)

    @staticmethod
    def _coherent_prime_hash(gx: torch.Tensor, gy: torch.Tensor) -> torch.Tensor:
        mask = 0xFFFFFFFF
        return ((gx & mask) ^ ((gy * 2654435761) & mask)) & mask

    def _grid_index(self, gx: torch.Tensor, gy: torch.Tensor, resolution: int, hashmap_size: int) -> torch.Tensor:
        dense_size = resolution * resolution
        if hashmap_size >= dense_size:
            return gx + gy * resolution
        return self._coherent_prime_hash(gx, gy) % hashmap_size

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        xy_f32 = xy.float()
        table = self.params.view(self.total_entries, self.n_features_per_level)
        outs = []

        for level, resolution in enumerate(self.resolutions):
            offset = self.level_offsets[level]
            hashmap_size = (
                self.level_offsets[level + 1] - offset
                if level + 1 < len(self.level_offsets)
                else self.total_entries - offset
            )
            scale = resolution - 1.0

            pos = xy_f32 * scale + 0.5
            pos_grid = torch.floor(pos).long()
            frac = pos - pos_grid.float()

            gx0, gy0 = pos_grid[:, 0], pos_grid[:, 1]
            gx1, gy1 = gx0 + 1, gy0 + 1
            wx, wy = frac[:, 0:1], frac[:, 1:2]
            one = torch.ones_like(wx)

            idx00 = offset + self._grid_index(gx0, gy0, resolution, hashmap_size)
            idx10 = offset + self._grid_index(gx1, gy0, resolution, hashmap_size)
            idx01 = offset + self._grid_index(gx0, gy1, resolution, hashmap_size)
            idx11 = offset + self._grid_index(gx1, gy1, resolution, hashmap_size)

            feat00 = table[idx00]
            feat10 = table[idx10]
            feat01 = table[idx01]
            feat11 = table[idx11]

            out_level = (
                feat00 * ((one - wx) * (one - wy))
                + feat10 * (wx * (one - wy))
                + feat01 * ((one - wx) * wy)
                + feat11 * (wx * wy)
            )
            outs.append(out_level)

        return torch.cat(outs, dim=-1).to(dtype=xy.dtype)


class _TorchHashGrid(nn.Module):
    def __init__(
        self,
        n_levels: int,
        n_features_per_level: int,
        log2_hashmap_size: int,
        base_resolution: int,
        per_level_scale: float,
    ):
        super().__init__()
        self.enc = _TorchHashEncoding(
            n_levels=n_levels,
            n_features_per_level=n_features_per_level,
            log2_hashmap_size=log2_hashmap_size,
            base_resolution=base_resolution,
            per_level_scale=per_level_scale,
        )

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        return self.enc(xy)


def _build_hashgrid_namespace():
    try:
        from tinycudann import hashgrid as native_hashgrid
        return native_hashgrid
    except ImportError:
        pass

    try:
        import tinycudann as tcnn
    except ImportError:
        return types.SimpleNamespace(HashGrid=_TorchHashGrid)

    class _CompatHashGrid(nn.Module):
        def __init__(
            self,
            n_levels: int,
            n_features_per_level: int,
            log2_hashmap_size: int,
            base_resolution: int,
            per_level_scale: float,
        ):
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
                ),
            )

        def forward(self, xy: torch.Tensor) -> torch.Tensor:
            return self.enc(xy)

    return types.SimpleNamespace(HashGrid=_CompatHashGrid)


hashgrid = _build_hashgrid_namespace()
