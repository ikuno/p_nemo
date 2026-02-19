"""Multi-cylinder rectangular domain data generation.

REQ-MC-001: Synthetic data for 3-cylinder potential flow in a rectangular domain.

Analytical solution: superposition of potential flow around each cylinder.

For cylinder i centered at (x_i, y_i) with radius R_i:
    xi = x - x_i,  eta = y - y_i,  r_i^2 = xi^2 + eta^2

Perturbation velocity (doublet):
    dv_x_i = U * R_i^2 * (eta^2 - xi^2) / r_i^4
    dv_y_i = U * R_i^2 * (-2 * xi * eta)  / r_i^4

Total velocity:
    v_x = U + sum_i(dv_x_i)
    v_y = 0 + sum_i(dv_y_i)

Pressure (Bernoulli):
    p = p_inf + 0.5 * rho * (U^2 - |v|^2)
"""

import math
import torch
from torch.utils.data import Dataset

from src.geotransolver.config import (
    MultiCylinderFlowConfig,
    RectDataConfig,
    CylinderSpec,
)


# ---------------------------------------------------------------------------
# Domain point generation
# ---------------------------------------------------------------------------

def generate_rect_domain_points(
    n_points: int,
    domain: tuple[float, float, float, float],
    cylinders: list[CylinderSpec],
    *,
    seed: int | None = None,
) -> torch.Tensor:
    """REQ-MC-002-01: Sample points uniformly in the rectangular domain.

    Points inside any cylinder are rejected (rejection sampling).

    Args:
        n_points: Number of valid points to return.
        domain: (x_min, x_max, y_min, y_max).
        cylinders: List of CylinderSpec (center + radius).
        seed: Optional RNG seed for reproducibility.

    Returns:
        positions: (n_points, 2) Cartesian coordinates.
    """
    x_min, x_max, y_min, y_max = domain
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)

    accepted = []
    # Cylinder area fraction is small (~1.8%), so batch sampling is efficient
    batch = n_points * 4
    while len(accepted) < n_points:
        xs = torch.empty(batch).uniform_(x_min, x_max, generator=rng)
        ys = torch.empty(batch).uniform_(y_min, y_max, generator=rng)
        pts = torch.stack([xs, ys], dim=-1)  # (batch, 2)

        # Keep points outside ALL cylinders
        valid = torch.ones(batch, dtype=torch.bool)
        for cyl in cylinders:
            cx, cy, R = cyl.x, cyl.y, cyl.R
            dist2 = (pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2
            valid &= dist2 >= R * R

        accepted.append(pts[valid])
        batch = max(batch, (n_points - sum(len(a) for a in accepted)) * 4 + 1)

    return torch.cat(accepted, dim=0)[:n_points]


# ---------------------------------------------------------------------------
# Geometry (boundary) point generation
# ---------------------------------------------------------------------------

def generate_multi_cylinder_geometry(
    n_geom: int,
    cylinders: list[CylinderSpec],
) -> tuple[torch.Tensor, torch.Tensor]:
    """REQ-MC-002-02: Generate boundary points on all cylinders.

    Distributes n_geom points equally among all cylinders.

    Returns:
        positions: (n_geom_actual, 2) boundary point coordinates.
        normals:   (n_geom_actual, 2) outward unit normals.
    """
    n_cyl = len(cylinders)
    pts_per_cyl = n_geom // n_cyl

    all_positions = []
    all_normals = []
    for cyl in cylinders:
        theta = torch.linspace(0.0, 2.0 * math.pi, pts_per_cyl + 1)[:-1]
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        pos = torch.stack(
            [cyl.x + cyl.R * cos_t, cyl.y + cyl.R * sin_t], dim=-1
        )  # (pts_per_cyl, 2)
        # Outward normal = (cos_t, sin_t)
        nrm = torch.stack([cos_t, sin_t], dim=-1)  # (pts_per_cyl, 2)
        all_positions.append(pos)
        all_normals.append(nrm)

    return torch.cat(all_positions, dim=0), torch.cat(all_normals, dim=0)


# ---------------------------------------------------------------------------
# Analytical solution
# ---------------------------------------------------------------------------

def analytical_multi_cylinder_velocity(
    positions: torch.Tensor,
    cylinders: list[CylinderSpec],
    U: float,
) -> torch.Tensor:
    """REQ-MC-001-01: Superposed potential flow velocity.

    Args:
        positions: (..., 2) point coordinates.
        cylinders: List of CylinderSpec.
        U: Freestream velocity (x-direction).

    Returns:
        velocity: (..., 2) [v_x, v_y].
    """
    x = positions[..., 0]
    y = positions[..., 1]

    vx = torch.full_like(x, U)  # start from freestream
    vy = torch.zeros_like(y)

    for cyl in cylinders:
        xi = x - cyl.x
        eta = y - cyl.y
        r2 = xi * xi + eta * eta
        # Clamp to avoid division by zero (shouldn't happen after rejection sampling)
        r2 = r2.clamp(min=1e-8)
        r4 = r2 * r2
        R2 = cyl.R * cyl.R

        vx = vx + U * R2 * (eta * eta - xi * xi) / r4
        vy = vy + U * R2 * (-2.0 * xi * eta) / r4

    return torch.stack([vx, vy], dim=-1)


def analytical_multi_cylinder_pressure(
    positions: torch.Tensor,
    velocity: torch.Tensor,
    rho: float,
    p_inf: float,
    U: float,
) -> torch.Tensor:
    """REQ-MC-001-02: Bernoulli pressure for superposed potential flow.

    Args:
        positions: (..., 2) unused (kept for API symmetry).
        velocity:  (..., 2) velocity field.
        rho:       Fluid density.
        p_inf:     Far-field pressure.
        U:         Freestream speed.

    Returns:
        pressure: (..., 1) pressure field.
    """
    v_mag2 = (velocity ** 2).sum(dim=-1)  # (...)
    p = p_inf + 0.5 * rho * (U * U - v_mag2)
    return p.unsqueeze(-1)  # (..., 1)


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_multi_cylinder_dataset(
    data_cfg: RectDataConfig,
    flow_cfg: MultiCylinderFlowConfig,
    seed: int = 42,
) -> dict:
    """REQ-MC-002-03: Generate train/val/test split for multi-cylinder flow.

    Returns a dict matching the single-cylinder generate_dataset() structure:
    {
        "train" / "val" / "test": {
            "positions":            (n_split, n_points, 2),
            "geometry_positions":   (n_geom_actual, 2),   # shared, same for all
            "geometry_normals":     (n_geom_actual, 2),   # shared
            "global_params":        (n_split, 1),
            "velocity":             (n_split, n_points, 2),
            "pressure":             (n_split, n_points, 1),
        }
    }
    """
    torch.manual_seed(seed)

    U_lo, U_hi = flow_cfg.U_range
    U_values = torch.linspace(U_lo, U_hi, data_cfg.n_samples)

    # Geometry is fixed (same for all samples)
    geom_pos, geom_normals = generate_multi_cylinder_geometry(
        data_cfg.n_geom, flow_cfg.cylinders
    )

    all_positions = []
    all_velocity = []
    all_pressure = []
    all_params = []

    for i, U in enumerate(U_values):
        u_val = U.item()
        pts = generate_rect_domain_points(
            data_cfg.n_points,
            flow_cfg.domain,
            flow_cfg.cylinders,
            seed=seed + i,
        )  # (n_points, 2)

        vel = analytical_multi_cylinder_velocity(pts, flow_cfg.cylinders, u_val)
        prs = analytical_multi_cylinder_pressure(
            pts, vel, flow_cfg.rho, flow_cfg.p_inf, u_val
        )

        all_positions.append(pts)
        all_velocity.append(vel)
        all_pressure.append(prs)
        all_params.append(torch.tensor([u_val]))

    positions_t = torch.stack(all_positions)   # (n_samples, n_points, 2)
    velocity_t  = torch.stack(all_velocity)    # (n_samples, n_points, 2)
    pressure_t  = torch.stack(all_pressure)    # (n_samples, n_points, 1)
    params_t    = torch.stack(all_params)      # (n_samples, 1)

    # Shuffle
    perm = torch.randperm(data_cfg.n_samples)
    positions_t = positions_t[perm]
    velocity_t  = velocity_t[perm]
    pressure_t  = pressure_t[perm]
    params_t    = params_t[perm]

    # Split
    n = data_cfg.n_samples
    n_train = int(n * data_cfg.train_ratio)
    n_val   = int(n * data_cfg.val_ratio)

    def _split(t: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
        return t[lo:hi]

    splits = {}
    for name, lo, hi in [
        ("train", 0, n_train),
        ("val", n_train, n_train + n_val),
        ("test", n_train + n_val, n),
    ]:
        splits[name] = {
            "positions":          _split(positions_t, lo, hi),
            "geometry_positions": geom_pos,
            "geometry_normals":   geom_normals,
            "global_params":      _split(params_t, lo, hi),
            "velocity":           _split(velocity_t, lo, hi),
            "pressure":           _split(pressure_t, lo, hi),
        }

    return splits


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class MultiCylinderFlowDataset(Dataset):
    """REQ-MC-002-04: PyTorch Dataset for multi-cylinder flow samples."""

    def __init__(self, data: dict):
        self.positions          = data["positions"]           # (N, n_pts, 2)
        self.geometry_positions = data["geometry_positions"]  # (n_geom, 2)
        self.geometry_normals   = data["geometry_normals"]    # (n_geom, 2)
        self.global_params      = data["global_params"]       # (N, 1)
        self.velocity           = data["velocity"]            # (N, n_pts, 2)
        self.pressure           = data["pressure"]            # (N, n_pts, 1)

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, idx: int) -> dict:
        return {
            "positions":          self.positions[idx],
            "geometry_positions": self.geometry_positions,
            "geometry_normals":   self.geometry_normals,
            "global_params":      self.global_params[idx],
            "velocity":           self.velocity[idx],
            "pressure":           self.pressure[idx],
        }
