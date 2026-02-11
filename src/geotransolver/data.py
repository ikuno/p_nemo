"""2D cylinder potential flow data generation.

REQ-001: Synthetic data generation for 2D cylinder flow.
Analytical solution for inviscid, incompressible potential flow around a circular cylinder.

Velocity field (polar):
    v_r     = U * (1 - R^2/r^2) * cos(theta)
    v_theta = -U * (1 + R^2/r^2) * sin(theta)

Pressure (Bernoulli):
    p = p_inf + 0.5 * rho * U^2 * (1 - (v/U)^2)
"""

import math
import torch
from torch.utils.data import Dataset

from src.geotransolver.config import CylinderFlowConfig, DataConfig


def generate_domain_points(n_points: int, R: float, R_far: float) -> torch.Tensor:
    """REQ-001-01: Generate random 2D points in annular domain [R, R_far].

    Uses uniform sampling in polar coordinates with area correction.

    Returns:
        positions: (n_points, 2) Cartesian coordinates.
    """
    # Sample r^2 uniformly for area-correct sampling, then clamp to [R, R_far]
    r = torch.sqrt(torch.rand(n_points) * (R_far**2 - R**2) + R**2)
    theta = torch.rand(n_points) * 2 * math.pi
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    return torch.stack([x, y], dim=-1)


def generate_geometry_points(
    n_geom: int, R: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """REQ-001-04: Generate circle boundary points and outward normals.

    Returns:
        positions: (n_geom, 2) on the circle of radius R.
        normals: (n_geom, 2) unit outward normals.
    """
    theta = torch.linspace(0, 2 * math.pi, n_geom + 1)[:-1]
    x = R * torch.cos(theta)
    y = R * torch.sin(theta)
    positions = torch.stack([x, y], dim=-1)
    # Outward normal on circle = position / R
    normals = positions / R
    return positions, normals


def analytical_velocity(
    positions: torch.Tensor, R: float, U: float
) -> torch.Tensor:
    """REQ-001-02: Compute potential flow velocity in Cartesian coordinates.

    Args:
        positions: (..., 2) point positions (x, y).
        R: cylinder radius.
        U: freestream velocity magnitude.

    Returns:
        velocity: (..., 2) velocity (v_x, v_y).
    """
    x = positions[..., 0]
    y = positions[..., 1]
    r_sq = x**2 + y**2
    r_sq = torch.clamp(r_sq, min=1e-10)  # Avoid division by zero

    R2 = R**2
    # v_x = U * (1 - R^2*(x^2-y^2)/r^4)
    # v_y = U * (-2*R^2*x*y/r^4)
    # Derived from polar -> Cartesian transformation:
    #   v_x = v_r*cos(theta) - v_theta*sin(theta)
    #   v_y = v_r*sin(theta) + v_theta*cos(theta)
    r4 = r_sq**2
    vx = U * (1.0 - R2 * (x**2 - y**2) / r4)
    vy = U * (-2.0 * R2 * x * y / r4)

    return torch.stack([vx, vy], dim=-1)


def analytical_pressure(
    positions: torch.Tensor,
    velocity: torch.Tensor,
    rho: float,
    p_inf: float,
    U: float,
) -> torch.Tensor:
    """REQ-001-03: Compute pressure from Bernoulli equation.

    p = p_inf + 0.5 * rho * U^2 * (1 - (v_mag/U)^2)

    Args:
        positions: (..., 2) point positions.
        velocity: (..., 2) velocity at those points.
        rho: fluid density.
        p_inf: freestream pressure.
        U: freestream velocity.

    Returns:
        pressure: (..., 1) pressure values.
    """
    v_mag_sq = (velocity**2).sum(dim=-1, keepdim=True)
    p = p_inf + 0.5 * rho * (U**2 - v_mag_sq)
    return p


def generate_dataset(
    data_config: DataConfig, flow_config: CylinderFlowConfig
) -> dict:
    """REQ-001-05, REQ-001-06: Generate parametric dataset with train/val/test splits.

    Generates n_samples with varying U_inf, each with n_points domain points
    and n_geom geometry points.

    Returns:
        dict with 'train', 'val', 'test' keys, each containing:
            positions: (n_split, n_points, 2)
            geometry_positions: (n_geom, 2) shared across samples
            geometry_normals: (n_geom, 2) shared across samples
            global_params: (n_split, 1) U_inf values
            velocity: (n_split, n_points, 2)
            pressure: (n_split, n_points, 1)
    """
    n = data_config.n_samples
    U_lo, U_hi = data_config.U_range

    # Generate shared geometry
    geom_pos, geom_normals = generate_geometry_points(
        data_config.n_geom, flow_config.R
    )

    # Varying freestream velocities
    U_values = torch.linspace(U_lo, U_hi, n)

    # Generate domain points (shared positions, different fields per U)
    all_positions = []
    all_velocity = []
    all_pressure = []
    all_params = []

    for i in range(n):
        pts = generate_domain_points(
            data_config.n_points, flow_config.R, flow_config.R_far
        )
        U_i = U_values[i].item()
        vel = analytical_velocity(pts, flow_config.R, U_i)
        prs = analytical_pressure(pts, vel, flow_config.rho, flow_config.p_inf, U_i)

        all_positions.append(pts)
        all_velocity.append(vel)
        all_pressure.append(prs)
        all_params.append(torch.tensor([U_i]))

    positions = torch.stack(all_positions)  # (n, n_points, 2)
    velocity = torch.stack(all_velocity)  # (n, n_points, 2)
    pressure = torch.stack(all_pressure)  # (n, n_points, 1)
    global_params = torch.stack(all_params)  # (n, 1)

    # Split
    n_train = int(n * data_config.train_ratio)
    n_val = int(n * data_config.val_ratio)

    # Shuffle indices
    perm = torch.randperm(n)
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]

    def make_split(idx):
        return {
            "positions": positions[idx],
            "geometry_positions": geom_pos,
            "geometry_normals": geom_normals,
            "global_params": global_params[idx],
            "velocity": velocity[idx],
            "pressure": pressure[idx],
        }

    return {
        "train": make_split(train_idx),
        "val": make_split(val_idx),
        "test": make_split(test_idx),
    }


class CylinderFlowDataset(Dataset):
    """REQ-001: PyTorch Dataset wrapping generated cylinder flow data."""

    def __init__(self, split_data: dict):
        self.positions = split_data["positions"]
        self.geometry_positions = split_data["geometry_positions"]
        self.geometry_normals = split_data["geometry_normals"]
        self.global_params = split_data["global_params"]
        self.velocity = split_data["velocity"]
        self.pressure = split_data["pressure"]

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, idx: int) -> dict:
        return {
            "positions": self.positions[idx],
            "geometry_positions": self.geometry_positions,
            "geometry_normals": self.geometry_normals,
            "global_params": self.global_params[idx],
            "velocity": self.velocity[idx],
            "pressure": self.pressure[idx],
        }
