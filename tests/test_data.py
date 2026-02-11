"""Tests for 2D cylinder flow data generation. REQ-001.

TDD Red phase: These tests define the expected behavior of data.py.
"""

import math
import torch
import pytest

from src.geotransolver.data import (
    generate_domain_points,
    generate_geometry_points,
    analytical_velocity,
    analytical_pressure,
    generate_dataset,
    CylinderFlowDataset,
)
from src.geotransolver.config import CylinderFlowConfig, DataConfig


class TestDomainPoints:
    """REQ-001-01: Domain point generation in annular region."""

    def test_domain_points_within_annulus(self, flow_config):
        """All points satisfy R <= r <= R_far."""
        pts = generate_domain_points(256, flow_config.R, flow_config.R_far)
        r = torch.norm(pts, dim=-1)
        assert torch.all(r >= flow_config.R - 1e-6)
        assert torch.all(r <= flow_config.R_far + 1e-6)

    def test_domain_points_shape(self, flow_config):
        """Output shape is (n_points, 2)."""
        pts = generate_domain_points(128, flow_config.R, flow_config.R_far)
        assert pts.shape == (128, 2)

    def test_domain_points_dtype(self, flow_config):
        """Points are float32 tensors."""
        pts = generate_domain_points(64, flow_config.R, flow_config.R_far)
        assert pts.dtype == torch.float32


class TestGeometryPoints:
    """REQ-001-04: Circle boundary geometry points."""

    def test_geometry_points_on_circle(self, flow_config):
        """All geometry points satisfy ||g_j|| == R."""
        positions, normals = generate_geometry_points(64, flow_config.R)
        r = torch.norm(positions, dim=-1)
        assert torch.allclose(r, torch.full_like(r, flow_config.R), atol=1e-6)

    def test_geometry_normals_unit_length(self, flow_config):
        """All normals have unit length."""
        positions, normals = generate_geometry_points(64, flow_config.R)
        norms = torch.norm(normals, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    def test_geometry_normals_outward(self, flow_config):
        """Normals point outward: dot(n_j, g_j) > 0."""
        positions, normals = generate_geometry_points(64, flow_config.R)
        dots = (normals * positions).sum(dim=-1)
        assert torch.all(dots > 0)

    def test_geometry_points_shape(self, flow_config):
        """Positions (n_geom, 2) and normals (n_geom, 2)."""
        positions, normals = generate_geometry_points(32, flow_config.R)
        assert positions.shape == (32, 2)
        assert normals.shape == (32, 2)


class TestAnalyticalVelocity:
    """REQ-001-02: Potential flow velocity field."""

    def test_velocity_boundary_no_penetration(self):
        """v_r = 0 at r=R (no-penetration boundary condition)."""
        R, U = 1.0, 1.0
        # Points on the cylinder surface
        theta = torch.linspace(0, 2 * math.pi, 33)[:-1]
        pts = torch.stack([R * torch.cos(theta), R * torch.sin(theta)], dim=-1)
        vel = analytical_velocity(pts, R, U)
        # Radial component: v_r = (v . r_hat) should be zero on surface
        r_hat = pts / torch.norm(pts, dim=-1, keepdim=True)
        v_r = (vel * r_hat).sum(dim=-1)
        assert torch.allclose(v_r, torch.zeros_like(v_r), atol=1e-5)

    def test_velocity_far_field(self):
        """v → (U, 0) as r → large."""
        R, U = 1.0, 1.5
        # Points far from cylinder
        r_far = 100.0
        theta = torch.linspace(0, 2 * math.pi, 17)[:-1]
        pts = torch.stack(
            [r_far * torch.cos(theta), r_far * torch.sin(theta)], dim=-1
        )
        vel = analytical_velocity(pts, R, U)
        expected = torch.tensor([U, 0.0]).expand_as(vel)
        assert torch.allclose(vel, expected, atol=0.01)

    def test_velocity_shape(self):
        """Output shape matches input points."""
        pts = torch.randn(50, 2) + 3.0  # Ensure outside cylinder
        vel = analytical_velocity(pts, R=1.0, U=1.0)
        assert vel.shape == (50, 2)

    def test_velocity_varies_with_U(self):
        """Different U_inf values produce different velocity fields."""
        pts = torch.tensor([[3.0, 0.0], [0.0, 3.0]])
        v1 = analytical_velocity(pts, R=1.0, U=1.0)
        v2 = analytical_velocity(pts, R=1.0, U=2.0)
        assert not torch.allclose(v1, v2)


class TestAnalyticalPressure:
    """REQ-001-03: Pressure from Bernoulli equation."""

    def test_pressure_stagnation_point(self):
        """p = p_inf + 0.5*rho*U^2 at front stagnation point (theta=0, r=R)
        where velocity is zero."""
        R, U, rho, p_inf = 1.0, 1.0, 1.0, 0.0
        # Front stagnation: theta=0, r=R -> v_r=0, v_theta=0 (both cancel)
        # Actually for potential flow: v_theta = -2U*sin(0) = 0, v_r = 0
        pts = torch.tensor([[R, 0.0]])
        vel = analytical_velocity(pts, R, U)
        p = analytical_pressure(pts, vel, rho, p_inf, U)
        expected = p_inf + 0.5 * rho * U**2
        assert torch.allclose(p, torch.tensor([[expected]]), atol=1e-5)

    def test_pressure_symmetry(self):
        """p(theta) = p(-theta) for symmetric flow."""
        R, U, rho, p_inf = 1.0, 1.0, 1.0, 0.0
        theta = torch.tensor([math.pi / 4])
        pt_pos = torch.stack([2 * torch.cos(theta), 2 * torch.sin(theta)], dim=-1)
        pt_neg = torch.stack([2 * torch.cos(-theta), 2 * torch.sin(-theta)], dim=-1)
        v_pos = analytical_velocity(pt_pos, R, U)
        v_neg = analytical_velocity(pt_neg, R, U)
        p_pos = analytical_pressure(pt_pos, v_pos, rho, p_inf, U)
        p_neg = analytical_pressure(pt_neg, v_neg, rho, p_inf, U)
        assert torch.allclose(p_pos, p_neg, atol=1e-6)

    def test_pressure_shape(self):
        """Output shape is (n_points, 1)."""
        pts = torch.randn(30, 2) + 3.0
        vel = analytical_velocity(pts, R=1.0, U=1.0)
        p = analytical_pressure(pts, vel, rho=1.0, p_inf=0.0, U=1.0)
        assert p.shape == (30, 1)


class TestDataset:
    """REQ-001-05, REQ-001-06: Parametric dataset with splits."""

    def test_dataset_splits_size(self, data_config, flow_config):
        """Split sizes match configured ratios."""
        ds = generate_dataset(data_config, flow_config)
        total = data_config.n_samples
        assert len(ds["train"]["global_params"]) == int(total * data_config.train_ratio)
        assert len(ds["val"]["global_params"]) == int(total * data_config.val_ratio)
        # test gets the remainder
        expected_test = total - int(total * data_config.train_ratio) - int(
            total * data_config.val_ratio
        )
        assert len(ds["test"]["global_params"]) == expected_test

    def test_dataset_splits_disjoint(self, data_config, flow_config):
        """Train/val/test sets have no overlapping U_inf values."""
        ds = generate_dataset(data_config, flow_config)
        train_u = set(ds["train"]["global_params"][:, 0].tolist())
        val_u = set(ds["val"]["global_params"][:, 0].tolist())
        test_u = set(ds["test"]["global_params"][:, 0].tolist())
        assert len(train_u & val_u) == 0
        assert len(train_u & test_u) == 0
        assert len(val_u & test_u) == 0

    def test_parametric_velocity_varies(self, flow_config):
        """Different U_inf values produce different fields."""
        cfg = DataConfig(n_samples=5, n_points=32, n_geom=16, U_range=(0.5, 2.0))
        ds = generate_dataset(cfg, flow_config)
        vels = ds["train"]["velocity"]
        # At least some samples should have different velocity fields
        assert not torch.allclose(vels[0], vels[-1])

    def test_pytorch_dataset(self, data_config, flow_config):
        """CylinderFlowDataset is a valid torch Dataset."""
        ds = generate_dataset(data_config, flow_config)
        torch_ds = CylinderFlowDataset(ds["train"])
        assert len(torch_ds) > 0
        sample = torch_ds[0]
        assert "positions" in sample
        assert "velocity" in sample
        assert "pressure" in sample
