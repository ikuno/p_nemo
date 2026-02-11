"""Tests for benchmarking evaluation metrics. REQ-006.

TDD Red phase: Tests for MAE, relative L1, R², and force computation.
Based on Benchmarking Framework paper (arXiv: 2507.10747v1).
"""

import math
import torch
import pytest

from src.geotransolver.metrics import (
    mean_absolute_error,
    relative_l1_norm,
    r_squared,
    compute_drag_lift_2d,
)


class TestMAE:
    """REQ-006-01: Mean Absolute Error."""

    def test_mae_zero_for_identical(self):
        """MAE(x, x) == 0."""
        x = torch.randn(10, 3)
        assert torch.allclose(mean_absolute_error(x, x), torch.tensor(0.0), atol=1e-7)

    def test_mae_known_value(self):
        """Hand-computed MAE matches."""
        pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target = torch.tensor([[2.0, 2.0], [1.0, 4.0]])
        # |1-2|+|2-2|+|3-1|+|4-4| = 1+0+2+0 = 3, MAE = 3/4 = 0.75
        expected = torch.tensor(0.75)
        assert torch.allclose(mean_absolute_error(pred, target), expected, atol=1e-6)

    def test_mae_nonnegative(self):
        """MAE >= 0 always."""
        pred = torch.randn(50, 3)
        target = torch.randn(50, 3)
        assert mean_absolute_error(pred, target) >= 0.0

    def test_mae_symmetric(self):
        """MAE(a, b) == MAE(b, a)."""
        a = torch.randn(20, 2)
        b = torch.randn(20, 2)
        assert torch.allclose(
            mean_absolute_error(a, b), mean_absolute_error(b, a), atol=1e-6
        )


class TestRelativeL1:
    """REQ-006-02: Relative L1 Norm."""

    def test_relative_l1_zero_for_identical(self):
        """rel_L1(x, x) == 0."""
        x = torch.randn(10, 3) + 1  # Ensure nonzero for division
        assert torch.allclose(
            relative_l1_norm(x, x), torch.tensor(0.0), atol=1e-7
        )

    def test_relative_l1_known_value(self):
        """Hand-computed value matches."""
        pred = torch.tensor([[2.0], [4.0]])
        target = torch.tensor([[1.0], [2.0]])
        # sum(|pred-target|) = |1| + |2| = 3
        # sum(|target|) = 1 + 2 = 3
        # rel_L1 = 3/3 = 1.0
        expected = torch.tensor(1.0)
        assert torch.allclose(
            relative_l1_norm(pred, target), expected, atol=1e-6
        )

    def test_relative_l1_normalized(self):
        """2x error on 2x signal -> same relative L1."""
        pred1 = torch.tensor([[2.0]])
        target1 = torch.tensor([[1.0]])
        pred2 = torch.tensor([[4.0]])
        target2 = torch.tensor([[2.0]])
        r1 = relative_l1_norm(pred1, target1)
        r2 = relative_l1_norm(pred2, target2)
        assert torch.allclose(r1, r2, atol=1e-6)


class TestRSquared:
    """REQ-006-03: R-squared coefficient of determination."""

    def test_r_squared_perfect(self):
        """R²(x, x) == 1.0."""
        x = torch.randn(20)
        assert torch.allclose(r_squared(x, x), torch.tensor(1.0), atol=1e-6)

    def test_r_squared_known_value(self):
        """Hand-computed R² matches."""
        target = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        pred = torch.tensor([1.1, 2.0, 2.9, 4.1, 4.9])
        # SS_res = 0.01 + 0 + 0.01 + 0.01 + 0.01 = 0.04
        # mean = 3.0, SS_tot = 4+1+0+1+4 = 10
        # R² = 1 - 0.04/10 = 0.996
        expected = torch.tensor(0.996)
        assert torch.allclose(r_squared(pred, target), expected, atol=1e-3)

    def test_r_squared_mean_predictor(self):
        """R² = 0 when pred = mean(target)."""
        target = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        pred = torch.full_like(target, target.mean())
        assert torch.allclose(r_squared(pred, target), torch.tensor(0.0), atol=1e-6)

    def test_r_squared_worse_than_mean(self):
        """R² < 0 when predictions are worse than mean."""
        target = torch.tensor([1.0, 2.0, 3.0])
        pred = torch.tensor([10.0, 20.0, 30.0])  # Way off
        assert r_squared(pred, target) < 0.0


class TestDragLift2D:
    """REQ-006-04: 2D drag and lift from surface pressure integration."""

    def test_uniform_pressure_zero_force(self):
        """Uniform pressure on circle -> zero net force."""
        n = 64
        theta = torch.linspace(0, 2 * math.pi, n + 1)[:-1]
        normals = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
        pressure = torch.ones(n, 1)  # Uniform pressure
        ds = torch.full((n,), 2 * math.pi / n)  # Equal arc lengths
        drag, lift = compute_drag_lift_2d(pressure, normals, ds)
        assert torch.allclose(drag, torch.tensor(0.0), atol=1e-4)
        assert torch.allclose(lift, torch.tensor(0.0), atol=1e-4)

    def test_dalembert_paradox(self):
        """Potential flow around cylinder -> zero drag (D'Alembert's paradox)."""
        n = 256
        R = 1.0
        U = 1.0
        theta = torch.linspace(0, 2 * math.pi, n + 1)[:-1]
        normals = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
        # Pressure on cylinder surface from potential flow:
        # p = p_inf + 0.5*rho*U^2*(1 - 4*sin^2(theta))
        # (since v_theta = -2U*sin(theta) on surface)
        p_surface = 0.5 * (1 - 4 * torch.sin(theta) ** 2)
        pressure = p_surface.unsqueeze(-1)
        ds = torch.full((n,), 2 * math.pi * R / n)
        drag, lift = compute_drag_lift_2d(pressure, normals, ds)
        assert torch.allclose(drag, torch.tensor(0.0), atol=1e-3)
        assert torch.allclose(lift, torch.tensor(0.0), atol=1e-3)

    def test_drag_lift_shape(self):
        """Output are scalar tensors."""
        pressure = torch.randn(10, 1)
        normals = torch.randn(10, 2)
        ds = torch.ones(10)
        drag, lift = compute_drag_lift_2d(pressure, normals, ds)
        assert drag.shape == ()
        assert lift.shape == ()
