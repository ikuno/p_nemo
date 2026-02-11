"""Tests for context projector module. REQ-003.

TDD Red phase: Tests for geometry-context projection.
"""

import torch
import pytest

from src.geotransolver.context_projector import ContextProjector
from src.geotransolver.config import ModelConfig, BallQueryConfig


@pytest.fixture
def projector(small_model_config, bq_config):
    """Small ContextProjector for testing."""
    return ContextProjector(small_model_config, bq_config)


class TestContextProjector:
    """REQ-003: Context vector construction."""

    def test_context_vector_shape(
        self, projector, sample_points_2d, sample_geometry_2d, sample_global_params
    ):
        """REQ-003-01: Context C has shape (B, d_context)."""
        geom_pos, geom_normals = sample_geometry_2d
        context, augmentation = projector(
            sample_global_params, geom_pos, geom_normals,
            sample_points_2d, sample_points_2d,
        )
        assert context.shape == (1, projector.config.d_context)

    def test_augmentation_shape(
        self, projector, sample_points_2d, sample_geometry_2d, sample_global_params
    ):
        """REQ-003-04: Augmentation output shape (B, N, d_aug)."""
        geom_pos, geom_normals = sample_geometry_2d
        context, augmentation = projector(
            sample_global_params, geom_pos, geom_normals,
            sample_points_2d, sample_points_2d,
        )
        B, N, _ = sample_points_2d.shape
        assert augmentation.shape[0] == B
        assert augmentation.shape[1] == N
        assert augmentation.shape[2] > 0  # Has some augmentation features

    def test_context_contains_global(
        self, projector, sample_points_2d, sample_geometry_2d
    ):
        """REQ-003-01: Different global params produce different context C."""
        geom_pos, geom_normals = sample_geometry_2d
        p1 = torch.tensor([[1.0]])
        p2 = torch.tensor([[2.0]])
        c1, _ = projector(p1, geom_pos, geom_normals, sample_points_2d, sample_points_2d)
        c2, _ = projector(p2, geom_pos, geom_normals, sample_points_2d, sample_points_2d)
        assert not torch.allclose(c1, c2)

    def test_context_geom_pool_permutation_invariant(
        self, projector, sample_points_2d, sample_global_params
    ):
        """REQ-003-02: Permuting geometry points yields same context."""
        torch.manual_seed(42)
        n_geom = 16
        geom_pos = torch.randn(1, n_geom, 2)
        geom_normals = torch.randn(1, n_geom, 2)
        geom_normals = geom_normals / torch.norm(geom_normals, dim=-1, keepdim=True)

        # Compute context with original order
        c1, _ = projector(
            sample_global_params, geom_pos, geom_normals,
            sample_points_2d, sample_points_2d,
        )

        # Permute geometry points
        perm = torch.randperm(n_geom)
        geom_pos_perm = geom_pos[:, perm]
        geom_normals_perm = geom_normals[:, perm]
        c2, _ = projector(
            sample_global_params, geom_pos_perm, geom_normals_perm,
            sample_points_2d, sample_points_2d,
        )

        # c_geom component should be identical (mean pooling is permutation-invariant)
        # Full context may differ slightly due to ball query ordering
        # but c_geom part should be the same
        assert torch.allclose(c1, c2, atol=1e-5)

    def test_context_projector_gradient_flows(
        self, projector, sample_points_2d, sample_geometry_2d, sample_global_params
    ):
        """REQ-003-04: Gradients flow through all parameters."""
        geom_pos, geom_normals = sample_geometry_2d
        context, aug = projector(
            sample_global_params, geom_pos, geom_normals,
            sample_points_2d, sample_points_2d,
        )
        loss = context.sum() + aug.sum()
        loss.backward()
        for name, param in projector.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_context_projector_deterministic(
        self, projector, sample_points_2d, sample_geometry_2d, sample_global_params
    ):
        """Same inputs produce identical outputs."""
        geom_pos, geom_normals = sample_geometry_2d
        projector.eval()
        c1, a1 = projector(
            sample_global_params, geom_pos, geom_normals,
            sample_points_2d, sample_points_2d,
        )
        c2, a2 = projector(
            sample_global_params, geom_pos, geom_normals,
            sample_points_2d, sample_points_2d,
        )
        assert torch.allclose(c1, c2)
        assert torch.allclose(a1, a2)
