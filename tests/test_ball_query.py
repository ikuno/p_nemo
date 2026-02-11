"""Tests for 2D multi-scale ball query. REQ-002.

TDD Red phase: These tests define the expected behavior of ball_query.py.
"""

import torch
import pytest

from src.geotransolver.ball_query import (
    ball_query_2d,
    multi_scale_ball_query,
    gather_ball_query_features,
)


class TestBallQuery2D:
    """REQ-002-01, REQ-002-04, REQ-002-05: Single-scale 2D ball query."""

    def test_ball_query_radius_bound(self):
        """All returned neighbors are within the specified radius."""
        query = torch.tensor([[[0.0, 0.0], [3.0, 0.0]]])  # (1, 2, 2)
        ref = torch.tensor(
            [[[0.1, 0.1], [0.5, 0.0], [2.9, 0.0], [10.0, 10.0]]]
        )  # (1, 4, 2)
        indices, dists = ball_query_2d(query, ref, radius=1.0, max_neighbors=4)
        # For valid (non-sentinel) entries, distance should be <= radius
        valid = indices != -1
        assert torch.all(dists[valid] <= 1.0 + 1e-6)

    def test_ball_query_finds_self(self):
        """When query == reference, each point finds itself."""
        pts = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # (1, 2, 2)
        indices, dists = ball_query_2d(pts, pts, radius=0.01, max_neighbors=4)
        # Each point should find itself (distance ~0)
        # The first valid neighbor for each point should be itself
        for i in range(2):
            valid_idx = indices[0, i][indices[0, i] != -1]
            assert i in valid_idx.tolist()

    def test_ball_query_max_neighbors(self):
        """Output shape is (B, N, max_neighbors)."""
        query = torch.randn(2, 10, 2)
        ref = torch.randn(2, 20, 2)
        indices, dists = ball_query_2d(query, ref, radius=5.0, max_neighbors=8)
        assert indices.shape == (2, 10, 8)
        assert dists.shape == (2, 10, 8)

    def test_ball_query_padding_sentinel(self):
        """Isolated point far from all references returns -1 sentinels."""
        query = torch.tensor([[[100.0, 100.0]]])  # Far away
        ref = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]])
        indices, dists = ball_query_2d(query, ref, radius=1.0, max_neighbors=4)
        assert torch.all(indices[0, 0] == -1)

    def test_ball_query_empty_neighborhood(self):
        """Point with no neighbors within radius gets all -1."""
        query = torch.tensor([[[50.0, 50.0]]])
        ref = torch.tensor([[[0.0, 0.0]]])
        indices, dists = ball_query_2d(query, ref, radius=1.0, max_neighbors=2)
        assert torch.all(indices == -1)
        assert torch.all(dists == float("inf"))

    def test_2d_euclidean_distance(self):
        """Known point pair returns correct distance."""
        query = torch.tensor([[[0.0, 0.0]]])
        ref = torch.tensor([[[3.0, 4.0]]])  # Distance = 5.0
        indices, dists = ball_query_2d(query, ref, radius=10.0, max_neighbors=1)
        assert torch.allclose(dists[0, 0, 0], torch.tensor(5.0), atol=1e-5)

    def test_ball_query_sorted_by_distance(self):
        """Returned neighbors are sorted by distance (closest first)."""
        query = torch.tensor([[[0.0, 0.0]]])
        ref = torch.tensor([[[3.0, 0.0], [1.0, 0.0], [2.0, 0.0]]])
        indices, dists = ball_query_2d(query, ref, radius=5.0, max_neighbors=3)
        valid = indices[0, 0] != -1
        valid_dists = dists[0, 0][valid]
        # Should be sorted ascending
        assert torch.all(valid_dists[:-1] <= valid_dists[1:])


class TestMultiScaleBallQuery:
    """REQ-002-02: Multi-scale ball query."""

    def test_multi_scale_count(self):
        """Returns S sets of (indices, dists)."""
        query = torch.randn(1, 8, 2)
        ref = torch.randn(1, 12, 2)
        scales = [(0.5, 4), (1.0, 8), (2.0, 16)]
        results = multi_scale_ball_query(query, ref, scales)
        assert len(results) == 3
        for indices, dists in results:
            assert indices.shape[0] == 1
            assert indices.shape[1] == 8

    def test_multi_scale_neighbor_counts(self):
        """Each scale uses its own max_neighbors."""
        query = torch.randn(1, 5, 2)
        ref = torch.randn(1, 20, 2)
        scales = [(1.0, 4), (2.0, 8)]
        results = multi_scale_ball_query(query, ref, scales)
        assert results[0][0].shape[2] == 4
        assert results[1][0].shape[2] == 8

    def test_multi_scale_nesting(self):
        """Neighbors found at smaller radius are a subset of larger radius neighbors."""
        # Dense cluster so that smaller radius definitely finds some neighbors
        query = torch.tensor([[[0.0, 0.0]]])
        ref_pts = torch.randn(1, 50, 2) * 0.5  # Points clustered near origin
        scales = [(0.3, 50), (1.0, 50)]
        results = multi_scale_ball_query(query, ref_pts, scales)
        small_idx = set(results[0][0][0][results[0][0][0] != -1].tolist())
        large_idx = set(results[1][0][0][results[1][0][0] != -1].tolist())
        # All neighbors at r=0.3 should also be neighbors at r=1.0
        assert small_idx.issubset(large_idx)


class TestGatherFeatures:
    """REQ-002-03: Feature gathering at ball query indices."""

    def test_gather_features_shape(self):
        """Output shape is (B, N, k, d_feat)."""
        features = torch.randn(2, 10, 8)  # (B, M, d_feat)
        indices = torch.randint(0, 10, (2, 5, 4))  # (B, N, k)
        out = gather_ball_query_features(features, indices)
        assert out.shape == (2, 5, 4, 8)

    def test_gather_features_padding_zero(self):
        """Sentinel indices (-1) yield zero vectors."""
        features = torch.randn(1, 5, 3)
        indices = torch.tensor([[[0, -1, -1]]])  # (1, 1, 3)
        out = gather_ball_query_features(features, indices)
        # Position 0 should have the feature, positions 1,2 should be zeros
        assert torch.allclose(out[0, 0, 0], features[0, 0])
        assert torch.allclose(out[0, 0, 1], torch.zeros(3))
        assert torch.allclose(out[0, 0, 2], torch.zeros(3))

    def test_gather_features_correct_values(self):
        """Gathered features match reference at the given indices."""
        features = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
        indices = torch.tensor([[[2, 0]]])  # Pick features[2] and features[0]
        out = gather_ball_query_features(features, indices)
        assert torch.allclose(out[0, 0, 0], torch.tensor([5.0, 6.0]))
        assert torch.allclose(out[0, 0, 1], torch.tensor([1.0, 2.0]))
