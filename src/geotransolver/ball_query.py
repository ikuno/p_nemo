"""2D multi-scale ball query for neighborhood search.

REQ-002: Ball query operations for GeoTransolver's multi-scale
geometry-aware neighborhood computation.

Uses torch.cdist for pairwise distance computation (Article VIII:
use framework APIs directly, no external point cloud libraries needed
for 2D scale).
"""

import torch


def ball_query_2d(
    query_points: torch.Tensor,
    reference_points: torch.Tensor,
    radius: float,
    max_neighbors: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """REQ-002-01, REQ-002-04, REQ-002-05: Single-scale 2D ball query.

    For each query point, find up to max_neighbors reference points
    within the given radius, sorted by distance.

    Args:
        query_points: (B, N, 2) query positions.
        reference_points: (B, M, 2) reference positions.
        radius: search radius.
        max_neighbors: maximum neighbors per query point.

    Returns:
        indices: (B, N, max_neighbors) int64, padded with -1 sentinel.
        dists: (B, N, max_neighbors) float, padded with inf.
    """
    B, N, _ = query_points.shape
    M = reference_points.shape[1]

    # Pairwise distances: (B, N, M)
    all_dists = torch.cdist(query_points, reference_points)

    # Mask out points beyond radius
    mask = all_dists > radius
    all_dists_masked = all_dists.clone()
    all_dists_masked[mask] = float("inf")

    # Get top-k closest (smallest distances)
    k = min(max_neighbors, M)
    topk_dists, topk_indices = torch.topk(all_dists_masked, k, dim=-1, largest=False)

    # Pad to max_neighbors if M < max_neighbors
    if k < max_neighbors:
        pad_size = max_neighbors - k
        topk_dists = torch.cat(
            [topk_dists, torch.full((B, N, pad_size), float("inf"))], dim=-1
        )
        topk_indices = torch.cat(
            [topk_indices, torch.full((B, N, pad_size), -1, dtype=torch.long)], dim=-1
        )

    # Set sentinel for entries beyond radius
    sentinel_mask = topk_dists == float("inf")
    topk_indices[sentinel_mask] = -1

    return topk_indices, topk_dists


def multi_scale_ball_query(
    query_points: torch.Tensor,
    reference_points: torch.Tensor,
    scales: list[tuple[float, int]],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """REQ-002-02: Multi-scale ball query at S scales.

    Args:
        query_points: (B, N, 2) query positions.
        reference_points: (B, M, 2) reference positions.
        scales: list of (radius, max_neighbors) tuples.

    Returns:
        List of (indices, dists) tuples, one per scale.
    """
    results = []
    for radius, max_neighbors in scales:
        indices, dists = ball_query_2d(
            query_points, reference_points, radius, max_neighbors
        )
        results.append((indices, dists))
    return results


def gather_ball_query_features(
    features: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    """REQ-002-03: Gather features at ball query neighbor indices.

    Handles -1 sentinel indices by replacing with zeros.

    Args:
        features: (B, M, d_feat) reference point features.
        indices: (B, N, k) neighbor indices, -1 for padding.

    Returns:
        gathered: (B, N, k, d_feat) gathered features (zeros for sentinels).
    """
    B, N, k = indices.shape
    d_feat = features.shape[-1]

    # Replace -1 with 0 for safe gathering
    safe_indices = indices.clamp(min=0)  # (B, N, k)

    # Expand for gathering: (B, N, k) -> (B, N, k, d_feat)
    expanded = safe_indices.unsqueeze(-1).expand(-1, -1, -1, d_feat)

    # Expand features: (B, M, d_feat) -> (B, 1, M, d_feat) for broadcast
    feat_expanded = features.unsqueeze(1).expand(-1, N, -1, -1)

    # Gather
    gathered = torch.gather(feat_expanded, 2, expanded)  # (B, N, k, d_feat)

    # Zero out sentinel positions
    sentinel_mask = (indices == -1).unsqueeze(-1).expand_as(gathered)
    gathered[sentinel_mask] = 0.0

    return gathered
