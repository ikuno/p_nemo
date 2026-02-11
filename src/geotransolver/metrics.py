"""Benchmarking evaluation metrics.

REQ-006: Metrics from the Benchmarking Framework paper (arXiv: 2507.10747v1).
- Mean Absolute Error (Eq. 15)
- Relative L1 Norm (Eq. 16)
- R-squared for coefficient of determination
- 2D surface force integration (Eq. 17 adapted to 2D)
"""

import torch


def mean_absolute_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """REQ-006-01: MAE = (1/N) * sum(|pred - target|).

    Args:
        pred: predicted values (any shape).
        target: ground truth values (same shape).

    Returns:
        Scalar MAE value.
    """
    return (pred - target).abs().mean()


def relative_l1_norm(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """REQ-006-02: Relative L1 = sum(|pred - target|) / sum(|target|).

    Args:
        pred: predicted values.
        target: ground truth values.

    Returns:
        Scalar relative L1 value.
    """
    return (pred - target).abs().sum() / target.abs().sum().clamp(min=1e-10)


def r_squared(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """REQ-006-03: R² = 1 - SS_res / SS_tot.

    Args:
        pred: predicted values (1D tensor).
        target: ground truth values (1D tensor).

    Returns:
        Scalar R² value.
    """
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    return 1.0 - ss_res / ss_tot.clamp(min=1e-10)


def compute_drag_lift_2d(
    pressure: torch.Tensor,
    normals: torch.Tensor,
    ds: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """REQ-006-04: 2D surface force integration.

    F = -integral_S (p * n_hat) dS
    drag = F_x (streamwise), lift = F_y (normal to flow)

    Args:
        pressure: (N, 1) surface pressure values.
        normals: (N, 2) outward unit normals.
        ds: (N,) arc length element for each surface point.

    Returns:
        drag: scalar drag force.
        lift: scalar lift force.
    """
    # F = -sum(p * n * ds) over surface points
    force = -(pressure * normals * ds.unsqueeze(-1)).sum(dim=0)  # (2,)
    drag = force[0]
    lift = force[1]
    return drag, lift
