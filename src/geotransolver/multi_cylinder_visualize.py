"""Visualization utilities for multi-cylinder rectangular domain.

REQ-MC-003: Contour plots and quiver plots for 3-cylinder flow predictions.
Mirrors visualize.py but adapts to a rectangular domain with multiple cylinders.
"""

import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from src.geotransolver.config import CylinderSpec


def _add_cylinders(ax, cylinders: list[CylinderSpec]) -> None:
    """Draw all cylinders as filled circles on the given axes."""
    for cyl in cylinders:
        ax.add_patch(
            Circle(
                (cyl.x, cyl.y), cyl.R,
                fill=True, color="gray", alpha=0.6, zorder=3,
            )
        )


def plot_multi_cylinder_field_comparison(
    positions: torch.Tensor,
    pred_field: torch.Tensor,
    true_field: torch.Tensor,
    field_name: str,
    output_path: str,
    cylinders: list[CylinderSpec],
    domain: tuple[float, float, float, float],
) -> None:
    """REQ-MC-003-01: Side-by-side scatter plot of predicted vs. analytical fields.

    Produces a 3-panel figure: True | Predicted | |Error|.

    Args:
        positions:   (N, 2) point coordinates.
        pred_field:  (N,) predicted scalar field.
        true_field:  (N,) ground truth scalar field.
        field_name:  Label for the colorbar.
        output_path: File path to save the PNG.
        cylinders:   List of CylinderSpec for overlay.
        domain:      (x_min, x_max, y_min, y_max) for axis limits.
    """
    pos  = positions.detach().cpu().numpy()
    pred = pred_field.detach().cpu().numpy()
    true = true_field.detach().cpu().numpy()
    err  = np.abs(pred - true)

    x_min, x_max, y_min, y_max = domain
    vmin = min(true.min(), pred.min())
    vmax = max(true.max(), pred.max())

    fig, axes = plt.subplots(1, 3, figsize=(21, 5))

    for ax, data, title, use_shared in zip(
        axes,
        [true, pred, err],
        [f"True {field_name}", f"Predicted {field_name}", f"|Error| {field_name}"],
        [True, True, False],
    ):
        kw = {"vmin": vmin, "vmax": vmax} if use_shared else {}
        sc = ax.scatter(pos[:, 0], pos[:, 1], c=data, s=2, cmap="RdBu_r", **kw)
        _add_cylinders(ax, cylinders)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(sc, ax=ax)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cases_grid(
    cases: list[dict],
    sizes: list[float],
    velocities: list[float],
    output_path: str,
) -> None:
    """REQ-MC-005: 3x3 grid summary of analytical velocity magnitude for all 9 cases.

    Args:
        cases: List of 9 dicts (row-major: outer=sizes, inner=velocities), each with:
            'positions': (N, 2) Tensor
            'vel_mag':   (N,) Tensor — velocity magnitude
            'cylinders': list[CylinderSpec]
            'domain':    (x_min, x_max, y_min, y_max)
        sizes:      3 cylinder radii (row labels).
        velocities: 3 freestream velocities (column labels).
        output_path: File path to save PNG.
    """
    n_rows = len(sizes)
    n_cols = len(velocities)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows))

    for idx, case in enumerate(cases):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row][col]

        pos = case["positions"].detach().cpu().numpy()
        mag = case["vel_mag"].detach().cpu().numpy()
        x_min, x_max, y_min, y_max = case["domain"]

        sc = ax.scatter(pos[:, 0], pos[:, 1], c=mag, s=2, cmap="viridis")
        _add_cylinders(ax, case["cylinders"])
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")
        ax.set_title(f"R={sizes[row]:.1f},  U={velocities[col]:.1f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(sc, ax=ax, label="|V|")

    plt.suptitle("Analytical Velocity Magnitude — 3 Sizes × 3 Velocities", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cases_scalar_grid(
    cases: list[dict],
    field_key: str,
    cmap: str,
    suptitle: str,
    cbar_label: str,
    sizes: list[float],
    velocities: list[float],
    output_path: str,
) -> None:
    """REQ-MC-005-02: Generic 3x3 scalar field grid for all 9 cases.

    Args:
        cases:      List of 9 dicts with keys 'positions', field_key, 'cylinders', 'domain'.
        field_key:  Key into each case dict that holds the (N,) scalar tensor.
        cmap:       Matplotlib colormap name.
        suptitle:   Figure super-title.
        cbar_label: Colorbar label.
        sizes:      3 cylinder radii (row labels).
        velocities: 3 freestream velocities (column labels).
        output_path: File path to save PNG.
    """
    n_rows = len(sizes)
    n_cols = len(velocities)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows))

    for idx, case in enumerate(cases):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row][col]

        pos   = case["positions"].detach().cpu().numpy()
        field = case[field_key].detach().cpu().numpy()
        x_min, x_max, y_min, y_max = case["domain"]

        sc = ax.scatter(pos[:, 0], pos[:, 1], c=field, s=2, cmap=cmap)
        _add_cylinders(ax, case["cylinders"])
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")
        ax.set_title(f"R={sizes[row]:.1f},  U={velocities[col]:.1f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(sc, ax=ax, label=cbar_label)

    plt.suptitle(suptitle, fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cases_quiver_grid(
    cases: list[dict],
    sizes: list[float],
    velocities: list[float],
    output_path: str,
    scale: float = 50.0,
) -> None:
    """REQ-MC-005-03: 3x3 quiver grid of analytical velocity vectors for all 9 cases.

    Args:
        cases:      List of 9 dicts with keys 'positions', 'velocity', 'cylinders', 'domain'.
        sizes:      3 cylinder radii (row labels).
        velocities: 3 freestream velocities (column labels).
        output_path: File path to save PNG.
        scale:      Quiver scale (larger = shorter arrows).
    """
    n_rows = len(sizes)
    n_cols = len(velocities)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows))

    for idx, case in enumerate(cases):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row][col]

        pos = case["positions"].detach().cpu().numpy()
        vel = case["velocity"].detach().cpu().numpy()
        mag = np.linalg.norm(vel, axis=1)
        x_min, x_max, y_min, y_max = case["domain"]

        q = ax.quiver(
            pos[:, 0], pos[:, 1],
            vel[:, 0], vel[:, 1],
            mag, cmap="coolwarm", scale=scale, width=0.003,
        )
        _add_cylinders(ax, case["cylinders"])
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")
        ax.set_title(f"R={sizes[row]:.1f},  U={velocities[col]:.1f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(q, ax=ax, label="|V|")

    plt.suptitle("Analytical Velocity Vectors — 3 Sizes × 3 Velocities", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_multi_cylinder_velocity_vectors(
    positions: torch.Tensor,
    pred_velocity: torch.Tensor,
    true_velocity: torch.Tensor,
    output_path: str,
    cylinders: list[CylinderSpec],
    domain: tuple[float, float, float, float],
    scale: float = 50.0,
) -> None:
    """REQ-MC-003-02: Quiver plot of predicted vs. analytical velocity vectors.

    Produces a 2-panel figure: True | Predicted.

    Args:
        positions:     (N, 2) point coordinates.
        pred_velocity: (N, 2) predicted [v_x, v_y].
        true_velocity: (N, 2) ground truth [v_x, v_y].
        output_path:   File path to save the PNG.
        cylinders:     List of CylinderSpec for overlay.
        domain:        (x_min, x_max, y_min, y_max) for axis limits.
        scale:         Quiver scale (larger = shorter arrows).
    """
    pos  = positions.detach().cpu().numpy()
    pred = pred_velocity.detach().cpu().numpy()
    true = true_velocity.detach().cpu().numpy()

    pred_mag = np.linalg.norm(pred, axis=1)
    true_mag = np.linalg.norm(true, axis=1)
    vmin = 0.0
    vmax = max(true_mag.max(), pred_mag.max())

    x_min, x_max, y_min, y_max = domain

    fig, axes = plt.subplots(1, 2, figsize=(18, 5))

    for ax, vel, mag, title in zip(
        axes,
        [true, pred],
        [true_mag, pred_mag],
        ["True Velocity", "Predicted Velocity"],
    ):
        q = ax.quiver(
            pos[:, 0], pos[:, 1],
            vel[:, 0], vel[:, 1],
            mag, cmap="coolwarm", scale=scale,
            clim=(vmin, vmax), width=0.002,
        )
        _add_cylinders(ax, cylinders)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(q, ax=ax, label="|V|")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
