"""Visualization utilities for 2D GeoTransolver.

REQ-007-04: Contour plots comparing predicted and analytical fields.
"""

import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def plot_field_comparison(
    positions: torch.Tensor,
    pred_field: torch.Tensor,
    true_field: torch.Tensor,
    field_name: str,
    output_path: str,
    cylinder_R: float = 1.0,
):
    """REQ-007-04: Side-by-side contour plot of predicted vs. analytical fields.

    Args:
        positions: (N, 2) point positions.
        pred_field: (N,) predicted scalar field.
        true_field: (N,) ground truth scalar field.
        field_name: name for the colorbar label.
        output_path: path to save the figure.
        cylinder_R: cylinder radius for overlay.
    """
    pos = positions.detach().cpu().numpy()
    pred = pred_field.detach().cpu().numpy()
    true = true_field.detach().cpu().numpy()
    error = np.abs(pred - true)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, data, title in zip(
        axes,
        [true, pred, error],
        [f"True {field_name}", f"Predicted {field_name}", f"|Error| {field_name}"],
    ):
        sc = ax.scatter(pos[:, 0], pos[:, 1], c=data, s=3, cmap="RdBu_r")
        ax.add_patch(Circle((0, 0), cylinder_R, fill=True, color="gray", alpha=0.5))
        ax.set_aspect("equal")
        ax.set_title(title)
        plt.colorbar(sc, ax=ax)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_velocity_vectors(
    positions: torch.Tensor,
    pred_velocity: torch.Tensor,
    true_velocity: torch.Tensor,
    output_path: str,
    cylinder_R: float = 1.0,
    scale: float = 30.0,
):
    """REQ-007-04: Quiver plot comparing predicted vs. analytical velocity vectors.

    Args:
        positions: (N, 2) point positions.
        pred_velocity: (N, 2) predicted velocity (vx, vy).
        true_velocity: (N, 2) ground truth velocity (vx, vy).
        output_path: path to save the figure.
        cylinder_R: cylinder radius for overlay.
        scale: quiver scale factor (larger = shorter arrows).
    """
    pos = positions.detach().cpu().numpy()
    pred = pred_velocity.detach().cpu().numpy()
    true = true_velocity.detach().cpu().numpy()

    pred_mag = np.linalg.norm(pred, axis=1)
    true_mag = np.linalg.norm(true, axis=1)
    vmin = 0.0
    vmax = max(true_mag.max(), pred_mag.max())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

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
            clim=(vmin, vmax), width=0.003,
        )
        ax.add_patch(Circle((0, 0), cylinder_R, fill=True, color="gray", alpha=0.6))
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(q, ax=ax, label="|V|")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(
    train_losses: list[float],
    val_metrics: dict[str, list[float]],
    output_path: str,
):
    """Plot training loss and validation metric curves.

    Args:
        train_losses: list of training losses per epoch.
        val_metrics: dict of metric_name -> list of values per epoch.
        output_path: path to save the figure.
    """
    n_plots = 1 + len(val_metrics)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    axes[0].plot(train_losses)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].set_yscale("log")

    for ax, (name, values) in zip(axes[1:], val_metrics.items()):
        ax.plot(values)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(name)
        ax.set_title(f"Validation {name}")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
