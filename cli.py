"""CLI entry point for 2D GeoTransolver (Article II: CLI Interface Mandate).

REQ-007-01: Training, evaluation, and visualization.

Usage:
    python cli.py train [--epochs N] [--lr RATE] [--batch-size B]
    python cli.py eval --checkpoint PATH
    python cli.py visualize --checkpoint PATH [--output DIR]
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.geotransolver.config import (
    CylinderFlowConfig,
    DataConfig,
    BallQueryConfig,
    ModelConfig,
    TrainingConfig,
)
from src.geotransolver.data import generate_dataset, CylinderFlowDataset
from src.geotransolver.model import GeoTransolver2D
from src.geotransolver.metrics import mean_absolute_error, relative_l1_norm, r_squared
from src.geotransolver.visualize import plot_field_comparison, plot_training_curves


def cmd_train(args):
    """REQ-007-02, REQ-007-03: Train the model."""
    flow_cfg = CylinderFlowConfig()
    data_cfg = DataConfig(
        n_samples=args.n_samples,
        n_points=args.n_points,
        n_geom=args.n_geom,
    )
    bq_cfg = BallQueryConfig()
    model_cfg = ModelConfig(
        d_model=args.d_model,
        d_context=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
    )
    train_cfg = TrainingConfig(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        log_interval=args.log_interval,
    )

    print(f"Generating dataset: {data_cfg.n_samples} samples, "
          f"{data_cfg.n_points} points each...")
    dataset = generate_dataset(data_cfg, flow_cfg)
    train_ds = CylinderFlowDataset(dataset["train"])
    val_ds = CylinderFlowDataset(dataset["val"])
    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=train_cfg.batch_size)

    model = GeoTransolver2D(model_cfg, bq_cfg)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=train_cfg.scheduler_step, gamma=train_cfg.scheduler_gamma
    )
    criterion = nn.MSELoss()

    os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")
    train_losses = []
    val_maes = []

    for epoch in range(train_cfg.epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            positions = batch["positions"]
            geom_pos = batch["geometry_positions"]
            geom_normals = batch["geometry_normals"]
            params = batch["global_params"]
            target_vel = batch["velocity"]
            target_prs = batch["pressure"]
            target = torch.cat([target_vel, target_prs], dim=-1)

            optimizer.zero_grad()
            pred = model(positions, positions, geom_pos, geom_normals, params)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_loss)

        # Validation
        model.eval()
        val_mae_total = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                positions = batch["positions"]
                geom_pos = batch["geometry_positions"]
                geom_normals = batch["geometry_normals"]
                params = batch["global_params"]
                target = torch.cat(
                    [batch["velocity"], batch["pressure"]], dim=-1
                )
                pred = model(positions, positions, geom_pos, geom_normals, params)
                val_mae_total += mean_absolute_error(pred, target).item()
                val_batches += 1

        val_mae = val_mae_total / max(val_batches, 1)
        val_maes.append(val_mae)

        if (epoch + 1) % train_cfg.log_interval == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:4d}/{train_cfg.epochs} | "
                f"Loss: {avg_loss:.6f} | Val MAE: {val_mae:.6f} | "
                f"LR: {scheduler.get_last_lr()[0]:.2e}"
            )

        if val_mae < best_val_loss:
            best_val_loss = val_mae
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_mae": val_mae,
                    "model_config": model_cfg,
                    "bq_config": bq_cfg,
                },
                os.path.join(train_cfg.checkpoint_dir, "best.pt"),
            )

    print(f"\nTraining complete. Best val MAE: {best_val_loss:.6f}")

    # Save training curves
    plot_training_curves(
        train_losses,
        {"MAE": val_maes},
        os.path.join(train_cfg.checkpoint_dir, "training_curves.png"),
    )
    print(f"Training curves saved to {train_cfg.checkpoint_dir}/training_curves.png")


def cmd_eval(args):
    """REQ-006: Evaluate model on test set."""
    checkpoint = torch.load(args.checkpoint, weights_only=False)
    model_cfg = checkpoint["model_config"]
    bq_cfg = checkpoint["bq_config"]

    model = GeoTransolver2D(model_cfg, bq_cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    flow_cfg = CylinderFlowConfig()
    data_cfg = DataConfig(n_samples=50, n_points=256, n_geom=64)
    dataset = generate_dataset(data_cfg, flow_cfg)
    test_ds = CylinderFlowDataset(dataset["test"])
    test_loader = DataLoader(test_ds, batch_size=8)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            positions = batch["positions"]
            geom_pos = batch["geometry_positions"]
            geom_normals = batch["geometry_normals"]
            params = batch["global_params"]
            target = torch.cat([batch["velocity"], batch["pressure"]], dim=-1)

            pred = model(positions, positions, geom_pos, geom_normals, params)
            all_preds.append(pred)
            all_targets.append(target)

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    mae = mean_absolute_error(preds, targets).item()
    rel_l1 = relative_l1_norm(preds, targets).item()

    # R² for velocity magnitude
    pred_vel_mag = preds[..., :2].norm(dim=-1).flatten()
    true_vel_mag = targets[..., :2].norm(dim=-1).flatten()
    r2_vel = r_squared(pred_vel_mag, true_vel_mag).item()

    # R² for pressure
    r2_p = r_squared(preds[..., 2].flatten(), targets[..., 2].flatten()).item()

    print("=== Evaluation Results ===")
    print(f"MAE:           {mae:.6f}")
    print(f"Relative L1:   {rel_l1:.6f}")
    print(f"R² (velocity): {r2_vel:.6f}")
    print(f"R² (pressure): {r2_p:.6f}")


def cmd_visualize(args):
    """REQ-007-04: Generate visualization plots."""
    checkpoint = torch.load(args.checkpoint, weights_only=False)
    model_cfg = checkpoint["model_config"]
    bq_cfg = checkpoint["bq_config"]

    model = GeoTransolver2D(model_cfg, bq_cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    flow_cfg = CylinderFlowConfig()
    data_cfg = DataConfig(n_samples=10, n_points=1024, n_geom=64)
    dataset = generate_dataset(data_cfg, flow_cfg)

    os.makedirs(args.output, exist_ok=True)

    # Visualize first test sample
    test_data = dataset["test"]
    idx = 0
    positions = test_data["positions"][idx : idx + 1]
    geom_pos = test_data["geometry_positions"].unsqueeze(0)
    geom_normals = test_data["geometry_normals"].unsqueeze(0)
    params = test_data["global_params"][idx : idx + 1]

    with torch.no_grad():
        pred = model(positions, positions, geom_pos, geom_normals, params)

    target = torch.cat(
        [test_data["velocity"][idx], test_data["pressure"][idx]], dim=-1
    )
    pts = positions[0]

    # Velocity magnitude
    pred_vel_mag = pred[0, :, :2].norm(dim=-1)
    true_vel_mag = target[:, :2].norm(dim=-1)
    plot_field_comparison(
        pts, pred_vel_mag, true_vel_mag,
        "Velocity Magnitude", os.path.join(args.output, "velocity.png"),
    )

    # Pressure
    plot_field_comparison(
        pts, pred[0, :, 2], target[:, 2],
        "Pressure", os.path.join(args.output, "pressure.png"),
    )

    print(f"Visualizations saved to {args.output}/")


def main():
    parser = argparse.ArgumentParser(
        description="2D GeoTransolver CLI (Article II)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--epochs", type=int, default=200)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--n-samples", type=int, default=100)
    train_parser.add_argument("--n-points", type=int, default=512)
    train_parser.add_argument("--n-geom", type=int, default=128)
    train_parser.add_argument("--d-model", type=int, default=64)
    train_parser.add_argument("--n-heads", type=int, default=4)
    train_parser.add_argument("--n-layers", type=int, default=4)
    train_parser.add_argument("--log-interval", type=int, default=5)

    # Eval
    eval_parser = subparsers.add_parser("eval", help="Evaluate the model")
    eval_parser.add_argument("--checkpoint", type=str, required=True)

    # Visualize
    viz_parser = subparsers.add_parser("visualize", help="Generate plots")
    viz_parser.add_argument("--checkpoint", type=str, required=True)
    viz_parser.add_argument("--output", type=str, default="output/plots")

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "visualize":
        cmd_visualize(args)


if __name__ == "__main__":
    main()
