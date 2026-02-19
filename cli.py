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


def setup_device(gpus_str: str | None) -> tuple[torch.device, int]:
    """Configure CUDA device(s) for training/inference.

    Args:
        gpus_str: Comma-separated GPU IDs (e.g. "0,1,2,3"), or None for auto.

    Returns:
        (device, n_gpus) — device to use and number of available GPUs.
    """
    if gpus_str is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus_str
    n_gpus = torch.cuda.device_count()
    device = torch.device("cuda" if n_gpus > 0 else "cpu")
    return device, n_gpus


def _unwrap_state_dict(state_dict: dict) -> dict:
    """Strip 'module.' prefix from DataParallel checkpoint keys."""
    if all(k.startswith("module.") for k in state_dict):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict

from src.geotransolver.config import (
    CylinderFlowConfig,
    DataConfig,
    BallQueryConfig,
    ModelConfig,
    TrainingConfig,
    MultiCylinderFlowConfig,
    RectDataConfig,
    CylinderSpec,
)
from src.geotransolver.data import generate_dataset, CylinderFlowDataset
from src.geotransolver.model import GeoTransolver2D
from src.geotransolver.metrics import mean_absolute_error, relative_l1_norm, r_squared
from src.geotransolver.visualize import (
    plot_field_comparison,
    plot_training_curves,
    plot_velocity_vectors,
)
from src.geotransolver.multi_cylinder_data import (
    generate_multi_cylinder_dataset,
    MultiCylinderFlowDataset,
)
from src.geotransolver.multi_cylinder_visualize import (
    plot_multi_cylinder_field_comparison,
    plot_multi_cylinder_velocity_vectors,
    plot_cases_grid,
    plot_cases_scalar_grid,
    plot_cases_quiver_grid,
)


def cmd_train(args):
    """REQ-007-02, REQ-007-03: Train the model."""
    # Device setup — use all visible GPUs (or subset via --gpus)
    device, n_gpus = setup_device(getattr(args, "gpus", None))
    gpu_info = f"{n_gpus} GPU(s)" if n_gpus > 0 else "CPU"
    print(f"Device: {device} [{gpu_info}]")

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
    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=train_cfg.batch_size, shuffle=True,
        num_workers=train_cfg.num_workers, pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers, pin_memory=pin,
    )

    model = GeoTransolver2D(model_cfg, bq_cfg).to(device)
    if n_gpus > 1:
        model = nn.DataParallel(model)
        print(f"DataParallel enabled across {n_gpus} GPUs")
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
            positions    = batch["positions"].to(device)
            geom_pos     = batch["geometry_positions"].to(device)
            geom_normals = batch["geometry_normals"].to(device)
            params       = batch["global_params"].to(device)
            target = torch.cat(
                [batch["velocity"].to(device), batch["pressure"].to(device)], dim=-1
            )

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
                positions    = batch["positions"].to(device)
                geom_pos     = batch["geometry_positions"].to(device)
                geom_normals = batch["geometry_normals"].to(device)
                params       = batch["global_params"].to(device)
                target = torch.cat(
                    [batch["velocity"].to(device), batch["pressure"].to(device)], dim=-1
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
            raw_model = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": raw_model.state_dict(),
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
    device, n_gpus = setup_device(getattr(args, "gpus", None))
    print(f"Device: {device} [{n_gpus} GPU(s) if n_gpus > 0 else 'CPU']")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_cfg = checkpoint["model_config"]
    bq_cfg = checkpoint["bq_config"]

    model = GeoTransolver2D(model_cfg, bq_cfg).to(device)
    model.load_state_dict(_unwrap_state_dict(checkpoint["model_state_dict"]))
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
            positions    = batch["positions"].to(device)
            geom_pos     = batch["geometry_positions"].to(device)
            geom_normals = batch["geometry_normals"].to(device)
            params       = batch["global_params"].to(device)
            target = torch.cat(
                [batch["velocity"].to(device), batch["pressure"].to(device)], dim=-1
            )

            pred = model(positions, positions, geom_pos, geom_normals, params)
            all_preds.append(pred.cpu())
            all_targets.append(target.cpu())

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
    device, _ = setup_device(getattr(args, "gpus", None))

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_cfg = checkpoint["model_config"]
    bq_cfg = checkpoint["bq_config"]

    model = GeoTransolver2D(model_cfg, bq_cfg).to(device)
    model.load_state_dict(_unwrap_state_dict(checkpoint["model_state_dict"]))
    model.eval()

    flow_cfg = CylinderFlowConfig()
    data_cfg = DataConfig(n_samples=10, n_points=1024, n_geom=64)
    dataset = generate_dataset(data_cfg, flow_cfg)

    os.makedirs(args.output, exist_ok=True)

    # Visualize first test sample
    test_data = dataset["test"]
    idx = 0
    positions    = test_data["positions"][idx : idx + 1].to(device)
    geom_pos     = test_data["geometry_positions"].unsqueeze(0).to(device)
    geom_normals = test_data["geometry_normals"].unsqueeze(0).to(device)
    params       = test_data["global_params"][idx : idx + 1].to(device)

    with torch.no_grad():
        pred = model(positions, positions, geom_pos, geom_normals, params)
    pred = pred.cpu()
    positions = positions.cpu()

    target = torch.cat(
        [test_data["velocity"][idx], test_data["pressure"][idx]], dim=-1
    )
    pts = positions[0]

    # Velocity vectors (quiver plot)
    pred_vel = pred[0, :, :2]
    true_vel = target[:, :2]
    plot_velocity_vectors(
        pts, pred_vel, true_vel,
        os.path.join(args.output, "velocity_vectors.png"),
    )

    # Velocity magnitude (scalar comparison)
    pred_vel_mag = pred_vel.norm(dim=-1)
    true_vel_mag = true_vel.norm(dim=-1)
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


# ---------------------------------------------------------------------------
# Multi-cylinder commands (REQ-MC-004)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Parameter sweep constants (REQ-MC-005)
# ---------------------------------------------------------------------------

CASE_SIZES: list[float] = [0.5, 1.0, 1.5]       # Cylinder radii
CASE_VELOCITIES: list[float] = [0.5, 1.0, 2.0]  # Freestream velocities


def cmd_generate_cases(args):
    """REQ-MC-005-01: Generate 9 datasets (3 sizes × 3 velocities) and save to disk.

    For each combination of cylinder radius R ∈ {0.5, 1.0, 1.5} and freestream
    velocity U ∈ {0.5, 1.0, 2.0}, this command:
      1. Builds a MultiCylinderFlowConfig with three equal-size cylinders.
      2. Generates n_samples domain samples all at fixed U.
      3. Saves the dataset as a .pt file under output/cases/.
      4. Emits a 3×3 summary grid plot.

    Cylinder centres are fixed at x = -4, 0, +4  (y = 0).
    Domain is [-8,8] × [-4,4] for all cases.
    """
    from src.geotransolver.multi_cylinder_data import (
        generate_multi_cylinder_dataset,
        generate_rect_domain_points,
        analytical_multi_cylinder_velocity,
    )

    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)

    data_cfg = RectDataConfig(
        n_samples=args.n_samples,
        n_points=args.n_points,
        n_geom=args.n_geom,
    )

    grid_cases: list[dict] = []  # for 3x3 summary plot

    total = len(CASE_SIZES) * len(CASE_VELOCITIES)
    done = 0
    for R in CASE_SIZES:
        for U in CASE_VELOCITIES:
            done += 1
            tag = f"R{R:.1f}_U{U:.1f}"
            print(f"[{done}/{total}] Generating case {tag}  "
                  f"(R={R}, U={U}, n_samples={args.n_samples}) ...")

            # Three identical cylinders at x = -4, 0, +4
            cylinders = [
                CylinderSpec(x=-4.0, y=0.0, R=R),
                CylinderSpec(x=0.0,  y=0.0, R=R),
                CylinderSpec(x=4.0,  y=0.0, R=R),
            ]
            flow_cfg = MultiCylinderFlowConfig(
                cylinders=cylinders,
                domain=(-8.0, 8.0, -4.0, 4.0),
                rho=1.0,
                p_inf=0.0,
                U_range=(U, U),  # fixed velocity for this case
            )

            dataset = generate_multi_cylinder_dataset(data_cfg, flow_cfg, seed=42)

            save_path = os.path.join(out_dir, f"case_{tag}.pt")
            torch.save(
                {
                    "R": R,
                    "U": U,
                    "flow_config": flow_cfg,
                    "data_config": data_cfg,
                    "train": dataset["train"],
                    "val":   dataset["val"],
                    "test":  dataset["test"],
                },
                save_path,
            )
            print(f"  -> Saved: {save_path}")

            # Collect first test sample for summary grids
            test_positions = dataset["test"]["positions"][0]   # (n_points, 2)
            test_velocity  = dataset["test"]["velocity"][0]    # (n_points, 2)
            test_pressure  = dataset["test"]["pressure"][0, :, 0]  # (n_points,)
            vel_mag = test_velocity.norm(dim=-1)               # (n_points,)
            grid_cases.append({
                "positions": test_positions,
                "vel_mag":   vel_mag,
                "pressure":  test_pressure,
                "velocity":  test_velocity,
                "cylinders": cylinders,
                "domain":    flow_cfg.domain,
            })

    # --- 3x3 summary plots ---
    # Velocity magnitude
    vel_path = os.path.join(out_dir, "cases_grid_vel_mag.png")
    plot_cases_scalar_grid(
        grid_cases, "vel_mag", "viridis",
        "Analytical Velocity Magnitude — 3 Sizes × 3 Velocities",
        "|V|", CASE_SIZES, CASE_VELOCITIES, vel_path,
    )
    print(f"Velocity magnitude grid saved to: {vel_path}")

    # Pressure
    p_path = os.path.join(out_dir, "cases_grid_pressure.png")
    plot_cases_scalar_grid(
        grid_cases, "pressure", "RdBu_r",
        "Analytical Pressure — 3 Sizes × 3 Velocities",
        "p", CASE_SIZES, CASE_VELOCITIES, p_path,
    )
    print(f"Pressure grid saved to:           {p_path}")

    # Velocity vectors
    q_path = os.path.join(out_dir, "cases_grid_vectors.png")
    plot_cases_quiver_grid(grid_cases, CASE_SIZES, CASE_VELOCITIES, q_path)
    print(f"Velocity vector grid saved to:    {q_path}")

    # Legacy vel-mag grid (kept for backward compat)
    plot_cases_grid(grid_cases, CASE_SIZES, CASE_VELOCITIES,
                    os.path.join(out_dir, "cases_grid.png"))

    print(f"\nAll 9 cases generated.")
    print(f"Datasets saved in: {out_dir}/")


def cmd_train_mc(args):
    """REQ-MC-004-01: Train GeoTransolver on 3-cylinder rectangular domain."""
    device, n_gpus = setup_device(getattr(args, "gpus", None))
    gpu_info = f"{n_gpus} GPU(s)" if n_gpus > 0 else "CPU"
    print(f"Device: {device} [{gpu_info}]")

    flow_cfg = MultiCylinderFlowConfig()
    data_cfg = RectDataConfig(
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
        checkpoint_dir="output/checkpoints",
    )

    print(f"Generating multi-cylinder dataset: {data_cfg.n_samples} samples, "
          f"{data_cfg.n_points} domain points, {data_cfg.n_geom} geom points...")
    dataset = generate_multi_cylinder_dataset(data_cfg, flow_cfg)
    train_ds = MultiCylinderFlowDataset(dataset["train"])
    val_ds   = MultiCylinderFlowDataset(dataset["val"])
    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=train_cfg.batch_size, shuffle=True,
        num_workers=train_cfg.num_workers, pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers, pin_memory=pin,
    )

    model = GeoTransolver2D(model_cfg, bq_cfg).to(device)
    if n_gpus > 1:
        model = nn.DataParallel(model)
        print(f"DataParallel enabled across {n_gpus} GPUs")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

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
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            positions    = batch["positions"].to(device)
            geom_pos     = batch["geometry_positions"].to(device)
            geom_normals = batch["geometry_normals"].to(device)
            params       = batch["global_params"].to(device)
            target = torch.cat(
                [batch["velocity"].to(device), batch["pressure"].to(device)], dim=-1
            )
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

        model.eval()
        val_mae_total = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                positions    = batch["positions"].to(device)
                geom_pos     = batch["geometry_positions"].to(device)
                geom_normals = batch["geometry_normals"].to(device)
                params       = batch["global_params"].to(device)
                target = torch.cat(
                    [batch["velocity"].to(device), batch["pressure"].to(device)], dim=-1
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
            raw_model = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_mae": val_mae,
                    "model_config": model_cfg,
                    "bq_config": bq_cfg,
                    "flow_config": flow_cfg,
                },
                os.path.join(train_cfg.checkpoint_dir, "multi_cylinder_best.pt"),
            )

    print(f"\nTraining complete. Best val MAE: {best_val_loss:.6f}")
    plot_training_curves(
        train_losses,
        {"MAE": val_maes},
        os.path.join(train_cfg.checkpoint_dir, "multi_cylinder_training_curves.png"),
    )
    print(f"Training curves saved to "
          f"{train_cfg.checkpoint_dir}/multi_cylinder_training_curves.png")


def cmd_eval_mc(args):
    """REQ-MC-004-02: Evaluate multi-cylinder model on test set."""
    device, n_gpus = setup_device(getattr(args, "gpus", None))
    print(f"Device: {device} [{n_gpus} GPU(s) if n_gpus > 0 else 'CPU']")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_cfg = checkpoint["model_config"]
    bq_cfg    = checkpoint["bq_config"]
    flow_cfg  = checkpoint.get("flow_config", MultiCylinderFlowConfig())

    model = GeoTransolver2D(model_cfg, bq_cfg).to(device)
    model.load_state_dict(_unwrap_state_dict(checkpoint["model_state_dict"]))
    model.eval()

    data_cfg = RectDataConfig(n_samples=50, n_points=512, n_geom=192)
    dataset  = generate_multi_cylinder_dataset(data_cfg, flow_cfg)
    test_ds  = MultiCylinderFlowDataset(dataset["test"])
    test_loader = DataLoader(test_ds, batch_size=8)

    all_preds   = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            positions    = batch["positions"].to(device)
            geom_pos     = batch["geometry_positions"].to(device)
            geom_normals = batch["geometry_normals"].to(device)
            params       = batch["global_params"].to(device)
            target = torch.cat(
                [batch["velocity"].to(device), batch["pressure"].to(device)], dim=-1
            )
            pred = model(positions, positions, geom_pos, geom_normals, params)
            all_preds.append(pred.cpu())
            all_targets.append(target.cpu())

    preds   = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    mae    = mean_absolute_error(preds, targets).item()
    rel_l1 = relative_l1_norm(preds, targets).item()

    pred_vel_mag = preds[..., :2].norm(dim=-1).flatten()
    true_vel_mag = targets[..., :2].norm(dim=-1).flatten()
    r2_vel = r_squared(pred_vel_mag, true_vel_mag).item()
    r2_p   = r_squared(preds[..., 2].flatten(), targets[..., 2].flatten()).item()

    print("=== Multi-Cylinder Evaluation Results ===")
    print(f"MAE:           {mae:.6f}")
    print(f"Relative L1:   {rel_l1:.6f}")
    print(f"R² (velocity): {r2_vel:.6f}")
    print(f"R² (pressure): {r2_p:.6f}")


def cmd_visualize_mc(args):
    """REQ-MC-004-03: Generate visualization plots for multi-cylinder results."""
    device, _ = setup_device(getattr(args, "gpus", None))

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_cfg = checkpoint["model_config"]
    bq_cfg    = checkpoint["bq_config"]
    flow_cfg  = checkpoint.get("flow_config", MultiCylinderFlowConfig())

    model = GeoTransolver2D(model_cfg, bq_cfg).to(device)
    model.load_state_dict(_unwrap_state_dict(checkpoint["model_state_dict"]))
    model.eval()

    data_cfg = RectDataConfig(n_samples=10, n_points=1024, n_geom=192)
    dataset  = generate_multi_cylinder_dataset(data_cfg, flow_cfg)

    os.makedirs(args.output, exist_ok=True)

    test_data    = dataset["test"]
    idx          = 0
    positions    = test_data["positions"][idx : idx + 1].to(device)
    geom_pos     = test_data["geometry_positions"].unsqueeze(0).to(device)
    geom_normals = test_data["geometry_normals"].unsqueeze(0).to(device)
    params       = test_data["global_params"][idx : idx + 1].to(device)

    with torch.no_grad():
        pred = model(positions, positions, geom_pos, geom_normals, params)
    pred      = pred.cpu()
    positions = positions.cpu()

    target = torch.cat(
        [test_data["velocity"][idx], test_data["pressure"][idx]], dim=-1
    )
    pts = positions[0]

    # Velocity vectors
    pred_vel = pred[0, :, :2]
    true_vel = target[:, :2]
    plot_multi_cylinder_velocity_vectors(
        pts, pred_vel, true_vel,
        os.path.join(args.output, "mc_velocity_vectors.png"),
        cylinders=flow_cfg.cylinders,
        domain=flow_cfg.domain,
    )

    # Velocity magnitude
    plot_multi_cylinder_field_comparison(
        pts,
        pred_vel.norm(dim=-1),
        true_vel.norm(dim=-1),
        "Velocity Magnitude",
        os.path.join(args.output, "mc_velocity.png"),
        cylinders=flow_cfg.cylinders,
        domain=flow_cfg.domain,
    )

    # Pressure
    plot_multi_cylinder_field_comparison(
        pts,
        pred[0, :, 2],
        target[:, 2],
        "Pressure",
        os.path.join(args.output, "mc_pressure.png"),
        cylinders=flow_cfg.cylinders,
        domain=flow_cfg.domain,
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
    train_parser.add_argument(
        "--gpus", type=str, default=None,
        help="Comma-separated GPU IDs to use (e.g. '0,1,2,3'). Default: all available.",
    )

    # Eval
    eval_parser = subparsers.add_parser("eval", help="Evaluate the model")
    eval_parser.add_argument("--checkpoint", type=str, required=True)
    eval_parser.add_argument(
        "--gpus", type=str, default=None,
        help="Comma-separated GPU IDs to use (e.g. '0'). Default: all available.",
    )

    # Visualize
    viz_parser = subparsers.add_parser("visualize", help="Generate plots")
    viz_parser.add_argument("--checkpoint", type=str, required=True)
    viz_parser.add_argument("--output", type=str, default="output/plots")
    viz_parser.add_argument(
        "--gpus", type=str, default=None,
        help="Comma-separated GPU IDs to use (e.g. '0'). Default: all available.",
    )

    # --- Parameter sweep (9 cases) ---
    gen_cases = subparsers.add_parser(
        "generate-cases",
        help="Generate 9 datasets: 3 cylinder sizes × 3 flow velocities",
    )
    gen_cases.add_argument(
        "--output", type=str, default="output/cases",
        help="Directory to save .pt files and summary plot.",
    )
    gen_cases.add_argument("--n-samples", type=int, default=100,
                           help="Samples per case (default: 100).")
    gen_cases.add_argument("--n-points",  type=int, default=512,
                           help="Domain points per sample (default: 512).")
    gen_cases.add_argument("--n-geom",    type=int, default=192,
                           help="Geometry boundary points (default: 192).")

    # --- Multi-cylinder subcommands ---
    # train-mc
    train_mc = subparsers.add_parser(
        "train-mc", help="Train on 3-cylinder rectangular domain"
    )
    train_mc.add_argument("--epochs",       type=int,   default=200)
    train_mc.add_argument("--lr",           type=float, default=1e-3)
    train_mc.add_argument("--batch-size",   type=int,   default=16)
    train_mc.add_argument("--n-samples",    type=int,   default=100)
    train_mc.add_argument("--n-points",     type=int,   default=512)
    train_mc.add_argument("--n-geom",       type=int,   default=192)
    train_mc.add_argument("--d-model",      type=int,   default=64)
    train_mc.add_argument("--n-heads",      type=int,   default=4)
    train_mc.add_argument("--n-layers",     type=int,   default=4)
    train_mc.add_argument("--log-interval", type=int,   default=5)
    train_mc.add_argument(
        "--gpus", type=str, default=None,
        help="Comma-separated GPU IDs (e.g. '0,1,2,3'). Default: all available.",
    )

    # eval-mc
    eval_mc = subparsers.add_parser("eval-mc", help="Evaluate multi-cylinder model")
    eval_mc.add_argument("--checkpoint", type=str, required=True)
    eval_mc.add_argument(
        "--gpus", type=str, default=None,
        help="Comma-separated GPU IDs. Default: all available.",
    )

    # visualize-mc
    viz_mc = subparsers.add_parser("visualize-mc", help="Generate multi-cylinder plots")
    viz_mc.add_argument("--checkpoint", type=str, required=True)
    viz_mc.add_argument("--output", type=str, default="output/plots/multi_cylinder")
    viz_mc.add_argument(
        "--gpus", type=str, default=None,
        help="Comma-separated GPU IDs. Default: all available.",
    )

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "visualize":
        cmd_visualize(args)
    elif args.command == "generate-cases":
        cmd_generate_cases(args)
    elif args.command == "train-mc":
        cmd_train_mc(args)
    elif args.command == "eval-mc":
        cmd_eval_mc(args)
    elif args.command == "visualize-mc":
        cmd_visualize_mc(args)


if __name__ == "__main__":
    main()
