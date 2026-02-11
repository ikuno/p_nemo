"""Integration tests for the full GeoTransolver 2D pipeline. Article IX.

End-to-end tests verifying all components work together.
"""

import subprocess
import sys
import torch
import pytest

from src.geotransolver.config import (
    CylinderFlowConfig,
    DataConfig,
    BallQueryConfig,
    ModelConfig,
)
from src.geotransolver.data import generate_dataset, CylinderFlowDataset
from src.geotransolver.model import GeoTransolver2D
from src.geotransolver.metrics import mean_absolute_error, relative_l1_norm


class TestFullPipeline:
    """Article IX: Integration tests with real component interactions."""

    def test_full_pipeline_data_to_prediction(self):
        """All REQs: Data generation -> model forward -> metrics all succeed."""
        flow_cfg = CylinderFlowConfig()
        data_cfg = DataConfig(n_samples=5, n_points=32, n_geom=16)
        bq_cfg = BallQueryConfig(scales=[(0.5, 4), (1.0, 8)])
        model_cfg = ModelConfig(
            d_model=16, d_context=16, n_heads=2, n_layers=2, ffn_ratio=2
        )

        # Generate data
        dataset = generate_dataset(data_cfg, flow_cfg)
        train_ds = CylinderFlowDataset(dataset["train"])
        sample = train_ds[0]

        # Model forward
        model = GeoTransolver2D(model_cfg, bq_cfg)
        positions = sample["positions"].unsqueeze(0)
        geom_pos = sample["geometry_positions"].unsqueeze(0)
        geom_normals = sample["geometry_normals"].unsqueeze(0)
        params = sample["global_params"].unsqueeze(0)

        pred = model(positions, positions, geom_pos, geom_normals, params)

        # Compute metrics
        target = torch.cat(
            [sample["velocity"], sample["pressure"]], dim=-1
        ).unsqueeze(0)
        mae = mean_absolute_error(pred, target)
        rel_l1 = relative_l1_norm(pred, target)

        assert pred.shape == (1, 32, 3)
        assert mae.item() >= 0
        assert rel_l1.item() >= 0

    def test_training_reduces_loss(self):
        """REQ-007-02: 10 training steps reduce loss."""
        flow_cfg = CylinderFlowConfig()
        data_cfg = DataConfig(n_samples=5, n_points=32, n_geom=16)
        bq_cfg = BallQueryConfig(scales=[(1.0, 8)])
        model_cfg = ModelConfig(
            d_model=16, d_context=16, n_heads=2, n_layers=2, ffn_ratio=2
        )

        dataset = generate_dataset(data_cfg, flow_cfg)
        train_ds = CylinderFlowDataset(dataset["train"])

        model = GeoTransolver2D(model_cfg, bq_cfg)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()

        losses = []
        for epoch in range(10):
            total_loss = 0.0
            for i in range(len(train_ds)):
                sample = train_ds[i]
                positions = sample["positions"].unsqueeze(0)
                geom_pos = sample["geometry_positions"].unsqueeze(0)
                geom_normals = sample["geometry_normals"].unsqueeze(0)
                params = sample["global_params"].unsqueeze(0)
                target = torch.cat(
                    [sample["velocity"], sample["pressure"]], dim=-1
                ).unsqueeze(0)

                optimizer.zero_grad()
                pred = model(positions, positions, geom_pos, geom_normals, params)
                loss = criterion(pred, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            losses.append(total_loss / len(train_ds))

        # Loss should decrease
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    def test_analytical_solution_self_consistency(self):
        """REQ-001: Generated data satisfies Bernoulli equation."""
        from src.geotransolver.data import (
            generate_domain_points,
            analytical_velocity,
            analytical_pressure,
        )

        R, U, rho, p_inf = 1.0, 1.5, 1.0, 0.0
        pts = generate_domain_points(500, R, R * 5)
        vel = analytical_velocity(pts, R, U)
        prs = analytical_pressure(pts, vel, rho, p_inf, U)

        # Bernoulli: p + 0.5*rho*v^2 = p_inf + 0.5*rho*U^2 = const
        v_mag_sq = (vel**2).sum(dim=-1, keepdim=True)
        bernoulli = prs + 0.5 * rho * v_mag_sq
        expected = p_inf + 0.5 * rho * U**2
        assert torch.allclose(
            bernoulli, torch.full_like(bernoulli, expected), atol=1e-5
        )

    def test_cli_train_runs(self):
        """REQ-007-01: cli.py train --epochs 1 exits successfully."""
        result = subprocess.run(
            [
                sys.executable, "cli.py", "train",
                "--epochs", "2",
                "--n-samples", "5",
                "--n-points", "32",
                "--n-geom", "16",
                "--d-model", "16",
                "--n-heads", "2",
                "--n-layers", "2",
                "--batch-size", "4",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            cwd="/home/ikuno/git/musubix/nvidia_project",
        )
        assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
        assert "Training complete" in result.stdout

    def test_cli_eval_runs(self):
        """REQ-007-01: cli.py eval produces metrics output."""
        import os
        ckpt = "output/checkpoints/best.pt"
        if not os.path.exists(ckpt):
            pytest.skip("No checkpoint found, run train first")

        result = subprocess.run(
            [sys.executable, "cli.py", "eval", "--checkpoint", ckpt],
            capture_output=True,
            text=True,
            timeout=60,
            cwd="/home/ikuno/git/musubix/nvidia_project",
        )
        assert result.returncode == 0, f"CLI eval failed:\n{result.stderr}"
        assert "MAE" in result.stdout

    def test_dataloader_batching(self):
        """Verify DataLoader produces correct batch shapes."""
        from torch.utils.data import DataLoader

        flow_cfg = CylinderFlowConfig()
        data_cfg = DataConfig(n_samples=8, n_points=32, n_geom=16)
        dataset = generate_dataset(data_cfg, flow_cfg)
        train_ds = CylinderFlowDataset(dataset["train"])
        loader = DataLoader(train_ds, batch_size=2)

        batch = next(iter(loader))
        assert batch["positions"].shape == (2, 32, 2)
        assert batch["velocity"].shape == (2, 32, 2)
        assert batch["pressure"].shape == (2, 32, 1)
        assert batch["global_params"].shape == (2, 1)
