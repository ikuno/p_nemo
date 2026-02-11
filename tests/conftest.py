"""Shared pytest fixtures for GeoTransolver 2D tests."""

import torch
import pytest

from src.geotransolver.config import (
    CylinderFlowConfig,
    DataConfig,
    BallQueryConfig,
    ModelConfig,
    TrainingConfig,
)


@pytest.fixture(autouse=True)
def set_seed():
    """Ensure deterministic tests."""
    torch.manual_seed(42)


@pytest.fixture
def flow_config():
    """Standard cylinder flow configuration."""
    return CylinderFlowConfig()


@pytest.fixture
def data_config():
    """Small data config for fast tests."""
    return DataConfig(n_samples=10, n_points=64, n_geom=32)


@pytest.fixture
def bq_config():
    """Ball query config with 3 scales."""
    return BallQueryConfig(scales=[(0.5, 8), (1.0, 16), (2.0, 32)])


@pytest.fixture
def small_model_config():
    """Tiny model config for fast tests."""
    return ModelConfig(
        d_model=16,
        d_context=16,
        n_heads=2,
        n_layers=2,
        n_slices=2,
        d_input=2,
        d_geom=2,
        d_global=1,
        d_output=3,
        ffn_ratio=2,
        dropout=0.0,
    )


@pytest.fixture
def sample_points_2d():
    """(1, 32, 2) random 2D points in annular domain R=1..5."""
    r = torch.rand(1, 32) * 4.0 + 1.0  # [1, 5]
    theta = torch.rand(1, 32) * 2 * torch.pi
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    return torch.stack([x, y], dim=-1)  # (1, 32, 2)


@pytest.fixture
def sample_geometry_2d():
    """(1, 16, 2) circle boundary points + (1, 16, 2) normals."""
    theta = torch.linspace(0, 2 * torch.pi, 17)[:-1].unsqueeze(0)  # (1, 16)
    x = torch.cos(theta)
    y = torch.sin(theta)
    positions = torch.stack([x, y], dim=-1)  # (1, 16, 2)
    normals = positions.clone()  # Outward normals = position on unit circle
    return positions, normals


@pytest.fixture
def sample_global_params():
    """(1, 1) global parameter [U_inf=1.0]."""
    return torch.tensor([[1.0]])
