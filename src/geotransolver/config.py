"""Configuration dataclasses for GeoTransolver 2D.

REQ-001 through REQ-007: Centralized configuration for all components.
"""

from dataclasses import dataclass, field


@dataclass
class CylinderFlowConfig:
    """REQ-001: Cylinder flow physical parameters."""

    R: float = 1.0  # Cylinder radius
    R_far: float = 5.0  # Far-field radius
    U_inf: float = 1.0  # Freestream velocity
    rho: float = 1.0  # Fluid density
    p_inf: float = 0.0  # Freestream pressure


@dataclass
class DataConfig:
    """REQ-001-05, REQ-001-06: Dataset parameters."""

    n_samples: int = 100  # Number of samples (varying U_inf)
    n_points: int = 512  # Points per sample in domain
    n_geom: int = 128  # Geometry boundary points
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    U_range: tuple[float, float] = (0.5, 2.0)  # Freestream velocity range


@dataclass
class BallQueryConfig:
    """REQ-002: Multi-scale ball query parameters."""

    scales: list[tuple[float, int]] = field(
        default_factory=lambda: [(0.5, 8), (1.0, 16), (2.0, 32)]
    )


@dataclass
class ModelConfig:
    """REQ-004, REQ-005: GeoTransolver architecture."""

    d_model: int = 64  # Hidden dimension
    d_context: int = 64  # Context vector dimension
    n_heads: int = 4  # Attention heads
    n_layers: int = 4  # Number of GALE blocks
    n_slices: int = 2  # Physics slices (velocity, pressure)
    d_input: int = 2  # Input features (x, y position)
    d_geom: int = 2  # Geometry features (nx, ny normal)
    d_global: int = 1  # Global params (U_inf)
    d_output: int = 3  # Output fields (v_x, v_y, p)
    ffn_ratio: int = 4  # FFN hidden dim multiplier
    dropout: float = 0.0


@dataclass
class TrainingConfig:
    """REQ-007: Training parameters."""

    epochs: int = 200
    lr: float = 1e-3
    batch_size: int = 16
    weight_decay: float = 1e-4
    scheduler_step: int = 50
    scheduler_gamma: float = 0.5
    log_interval: int = 10
    checkpoint_dir: str = "output/checkpoints"
