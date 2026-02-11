"""GeoTransolver 2D model — complete assembly of all components.

REQ-005: Full 2D GeoTransolver model for cylinder flow prediction.

Architecture (paper arXiv: 2512.20399v2, Figure 1):
1. Encoder: projects input features to latent space
2. Context Projector: builds shared geometry context C (computed once)
3. Input augmentation: multi-scale ball query features appended
4. L x GALE blocks: self-attention + cross-attention with shared C
5. Output heads: per-slice LayerNorm + MLP -> field predictions
"""

import torch
import torch.nn as nn

from src.geotransolver.config import ModelConfig, BallQueryConfig
from src.geotransolver.context_projector import ContextProjector
from src.geotransolver.gale_attention import GALEBlock


class GeoTransolver2D(nn.Module):
    """REQ-005: Complete 2D GeoTransolver model.

    Two slices for the 2D cylinder problem:
    - Slice 0: velocity field (v_x, v_y) -> 2 outputs
    - Slice 1: pressure field (p) -> 1 output
    """

    def __init__(self, config: ModelConfig, bq_config: BallQueryConfig):
        super().__init__()
        self.config = config
        self.bq_config = bq_config

        # REQ-005-01, REQ-005-02: Context projector (also computes augmentation)
        self.context_projector = ContextProjector(config, bq_config)
        d_aug = self.context_projector.d_aug

        # REQ-005-01: Encoder — projects [input_features, augmentation] to d_model
        self.encoder = nn.Linear(config.d_input + d_aug, config.d_model)

        # Learned slicer: assigns tokens to physics slices via soft weights
        self.slicer = nn.Linear(config.d_model, config.n_slices)

        # REQ-005-03: Stack of L GALE blocks (one set per slice)
        self.gale_blocks = nn.ModuleList(
            [
                GALEBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    d_context=config.d_context,
                    ffn_ratio=config.ffn_ratio,
                    dropout=config.dropout,
                )
                for _ in range(config.n_layers)
            ]
        )

        # REQ-005-04: Output heads per slice
        # Slice 0: velocity (2 outputs), Slice 1: pressure (1 output)
        self.output_norms = nn.ModuleList(
            [nn.LayerNorm(config.d_model) for _ in range(config.n_slices)]
        )
        d_out_per_slice = [2, 1]  # velocity (2D), pressure (1D)
        self.output_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(config.d_model, config.d_model),
                    nn.GELU(),
                    nn.Linear(config.d_model, d_out),
                )
                for d_out in d_out_per_slice
            ]
        )

    def forward(
        self,
        positions: torch.Tensor,
        features: torch.Tensor,
        geometry_positions: torch.Tensor,
        geometry_features: torch.Tensor,
        global_params: torch.Tensor,
    ) -> torch.Tensor:
        """REQ-005-05, REQ-005-06: Forward pass.

        Args:
            positions: (B, N, 2) input point positions.
            features: (B, N, d_input) input features.
            geometry_positions: (B, M_g, 2) or (M_g, 2) boundary positions.
            geometry_features: (B, M_g, d_geom) or (M_g, d_geom) boundary normals.
            global_params: (B, d_global) global parameters.

        Returns:
            predictions: (B, N, d_output) predicted [v_x, v_y, p].
        """
        B, N, _ = positions.shape

        # REQ-005-01, REQ-005-02: Compute context and augmentation (once)
        context, augmentation = self.context_projector(
            global_params, geometry_positions, geometry_features,
            positions, features,
        )

        # Encode: concatenate features + augmentation, project to d_model
        encoder_input = torch.cat([features, augmentation], dim=-1)
        H = self.encoder(encoder_input)  # (B, N, d_model)

        # Compute slice weights (soft assignment)
        slice_weights = torch.softmax(
            self.slicer(H), dim=-1
        )  # (B, N, n_slices)

        # REQ-005-03: Apply GALE blocks with shared context
        for gale_block in self.gale_blocks:
            H = gale_block(H, context)

        # REQ-005-04: Per-slice output heads
        outputs = []
        for s in range(self.config.n_slices):
            w_s = slice_weights[..., s : s + 1]  # (B, N, 1)
            H_s = H * w_s  # Weighted hidden states for slice s
            H_s = self.output_norms[s](H_s)
            out_s = self.output_heads[s](H_s)  # (B, N, d_out_s)
            outputs.append(out_s)

        # Concatenate slice outputs: [v_x, v_y] + [p] = (B, N, 3)
        predictions = torch.cat(outputs, dim=-1)
        return predictions
