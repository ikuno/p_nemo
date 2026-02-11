"""Context projector module for GeoTransolver.

REQ-003: Builds context vector C from geometry, global parameters,
and multi-scale ball query embeddings.

C = [p_enc, c_geom, E_1, ..., E_S] projected to d_context.

Based on GeoTransolver paper (arXiv: 2512.20399v2), Equations 4-8.
"""

import torch
import torch.nn as nn

from src.geotransolver.config import ModelConfig, BallQueryConfig
from src.geotransolver.ball_query import (
    ball_query_2d,
    multi_scale_ball_query,
    gather_ball_query_features,
)


class ContextProjector(nn.Module):
    """REQ-003: Build context vector and input augmentation.

    Components:
    - Global param encoder: MLP(d_global -> d_enc)
    - Geometry encoder rho: MLP(d_geom -> d_enc), mean pool -> c_geom
    - Per-scale phi_s (input-to-geom): processes [feature, relative_pos] -> E_s
    - Per-scale psi_s (geom-to-input): processes [geom_feat, relative_pos] -> augmentation
    - Final projection: Linear(d_raw_context -> d_context)
    """

    def __init__(self, config: ModelConfig, bq_config: BallQueryConfig):
        super().__init__()
        self.config = config
        self.scales = bq_config.scales
        d_enc = config.d_context // 4  # Encoding dimension per component

        # REQ-003-01: Global parameter encoder
        self.global_encoder = nn.Sequential(
            nn.Linear(config.d_global, d_enc),
            nn.GELU(),
            nn.Linear(d_enc, d_enc),
        )

        # REQ-003-02: Geometry encoder rho
        self.geom_encoder = nn.Sequential(
            nn.Linear(config.d_geom, d_enc),
            nn.GELU(),
            nn.Linear(d_enc, d_enc),
        )

        # REQ-003-03: Per-scale input-to-geometry MLP phi_s
        # Input: [input_feature, relative_position] -> d_enc
        self.phi_mlps = nn.ModuleList()
        for _ in self.scales:
            self.phi_mlps.append(
                nn.Sequential(
                    nn.Linear(config.d_input + 2, d_enc),  # +2 for relative pos
                    nn.GELU(),
                    nn.Linear(d_enc, d_enc),
                )
            )

        # REQ-003-04: Per-scale geometry-to-input MLP psi_s
        # Input: [geom_feature, relative_position] -> d_aug_per_scale
        d_aug_per_scale = d_enc
        self.psi_mlps = nn.ModuleList()
        for _ in self.scales:
            self.psi_mlps.append(
                nn.Sequential(
                    nn.Linear(config.d_geom + 2, d_aug_per_scale),  # +2 for relative pos
                    nn.GELU(),
                    nn.Linear(d_aug_per_scale, d_aug_per_scale),
                )
            )

        # Final projection: [p_enc, c_geom, E_1, ..., E_S] -> d_context
        raw_dim = d_enc + d_enc + len(self.scales) * d_enc
        self.context_proj = nn.Linear(raw_dim, config.d_context)

        self.d_aug = len(self.scales) * d_aug_per_scale

    def forward(
        self,
        global_params: torch.Tensor,
        geometry_positions: torch.Tensor,
        geometry_features: torch.Tensor,
        input_positions: torch.Tensor,
        input_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute context vector and input augmentation.

        Args:
            global_params: (B, d_global) global parameters.
            geometry_positions: (B, M_g, 2) or (M_g, 2) geometry point positions.
            geometry_features: (B, M_g, d_geom) or (M_g, d_geom) geometry normals.
            input_positions: (B, N, 2) input point positions.
            input_features: (B, N, d_input) input point features.

        Returns:
            context: (B, d_context) context vector C.
            augmentation: (B, N, d_aug) multi-scale augmentation for inputs.
        """
        B = global_params.shape[0]

        # Handle unbatched geometry
        if geometry_positions.dim() == 2:
            geometry_positions = geometry_positions.unsqueeze(0).expand(B, -1, -1)
            geometry_features = geometry_features.unsqueeze(0).expand(B, -1, -1)

        # REQ-003-01: Encode global parameters
        p_enc = self.global_encoder(global_params)  # (B, d_enc)

        # REQ-003-02: Encode geometry and pool
        geom_encoded = self.geom_encoder(geometry_features)  # (B, M_g, d_enc)
        c_geom = geom_encoded.mean(dim=1)  # (B, d_enc) permutation-invariant

        # REQ-003-03: Multi-scale input-to-geometry embeddings E_s
        # Ball query: for each geometry point, find nearby input points
        inp_to_geom_results = multi_scale_ball_query(
            geometry_positions, input_positions, self.scales
        )

        E_list = []
        for s, ((indices, dists), phi) in enumerate(
            zip(inp_to_geom_results, self.phi_mlps)
        ):
            # Gather input features at neighbor indices
            gathered_feats = gather_ball_query_features(
                input_features, indices
            )  # (B, M_g, k_s, d_input)

            # Compute relative positions
            gathered_pos = gather_ball_query_features(
                input_positions, indices
            )  # (B, M_g, k_s, 2)
            rel_pos = gathered_pos - geometry_positions.unsqueeze(2)

            # Concatenate [feature, relative_pos]
            phi_input = torch.cat([gathered_feats, rel_pos], dim=-1)

            # Apply MLP and pool over neighbors
            h = phi(phi_input)  # (B, M_g, k_s, d_enc)

            # Masked mean pool (ignore sentinel positions)
            valid_mask = (indices != -1).unsqueeze(-1).float()  # (B, M_g, k_s, 1)
            h = (h * valid_mask).sum(dim=2) / valid_mask.sum(dim=2).clamp(min=1)

            # Pool over geometry points
            E_s = h.mean(dim=1)  # (B, d_enc)
            E_list.append(E_s)

        # REQ-003-04: Multi-scale geometry-to-input augmentation
        geom_to_inp_results = multi_scale_ball_query(
            input_positions, geometry_positions, self.scales
        )

        aug_list = []
        for s, ((indices, dists), psi) in enumerate(
            zip(geom_to_inp_results, self.psi_mlps)
        ):
            # Gather geometry features at neighbor indices
            gathered_feats = gather_ball_query_features(
                geometry_features, indices
            )  # (B, N, k_s, d_geom)

            # Compute relative positions
            gathered_pos = gather_ball_query_features(
                geometry_positions, indices
            )  # (B, N, k_s, 2)
            rel_pos = gathered_pos - input_positions.unsqueeze(2)

            # Concatenate [geom_feature, relative_pos]
            psi_input = torch.cat([gathered_feats, rel_pos], dim=-1)

            # Apply MLP and pool over neighbors
            h = psi(psi_input)  # (B, N, k_s, d_aug_per_scale)

            # Masked mean pool
            valid_mask = (indices != -1).unsqueeze(-1).float()
            h = (h * valid_mask).sum(dim=2) / valid_mask.sum(dim=2).clamp(min=1)
            aug_list.append(h)  # (B, N, d_aug_per_scale)

        augmentation = torch.cat(aug_list, dim=-1)  # (B, N, d_aug)

        # Assemble and project context
        raw_context = torch.cat([p_enc, c_geom] + E_list, dim=-1)
        context = self.context_proj(raw_context)  # (B, d_context)

        return context, augmentation
