"""GALE (Geometry Aware Latent Embeddings) attention block.

REQ-004: Core attention mechanism of GeoTransolver.
Based on paper (arXiv: 2512.20399v2), Equations 10-13.

Each GALE block performs:
1. Slice-wise Self-Attention (SA)
2. Cross-Attention (CA) to shared geometry context
3. Adaptive gate blending: H = (1-α)SA + αCA
4. FFN with residual connection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveGate(nn.Module):
    """REQ-004-03: Adaptive gate η for blending SA and CA.

    α = σ(η(Pool(SA), Pool(C))) ∈ (0, 1)
    """

    def __init__(self, d_model: int, d_context: int):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(d_model + d_context, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self, sa_pool: torch.Tensor, c_pool: torch.Tensor
    ) -> torch.Tensor:
        """Compute adaptive gate value.

        Args:
            sa_pool: (B, d_model) pooled self-attention output.
            c_pool: (B, d_context) pooled context vector.

        Returns:
            alpha: (B, 1) gate value in (0, 1).
        """
        combined = torch.cat([sa_pool, c_pool], dim=-1)
        return torch.sigmoid(self.gate_net(combined))


class GALEBlock(nn.Module):
    """REQ-004: Single GALE attention block.

    Architecture per block:
        SA_m = Attn(H W_Q, H W_K, H W_V)           -- self-attention
        CA_m = Attn(H W_Qc, C W_Kc, C W_Vc)        -- cross-attention
        α = σ(η(Pool(SA), Pool(C)))                  -- adaptive gate
        H = (1-α)SA + αCA                            -- blend
        H = H + MLP(LayerNorm(H))                    -- FFN + residual
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_context: int,
        ffn_ratio: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_context = d_context
        self.d_head = d_model // n_heads
        assert d_model % n_heads == 0

        # REQ-004-01: Self-attention projections
        self.sa_norm = nn.LayerNorm(d_model)
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.sa_out = nn.Linear(d_model, d_model)

        # REQ-004-02: Cross-attention projections
        self.W_Qc = nn.Linear(d_model, d_model)
        self.W_Kc = nn.Linear(d_context, d_model)
        self.W_Vc = nn.Linear(d_context, d_model)
        self.ca_out = nn.Linear(d_model, d_model)

        # REQ-004-03: Adaptive gate
        self.gate = AdaptiveGate(d_model, d_context)

        # REQ-004-04: FFN with residual
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_ratio),
            nn.GELU(),
            nn.Linear(d_model * ffn_ratio, d_model),
        )

        self.dropout = nn.Dropout(dropout)

    def _multi_head_attention(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
    ) -> torch.Tensor:
        """REQ-004-05: Scaled dot-product attention with multi-head.

        Args:
            Q: (B, N_q, d_model)
            K: (B, N_kv, d_model)
            V: (B, N_kv, d_model)

        Returns:
            output: (B, N_q, d_model)
        """
        B, N_q, _ = Q.shape
        N_kv = K.shape[1]

        # Reshape to multi-head: (B, n_heads, N, d_head)
        Q = Q.view(B, N_q, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B, N_kv, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, N_kv, self.n_heads, self.d_head).transpose(1, 2)

        # Use PyTorch's optimized SDPA (Article VIII)
        out = F.scaled_dot_product_attention(Q, K, V)

        # Reshape back: (B, N_q, d_model)
        out = out.transpose(1, 2).contiguous().view(B, N_q, self.d_model)
        return out

    def compute_gate(
        self, H: torch.Tensor, C: torch.Tensor
    ) -> torch.Tensor:
        """Compute adaptive gate value for testing.

        Args:
            H: (B, N, d_model) hidden states.
            C: (B, d_context) context vector.

        Returns:
            alpha: (B, 1) gate value.
        """
        H_normed = self.sa_norm(H)

        # Self-attention
        Q = self.W_Q(H_normed)
        K = self.W_K(H_normed)
        V = self.W_V(H_normed)
        sa = self.sa_out(self._multi_head_attention(Q, K, V))

        # Pool SA output and context
        sa_pool = sa.mean(dim=1)  # (B, d_model)
        return self.gate(sa_pool, C)

    def forward(
        self, H: torch.Tensor, C: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through GALE block.

        Args:
            H: (B, N, d_model) hidden states.
            C: (B, d_context) context vector.

        Returns:
            H_out: (B, N, d_model) updated hidden states.
        """
        # Pre-norm
        H_normed = self.sa_norm(H)

        # REQ-004-01: Self-attention
        Q = self.W_Q(H_normed)
        K = self.W_K(H_normed)
        V = self.W_V(H_normed)
        sa = self.sa_out(self._multi_head_attention(Q, K, V))

        # REQ-004-02: Cross-attention
        # Context as single-token KV sequence: (B, 1, d_context)
        C_seq = C.unsqueeze(1)
        Qc = self.W_Qc(H_normed)
        Kc = self.W_Kc(C_seq)
        Vc = self.W_Vc(C_seq)
        ca = self.ca_out(self._multi_head_attention(Qc, Kc, Vc))

        # REQ-004-03: Adaptive gate and blend
        sa_pool = sa.mean(dim=1)  # (B, d_model)
        alpha = self.gate(sa_pool, C)  # (B, 1)
        alpha = alpha.unsqueeze(1)  # (B, 1, 1) for broadcasting over N
        blended = (1 - alpha) * sa + alpha * ca

        # Residual connection after attention
        H = H + self.dropout(blended)

        # REQ-004-04: FFN with residual
        H = H + self.dropout(self.ffn(self.ffn_norm(H)))

        return H
