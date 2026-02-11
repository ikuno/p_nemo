"""Tests for GALE attention block. REQ-004.

TDD Red phase: Tests for the core GALE attention mechanism.
"""

import torch
import pytest

from src.geotransolver.gale_attention import GALEBlock, AdaptiveGate


@pytest.fixture
def gale_block():
    """Small GALE block for testing."""
    return GALEBlock(d_model=16, n_heads=2, d_context=16, ffn_ratio=2)


class TestGALEBlock:
    """REQ-004-01 through REQ-004-04: GALE attention block."""

    def test_self_attention_output_shape(self, gale_block):
        """REQ-004-01: SA output shape matches input (B, N, d_model)."""
        H = torch.randn(2, 10, 16)
        C = torch.randn(2, 16)
        out = gale_block(H, C)
        assert out.shape == (2, 10, 16)

    def test_cross_attention_output_shape(self, gale_block):
        """REQ-004-02: Block processes context correctly."""
        H = torch.randn(1, 8, 16)
        C = torch.randn(1, 16)
        out = gale_block(H, C)
        assert out.shape == (1, 8, 16)

    def test_adaptive_gate_range(self, gale_block):
        """REQ-004-03: alpha in (0, 1) strictly."""
        H = torch.randn(2, 10, 16)
        C = torch.randn(2, 16)
        # Access internal gate value
        gale_block.eval()
        with torch.no_grad():
            alpha = gale_block.compute_gate(H, C)
        assert torch.all(alpha > 0.0)
        assert torch.all(alpha < 1.0)

    def test_gale_block_output_shape(self, gale_block):
        """REQ-004-01-04: Output shape (B, N, d_model)."""
        H = torch.randn(3, 20, 16)
        C = torch.randn(3, 16)
        out = gale_block(H, C)
        assert out.shape == (3, 20, 16)

    def test_gale_block_residual(self, gale_block):
        """REQ-004-04: Output differs from input (FFN+residual changes values)."""
        H = torch.randn(1, 5, 16)
        C = torch.randn(1, 16)
        gale_block.eval()
        with torch.no_grad():
            out = gale_block(H, C)
        assert not torch.allclose(out, H)

    def test_gale_block_gradient_flows(self, gale_block):
        """REQ-004-01-04: Gradients nonzero on all weight parameters."""
        H = torch.randn(1, 8, 16, requires_grad=True)
        C = torch.randn(1, 16)
        out = gale_block(H, C)
        loss = out.sum()
        loss.backward()
        for name, param in gale_block.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                # At least some gradients should be nonzero
                assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"

    def test_alpha_varies_with_context(self, gale_block):
        """REQ-004-03: Different C vectors produce different alpha."""
        H = torch.randn(1, 8, 16)
        C1 = torch.randn(1, 16)
        C2 = torch.randn(1, 16) * 5.0  # Very different context
        gale_block.eval()
        with torch.no_grad():
            a1 = gale_block.compute_gate(H, C1)
            a2 = gale_block.compute_gate(H, C2)
        assert not torch.allclose(a1, a2)

    def test_gale_block_deterministic(self, gale_block):
        """Same input -> same output in eval mode."""
        H = torch.randn(1, 5, 16)
        C = torch.randn(1, 16)
        gale_block.eval()
        with torch.no_grad():
            o1 = gale_block(H, C)
            o2 = gale_block(H, C)
        assert torch.allclose(o1, o2)

    def test_gale_block_different_batch_sizes(self, gale_block):
        """Block handles various batch sizes."""
        for B in [1, 4, 8]:
            H = torch.randn(B, 10, 16)
            C = torch.randn(B, 16)
            out = gale_block(H, C)
            assert out.shape == (B, 10, 16)

    def test_gale_block_single_token(self, gale_block):
        """Block handles single-token sequences."""
        H = torch.randn(1, 1, 16)
        C = torch.randn(1, 16)
        out = gale_block(H, C)
        assert out.shape == (1, 1, 16)


class TestAdaptiveGate:
    """REQ-004-03: Adaptive gating mechanism."""

    def test_gate_output_range(self):
        """Gate output in (0, 1)."""
        gate = AdaptiveGate(d_model=16, d_context=16)
        sa_pool = torch.randn(2, 16)
        c_pool = torch.randn(2, 16)
        alpha = gate(sa_pool, c_pool)
        assert torch.all(alpha > 0.0)
        assert torch.all(alpha < 1.0)

    def test_gate_output_shape(self):
        """Gate outputs scalar per batch element."""
        gate = AdaptiveGate(d_model=16, d_context=16)
        sa_pool = torch.randn(3, 16)
        c_pool = torch.randn(3, 16)
        alpha = gate(sa_pool, c_pool)
        assert alpha.shape == (3, 1)
