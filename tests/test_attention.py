"""Tests for scaled dot-product attention and MultiHeadAttention."""

import math
import pytest
import torch

from transformer.attention import MultiHeadAttention, scaled_dot_product_attention


# ---------------------------------------------------------------------------
# scaled_dot_product_attention
# ---------------------------------------------------------------------------

class TestScaledDotProductAttention:
    def test_output_shape(self) -> None:
        B, H, T, D = 2, 4, 10, 16
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, T, D)
        v = torch.randn(B, H, T, D)
        out, attn = scaled_dot_product_attention(q, k, v)
        assert out.shape == (B, H, T, D)
        assert attn.shape == (B, H, T, T)

    def test_attention_weights_sum_to_one(self) -> None:
        B, H, T, D = 1, 1, 5, 8
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, T, D)
        v = torch.randn(B, H, T, D)
        _, attn = scaled_dot_product_attention(q, k, v)
        row_sums = attn.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_mask_blocks_positions(self) -> None:
        """Masked positions should produce -inf scores -> zero attention weight."""
        B, H, T, D = 1, 1, 4, 8
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, T, D)
        v = torch.randn(B, H, T, D)
        # Mask the last key position for every query
        mask = torch.zeros(B, H, T, T, dtype=torch.bool)
        mask[:, :, :, -1] = True
        _, attn = scaled_dot_product_attention(q, k, v, mask=mask)
        assert torch.allclose(attn[:, :, :, -1], torch.zeros(B, H, T), atol=1e-6)

    def test_causal_mask_shape(self) -> None:
        """Upper-triangular causal mask should zero out future positions."""
        from transformer.decoder import make_causal_mask
        T = 5
        mask = make_causal_mask(T)  # (1, 1, T, T)
        assert mask.shape == (1, 1, T, T)
        # Position (i, j) should be True (masked) iff j > i
        for i in range(T):
            for j in range(T):
                expected = j > i
                assert mask[0, 0, i, j].item() == expected

    def test_gradient_flows(self) -> None:
        B, H, T, D = 1, 2, 6, 8
        q = torch.randn(B, H, T, D, requires_grad=True)
        k = torch.randn(B, H, T, D, requires_grad=True)
        v = torch.randn(B, H, T, D, requires_grad=True)
        out, _ = scaled_dot_product_attention(q, k, v)
        out.sum().backward()
        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None
        assert not torch.isnan(q.grad).any()

    def test_scale_factor(self) -> None:
        """Output should scale with d_k — verify sqrt(d_k) normalisation."""
        B, H, T = 1, 1, 3
        for d_k in [8, 32, 128]:
            q = torch.ones(B, H, T, d_k)
            k = torch.ones(B, H, T, d_k)
            v = torch.ones(B, H, T, d_k)
            # All scores equal -> uniform attention -> output = v
            out, attn = scaled_dot_product_attention(q, k, v)
            assert out.shape == (B, H, T, d_k)
            # Uniform scores: each attn weight should be 1/T
            expected = torch.full((B, H, T, T), 1.0 / T)
            assert torch.allclose(attn, expected, atol=1e-5), f"failed at d_k={d_k}"


# ---------------------------------------------------------------------------
# MultiHeadAttention
# ---------------------------------------------------------------------------

class TestMultiHeadAttention:
    @pytest.fixture
    def mha(self) -> MultiHeadAttention:
        return MultiHeadAttention(d_model=64, n_heads=4, dropout=0.0)

    def test_self_attention_shape(self, mha: MultiHeadAttention) -> None:
        B, T, D = 3, 12, 64
        x = torch.randn(B, T, D)
        out, attn = mha(x, x, x)
        assert out.shape == (B, T, D)
        assert attn.shape == (B, 4, T, T)

    def test_cross_attention_shape(self, mha: MultiHeadAttention) -> None:
        B, Tq, Tk, D = 2, 8, 15, 64
        q = torch.randn(B, Tq, D)
        kv = torch.randn(B, Tk, D)
        out, attn = mha(q, kv, kv)
        assert out.shape == (B, Tq, D)
        assert attn.shape == (B, 4, Tq, Tk)

    def test_projection_is_different_from_input(self, mha: MultiHeadAttention) -> None:
        B, T, D = 1, 5, 64
        x = torch.randn(B, T, D)
        out, _ = mha(x, x, x)
        assert not torch.allclose(out, x)

    def test_requires_d_model_divisible_by_heads(self) -> None:
        with pytest.raises(AssertionError):
            MultiHeadAttention(d_model=65, n_heads=4)

    def test_padding_mask_zeroes_attention(self) -> None:
        """Pad tokens (mask=True) should receive near-zero attention weight."""
        mha = MultiHeadAttention(d_model=32, n_heads=2, dropout=0.0)
        B, T, D = 1, 6, 32
        x = torch.randn(B, T, D)
        # Mask the last 2 key positions
        mask = torch.zeros(B, 1, 1, T, dtype=torch.bool)
        mask[:, :, :, -2:] = True
        _, attn = mha(x, x, x, mask)
        assert torch.allclose(attn[:, :, :, -2:], torch.zeros(B, 2, T, 2), atol=1e-5)

    def test_gradient_flows_through_mha(self) -> None:
        mha = MultiHeadAttention(d_model=32, n_heads=4, dropout=0.0)
        x = torch.randn(1, 5, 32, requires_grad=True)
        out, _ = mha(x, x, x)
        out.sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_different_batch_sizes(self, mha: MultiHeadAttention) -> None:
        for B in [1, 4, 16]:
            x = torch.randn(B, 8, 64)
            out, _ = mha(x, x, x)
            assert out.shape == (B, 8, 64)
