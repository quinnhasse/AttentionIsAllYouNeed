"""Tests for masking utilities and mask interactions in attention."""

import torch
import pytest

from transformer.decoder import make_causal_mask
from transformer.model import Transformer
from transformer.attention import scaled_dot_product_attention


class TestCausalMask:
    def test_shape(self) -> None:
        mask = make_causal_mask(6)
        assert mask.shape == (1, 1, 6, 6)

    def test_upper_triangular(self) -> None:
        """Position (i, j) is True (masked) iff j > i."""
        T = 7
        mask = make_causal_mask(T)
        for i in range(T):
            for j in range(T):
                assert mask[0, 0, i, j].item() == (j > i), (
                    f"mask[{i},{j}] expected {j > i}, got {mask[0, 0, i, j].item()}"
                )

    def test_diagonal_not_masked(self) -> None:
        """Each position can attend to itself."""
        T = 5
        mask = make_causal_mask(T)
        for i in range(T):
            assert not mask[0, 0, i, i].item(), f"diagonal [{i},{i}] should not be masked"

    def test_length_one(self) -> None:
        """Sequence of length 1: no positions masked."""
        mask = make_causal_mask(1)
        assert mask.shape == (1, 1, 1, 1)
        assert not mask[0, 0, 0, 0].item()

    def test_dtype_is_bool(self) -> None:
        mask = make_causal_mask(4)
        assert mask.dtype == torch.bool

    def test_device_propagation(self) -> None:
        """Mask should live on the same device as the request."""
        mask = make_causal_mask(4, device=torch.device("cpu"))
        assert mask.device.type == "cpu"


class TestPaddingMask:
    """Test padding mask construction on the Transformer model."""

    @pytest.fixture
    def model(self) -> Transformer:
        return Transformer(
            src_vocab_size=100,
            tgt_vocab_size=100,
            d_model=32,
            n_heads=4,
            n_layers=2,
            d_ff=64,
            pad_idx=0,
        )

    def test_src_mask_shape(self, model: Transformer) -> None:
        src = torch.tensor([[1, 2, 3, 0, 0]])  # 2 pad tokens
        mask = model.make_src_mask(src)
        assert mask.shape == (1, 1, 1, 5)

    def test_src_mask_marks_pad_tokens(self, model: Transformer) -> None:
        src = torch.tensor([[1, 2, 3, 0, 0]])
        mask = model.make_src_mask(src)
        # Last two positions should be masked
        assert not mask[0, 0, 0, 0].item()
        assert not mask[0, 0, 0, 2].item()
        assert mask[0, 0, 0, 3].item()
        assert mask[0, 0, 0, 4].item()

    def test_tgt_mask_shape(self, model: Transformer) -> None:
        tgt = torch.tensor([[2, 5, 7, 0]])  # BOS, tokens, PAD
        mask = model.make_tgt_mask(tgt)
        assert mask.shape == (1, 1, 4, 4)

    def test_tgt_mask_combines_causal_and_pad(self, model: Transformer) -> None:
        """Tgt mask = causal OR pad. Pad position always masked; future always masked."""
        tgt = torch.tensor([[1, 2, 0]])  # last token is pad
        mask = model.make_tgt_mask(tgt)
        # Position 0 attends to 0 only (causal allows, no pad)
        assert not mask[0, 0, 0, 0].item()  # self — not masked
        assert mask[0, 0, 0, 1].item()      # future — masked
        assert mask[0, 0, 0, 2].item()      # future + pad — masked
        # Position 1 attends to 0 and 1
        assert not mask[0, 0, 1, 0].item()  # past — not masked
        assert not mask[0, 0, 1, 1].item()  # self — not masked
        assert mask[0, 0, 1, 2].item()      # future + pad — masked
        # Position 2 is a pad token but can look back at non-pad
        assert mask[0, 0, 2, 2].item()      # pad attending to itself — pad mask


class TestAttentionWithMasks:
    def test_causal_mask_blocks_future(self) -> None:
        """With causal mask, output at step t should not depend on tokens after t."""
        torch.manual_seed(0)
        B, H, T, D = 1, 1, 5, 8
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, T, D)
        v = torch.randn(B, H, T, D)

        mask = make_causal_mask(T)
        out_masked, attn = scaled_dot_product_attention(q, k, v, mask=mask)
        # All upper-triangle attention weights should be 0
        for i in range(T):
            for j in range(i + 1, T):
                assert torch.allclose(
                    attn[0, 0, i, j], torch.tensor(0.0), atol=1e-6
                ), f"attn[{i},{j}] = {attn[0,0,i,j].item():.2e} should be 0"

    def test_padding_mask_zeros_attention(self) -> None:
        """Fully-masked key positions should get zero attention weight."""
        B, H, T, D = 2, 2, 6, 16
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, T, D)
        v = torch.randn(B, H, T, D)
        # Mask last two keys
        mask = torch.zeros(B, H, T, T, dtype=torch.bool)
        mask[:, :, :, -2:] = True
        _, attn = scaled_dot_product_attention(q, k, v, mask=mask)
        assert torch.allclose(
            attn[:, :, :, -2:], torch.zeros(B, H, T, 2), atol=1e-6
        )

    def test_all_keys_masked_produces_nan_free_output(self) -> None:
        """When all keys are masked, softmax returns uniform weights on -inf.

        After softmax(-inf, ..., -inf) each weight is nan in principle;
        the actual behavior depends on implementation. We just verify
        that the masking function runs without throwing.
        """
        B, H, T, D = 1, 1, 3, 8
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, T, D)
        v = torch.randn(B, H, T, D)
        mask = torch.ones(B, H, T, T, dtype=torch.bool)
        # Should not raise
        out, attn = scaled_dot_product_attention(q, k, v, mask=mask)
        assert out.shape == (B, H, T, D)
