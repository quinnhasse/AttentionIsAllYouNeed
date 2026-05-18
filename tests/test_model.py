"""Tests for the full encoder-decoder Transformer model."""

import torch
import pytest

from transformer.model import Transformer


@pytest.fixture
def tiny_model() -> Transformer:
    """Small Transformer for fast shape tests."""
    return Transformer(
        src_vocab_size=50,
        tgt_vocab_size=50,
        d_model=32,
        n_heads=4,
        n_layers=2,
        d_ff=64,
        max_len=64,
        dropout=0.0,
        pad_idx=0,
        use_pe=True,
    )


@pytest.fixture
def tiny_model_no_pe() -> Transformer:
    """Transformer with positional encoding disabled."""
    return Transformer(
        src_vocab_size=50,
        tgt_vocab_size=50,
        d_model=32,
        n_heads=4,
        n_layers=2,
        d_ff=64,
        max_len=64,
        dropout=0.0,
        pad_idx=0,
        use_pe=False,
    )


class TestTransformerShapes:
    def test_forward_output_shape(self, tiny_model: Transformer) -> None:
        B, S, T = 3, 8, 6
        src = torch.randint(1, 50, (B, S))
        tgt = torch.randint(1, 50, (B, T))
        logits = tiny_model(src, tgt)
        assert logits.shape == (B, T, 50)

    def test_single_token_target(self, tiny_model: Transformer) -> None:
        src = torch.randint(1, 50, (2, 10))
        tgt = torch.randint(1, 50, (2, 1))
        logits = tiny_model(src, tgt)
        assert logits.shape == (2, 1, 50)

    def test_batch_size_one(self, tiny_model: Transformer) -> None:
        src = torch.randint(1, 50, (1, 5))
        tgt = torch.randint(1, 50, (1, 4))
        logits = tiny_model(src, tgt)
        assert logits.shape == (1, 4, 50)

    def test_encode_output_shape(self, tiny_model: Transformer) -> None:
        B, S = 2, 7
        src = torch.randint(1, 50, (B, S))
        memory = tiny_model.encode(src)
        assert memory.shape == (B, S, 32)

    def test_decode_output_shape(self, tiny_model: Transformer) -> None:
        B, S, T = 2, 7, 5
        src = torch.randint(1, 50, (B, S))
        tgt = torch.randint(1, 50, (B, T))
        memory = tiny_model.encode(src)
        out = tiny_model.decode(tgt, memory)
        assert out.shape == (B, T, 32)


class TestTransformerMasking:
    def test_make_src_mask_shape(self, tiny_model: Transformer) -> None:
        src = torch.tensor([[1, 2, 0, 0]])
        mask = tiny_model.make_src_mask(src)
        assert mask.shape == (1, 1, 1, 4)

    def test_make_tgt_mask_shape(self, tiny_model: Transformer) -> None:
        tgt = torch.randint(1, 10, (2, 6))
        mask = tiny_model.make_tgt_mask(tgt)
        assert mask.shape == (2, 1, 6, 6)

    def test_src_mask_flags_pad_tokens(self, tiny_model: Transformer) -> None:
        src = torch.tensor([[5, 3, 0, 0]])
        mask = tiny_model.make_src_mask(src)
        assert not mask[0, 0, 0, 0].item()
        assert not mask[0, 0, 0, 1].item()
        assert mask[0, 0, 0, 2].item()
        assert mask[0, 0, 0, 3].item()

    def test_tgt_mask_is_causal(self, tiny_model: Transformer) -> None:
        tgt = torch.randint(1, 10, (1, 5))
        mask = tiny_model.make_tgt_mask(tgt)
        # Upper triangle (excluding diagonal) must be True
        for i in range(5):
            for j in range(i + 1, 5):
                assert mask[0, 0, i, j].item(), f"mask[{i},{j}] should be True"


class TestGreedyDecode:
    def test_output_shape(self, tiny_model: Transformer) -> None:
        src = torch.randint(1, 50, (1, 6))
        out = tiny_model.greedy_decode(src, bos_idx=2, eos_idx=3, max_len=10)
        assert out.dim() == 2
        assert out.size(0) == 1
        assert out.size(1) <= 10

    def test_stops_at_eos(self, tiny_model: Transformer) -> None:
        """Output should not include tokens after EOS."""
        torch.manual_seed(0)
        src = torch.randint(1, 50, (1, 5))
        out = tiny_model.greedy_decode(src, bos_idx=2, eos_idx=3, max_len=20)
        ids = out[0].tolist()
        # If EOS is in the output, it must be the last token
        if 3 in ids:
            assert ids[-1] == 3

    def test_no_bos_in_output(self, tiny_model: Transformer) -> None:
        """BOS should be stripped from the returned tensor."""
        src = torch.randint(1, 50, (1, 4))
        out = tiny_model.greedy_decode(src, bos_idx=2, eos_idx=3, max_len=15)
        assert 2 not in out[0].tolist()


class TestNoPEVariant:
    def test_forward_runs_without_pe(self, tiny_model_no_pe: Transformer) -> None:
        src = torch.randint(1, 50, (2, 7))
        tgt = torch.randint(1, 50, (2, 5))
        logits = tiny_model_no_pe(src, tgt)
        assert logits.shape == (2, 5, 50)

    def test_pe_and_no_pe_differ(
        self,
        tiny_model: Transformer,
        tiny_model_no_pe: Transformer,
    ) -> None:
        """With the same weights structure, PE and no-PE should produce different outputs."""
        torch.manual_seed(42)
        src = torch.randint(1, 50, (1, 6))
        tgt = torch.randint(1, 50, (1, 4))
        # Copy weights so only use_pe differs
        tiny_model_no_pe.load_state_dict(tiny_model.state_dict())
        logits_pe   = tiny_model(src, tgt)
        logits_nope = tiny_model_no_pe(src, tgt)
        assert not torch.allclose(logits_pe, logits_nope)


class TestGradientFlow:
    def test_gradients_flow_to_embeddings(self, tiny_model: Transformer) -> None:
        src = torch.randint(1, 50, (2, 5))
        tgt = torch.randint(1, 50, (2, 4))
        tiny_model.train()
        logits = tiny_model(src, tgt)
        logits.sum().backward()
        for name, p in tiny_model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"{name} has no grad"
                assert not torch.isnan(p.grad).any(), f"{name} grad has NaN"
