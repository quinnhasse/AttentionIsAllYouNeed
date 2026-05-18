"""Shared pytest fixtures for transformer tests."""

import torch
import pytest

from transformer.model import Transformer


@pytest.fixture(scope="session", autouse=True)
def seed_everything() -> None:
    """Fix random seeds for the entire test session."""
    torch.manual_seed(0)


@pytest.fixture
def tiny_transformer() -> Transformer:
    """Minimal Transformer (d_model=32) for fast CPU tests."""
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
    )


@pytest.fixture
def src_batch() -> torch.Tensor:
    """Batch of 2 source sequences, length 8, vocab ids in [1, 49]."""
    return torch.randint(1, 50, (2, 8))


@pytest.fixture
def tgt_batch() -> torch.Tensor:
    """Batch of 2 target sequences, length 6, vocab ids in [1, 49]."""
    return torch.randint(1, 50, (2, 6))
