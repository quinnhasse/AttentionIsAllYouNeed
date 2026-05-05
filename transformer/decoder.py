"""Transformer decoder with masked self-attention and cross-attention."""

import torch
import torch.nn as nn
from torch import Tensor

from transformer.attention import MultiHeadAttention
from transformer.encoder import FeedForward


def make_causal_mask(seq_len: int, device: torch.device | None = None) -> Tensor:
    """Build an upper-triangular causal mask for autoregressive decoding.

    Returns a boolean tensor of shape (1, 1, seq_len, seq_len) where
    True means the position is masked (future positions are hidden).
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)


class DecoderLayer(nn.Module):
    """Single transformer decoder layer.

    Applies three sublayers, each with pre-norm and residual:
      1. Masked multi-head self-attention (causal mask prevents attending to future tokens).
      2. Cross-attention over encoder output (keys and values from encoder).
      3. Position-wise feed-forward network.
    """

    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        src_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass through one decoder layer.

        Args:
            x: Target embeddings, shape (batch, tgt_len, d_model).
            memory: Encoder output, shape (batch, src_len, d_model).
            tgt_mask: Causal mask for target self-attention,
                      shape (1, 1, tgt_len, tgt_len). True = masked.
            src_mask: Padding mask for source, shape (batch, 1, 1, src_len).

        Returns:
            Output of shape (batch, tgt_len, d_model).
        """
        # Masked self-attention
        sa_out, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), tgt_mask)
        x = x + self.dropout(sa_out)

        # Cross-attention
        ca_out, _ = self.cross_attn(self.norm2(x), memory, memory, src_mask)
        x = x + self.dropout(ca_out)

        # Feed-forward
        x = x + self.dropout(self.ff(self.norm3(x)))
        return x


class Decoder(nn.Module):
    """Stack of N decoder layers with a final layer norm."""

    def __init__(
        self, n_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        src_mask: Tensor | None = None,
    ) -> Tensor:
        """Pass target through all decoder layers.

        Args:
            x: Target embeddings, shape (batch, tgt_len, d_model).
            memory: Encoder output, shape (batch, src_len, d_model).
            tgt_mask: Causal mask for the target sequence.
            src_mask: Padding mask for the source sequence.

        Returns:
            Decoder output of shape (batch, tgt_len, d_model).
        """
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, src_mask)
        return self.norm(x)
