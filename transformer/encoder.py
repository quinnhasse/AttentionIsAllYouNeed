"""Transformer encoder: feed-forward sublayer and stacked encoder layers."""

import torch.nn as nn
from torch import Tensor

from transformer.attention import MultiHeadAttention


class FeedForward(nn.Module):
    """Position-wise feed-forward network.

    Two linear layers with ReLU activation and dropout:
    FFN(x) = max(0, x W_1 + b_1) W_2 + b_2

    d_ff is typically 4 * d_model.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply feed-forward network.

        Args:
            x: Input of shape (batch, seq_len, d_model).

        Returns:
            Output of shape (batch, seq_len, d_model).
        """
        return self.net(x)


class EncoderLayer(nn.Module):
    """Single transformer encoder layer.

    Applies:
      1. Multi-head self-attention with residual connection and layer norm.
      2. Position-wise feed-forward with residual connection and layer norm.

    Uses pre-norm (layer norm before each sublayer) which improves
    training stability compared to the original post-norm formulation.
    """

    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor, src_mask: Tensor | None = None) -> Tensor:
        """Forward pass through one encoder layer.

        Args:
            x: (batch, src_len, d_model)
            src_mask: Boolean padding mask of shape (batch, 1, 1, src_len)
                      or (batch, src_len). True = masked (pad tokens).

        Returns:
            Output of shape (batch, src_len, d_model).
        """
        # Self-attention sublayer
        attn_out, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), src_mask)
        x = x + self.dropout(attn_out)

        # Feed-forward sublayer
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class Encoder(nn.Module):
    """Stack of N encoder layers with a final layer norm."""

    def __init__(
        self, n_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, src_mask: Tensor | None = None) -> Tensor:
        """Pass input through all encoder layers.

        Args:
            x: Embedded source tokens, shape (batch, src_len, d_model).
            src_mask: Optional padding mask.

        Returns:
            Encoder output of shape (batch, src_len, d_model).
        """
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)
