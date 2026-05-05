"""Token embeddings and sinusoidal positional encoding."""

import math
import torch
import torch.nn as nn
from torch import Tensor


class TokenEmbedding(nn.Module):
    """Learned token embedding scaled by sqrt(d_model).

    The sqrt(d_model) scaling from the paper prevents the positional
    encoding from dominating the embedding signal at initialisation.
    """

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """Embed token ids and scale.

        Args:
            x: Token ids of shape (batch, seq_len).

        Returns:
            Embeddings of shape (batch, seq_len, d_model).
        """
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding from Vaswani et al., 2017.

    PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    The encoding is added to the token embedding and is not learned.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Register as buffer so it moves with .to(device) but is not a parameter.
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """Add positional encoding to input embeddings.

        Args:
            x: Embeddings of shape (batch, seq_len, d_model).

        Returns:
            x + PE[:, :seq_len], same shape, with dropout applied.
        """
        x = x + self.pe[:, : x.size(1)]  # type: ignore[index]
        return self.dropout(x)
