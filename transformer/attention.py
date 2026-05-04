"""Scaled dot-product and multi-head attention."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor | None = None,
    dropout: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """Compute scaled dot-product attention.

    Args:
        q: Query tensor of shape (batch, heads, seq_q, d_k).
        k: Key tensor of shape (batch, heads, seq_k, d_k).
        v: Value tensor of shape (batch, heads, seq_k, d_v).
        mask: Boolean mask of shape broadcastable to (batch, heads, seq_q, seq_k).
              True positions are masked out (set to -inf before softmax).
        dropout: Dropout probability applied to attention weights.

    Returns:
        Tuple of (output, attention_weights), both tensors.
        output shape: (batch, heads, seq_q, d_v).
        attention_weights shape: (batch, heads, seq_q, seq_k).
    """
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    attn = F.softmax(scores, dim=-1)

    if dropout > 0.0 and torch.is_grad_enabled():
        attn = F.dropout(attn, p=dropout)

    output = torch.matmul(attn, v)
    return output, attn


class MultiHeadAttention(nn.Module):
    """Multi-head attention as described in Vaswani et al., 2017.

    Splits d_model into h heads of dimension d_k = d_model / h,
    runs scaled dot-product attention in parallel, then projects
    the concatenated output back to d_model.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        """Initialize multi-head attention.

        Args:
            d_model: Model dimension. Must be divisible by n_heads.
            n_heads: Number of attention heads.
            dropout: Dropout on attention weights.
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.dropout = dropout

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Run multi-head attention.

        Args:
            query: (batch, seq_q, d_model)
            key:   (batch, seq_k, d_model)
            value: (batch, seq_k, d_model)
            mask:  Boolean mask broadcastable to (batch, n_heads, seq_q, seq_k).
                   True = masked out.

        Returns:
            Tuple of (output, attention_weights).
            output: (batch, seq_q, d_model)
            attention_weights: (batch, n_heads, seq_q, seq_k)
        """
        batch = query.size(0)

        # Project and split into heads: (batch, seq, d_model) -> (batch, heads, seq, d_k)
        def split_heads(x: Tensor) -> Tensor:
            return x.view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)

        q = split_heads(self.W_q(query))
        k = split_heads(self.W_k(key))
        v = split_heads(self.W_v(value))

        # Expand mask for head dimension if needed
        if mask is not None and mask.dim() == 3:
            mask = mask.unsqueeze(1)  # (batch, 1, seq_q, seq_k)

        out, attn = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout=self.dropout if self.training else 0.0
        )

        # Merge heads: (batch, heads, seq, d_k) -> (batch, seq, d_model)
        out = out.transpose(1, 2).contiguous().view(batch, -1, self.d_model)
        return self.W_o(out), attn
