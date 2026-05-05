"""Full encoder-decoder Transformer and decoder-only language model."""

import torch
import torch.nn as nn
from torch import Tensor

from transformer.attention import MultiHeadAttention
from transformer.decoder import Decoder, make_causal_mask
from transformer.embeddings import PositionalEncoding, TokenEmbedding
from transformer.encoder import Encoder


class Transformer(nn.Module):
    """Full encoder-decoder transformer for sequence-to-sequence tasks.

    Matches the architecture of Vaswani et al., 2017. The default
    hyperparameters correspond to the 'base' model from the paper:
    d_model=512, n_heads=8, n_layers=6, d_ff=2048, dropout=0.1.
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 512,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model

        self.src_embed = TokenEmbedding(src_vocab_size, d_model)
        self.tgt_embed = TokenEmbedding(tgt_vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)

        self.encoder = Encoder(n_layers, d_model, n_heads, d_ff, dropout)
        self.decoder = Decoder(n_layers, d_model, n_heads, d_ff, dropout)

        self.output_proj = nn.Linear(d_model, tgt_vocab_size, bias=False)
        # Tie output projection weights to target embedding (saves params, improves perf)
        self.output_proj.weight = self.tgt_embed.embedding.weight

        self._init_params()

    def _init_params(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src: Tensor) -> Tensor:
        """Build padding mask for source tokens.

        Returns bool mask of shape (batch, 1, 1, src_len).
        True = pad token (masked out).
        """
        return (src == self.pad_idx).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt: Tensor) -> Tensor:
        """Build combined causal + padding mask for target tokens.

        Returns bool mask of shape (batch, 1, tgt_len, tgt_len).
        True = masked out.
        """
        tgt_len = tgt.size(1)
        causal = make_causal_mask(tgt_len, device=tgt.device)  # (1, 1, T, T)
        pad = (tgt == self.pad_idx).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
        return causal | pad

    def encode(self, src: Tensor, src_mask: Tensor | None = None) -> Tensor:
        """Run encoder on source tokens.

        Args:
            src: Token ids, shape (batch, src_len).
            src_mask: Optional padding mask.

        Returns:
            Encoder output, shape (batch, src_len, d_model).
        """
        return self.encoder(self.pos_enc(self.src_embed(src)), src_mask)

    def decode(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        src_mask: Tensor | None = None,
    ) -> Tensor:
        """Run decoder on target tokens given encoder memory.

        Args:
            tgt: Target token ids, shape (batch, tgt_len).
            memory: Encoder output, shape (batch, src_len, d_model).
            tgt_mask: Causal+padding mask for target.
            src_mask: Padding mask for source.

        Returns:
            Decoder output, shape (batch, tgt_len, d_model).
        """
        return self.decoder(self.pos_enc(self.tgt_embed(tgt)), memory, tgt_mask, src_mask)

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        """Full forward pass: encode source, decode target, project to vocab.

        Args:
            src: Source token ids, shape (batch, src_len).
            tgt: Target token ids (teacher-forced), shape (batch, tgt_len).

        Returns:
            Logits of shape (batch, tgt_len, tgt_vocab_size).
        """
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        memory = self.encode(src, src_mask)
        out = self.decode(tgt, memory, tgt_mask, src_mask)
        return self.output_proj(out)

    @torch.no_grad()
    def greedy_decode(
        self, src: Tensor, bos_idx: int, eos_idx: int, max_len: int = 100
    ) -> Tensor:
        """Greedy decoding (argmax at each step).

        Args:
            src: Source token ids, shape (1, src_len).
            bos_idx: Beginning-of-sequence token id.
            eos_idx: End-of-sequence token id.
            max_len: Maximum number of tokens to generate.

        Returns:
            Generated token ids (without BOS), shape (1, generated_len).
        """
        self.eval()
        src_mask = self.make_src_mask(src)
        memory = self.encode(src, src_mask)

        tgt = torch.tensor([[bos_idx]], dtype=torch.long, device=src.device)
        for _ in range(max_len - 1):
            tgt_mask = self.make_tgt_mask(tgt)
            out = self.decode(tgt, memory, tgt_mask, src_mask)
            next_token = self.output_proj(out[:, -1, :]).argmax(dim=-1, keepdim=True)
            tgt = torch.cat([tgt, next_token], dim=1)
            if next_token.item() == eos_idx:
                break

        return tgt[:, 1:]  # strip BOS


class TransformerLM(nn.Module):
    """Decoder-only transformer for causal language modeling.

    Uses only the decoder stack with a causal mask. No encoder or
    cross-attention — each layer uses only masked self-attention and FFN.
    Suitable for language modeling on WikiText-2.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 1024,
        max_len: int = 512,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.vocab_size = vocab_size

        self.embed = TokenEmbedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)

        # We use the Encoder stack for the LM (self-attention only, no cross-attn needed)
        # But we need causal masking, so we use the Decoder with memory=x trick.
        # Simpler: implement a dedicated causal encoder stack.
        self.layers = nn.ModuleList(
            [_CausalEncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        self.output_proj.weight = self.embed.embedding.weight

        self._init_params()

    def _init_params(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through causal language model.

        Args:
            x: Token ids of shape (batch, seq_len).

        Returns:
            Logits of shape (batch, seq_len, vocab_size).
        """
        seq_len = x.size(1)
        causal_mask = make_causal_mask(seq_len, device=x.device)  # (1, 1, T, T)

        h = self.pos_enc(self.embed(x))
        for layer in self.layers:
            h = layer(h, causal_mask)
        h = self.norm(h)
        return self.output_proj(h)

    @torch.no_grad()
    def generate(
        self,
        prompt: Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> Tensor:
        """Autoregressive generation with top-k sampling.

        Args:
            prompt: Token ids of shape (1, prompt_len).
            max_new_tokens: Number of tokens to generate beyond the prompt.
            temperature: Softmax temperature (lower = more deterministic).
            top_k: Keep only the top-k logits before sampling.

        Returns:
            Generated token ids of shape (1, prompt_len + max_new_tokens).
        """
        self.eval()
        tokens = prompt.clone()

        for _ in range(max_new_tokens):
            logits = self.forward(tokens)[:, -1, :] / temperature
            if top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, -1:]] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)

        return tokens


class _CausalEncoderLayer(nn.Module):
    """Encoder layer with causal masking for the LM.

    Identical to EncoderLayer except it accepts a pre-built causal
    mask argument so TransformerLM can reuse the same mask across all
    layers without recomputing it.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        from transformer.encoder import FeedForward
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """Apply causal self-attention and FFN.

        Args:
            x: (batch, seq_len, d_model)
            mask: Causal mask (1, 1, seq_len, seq_len).

        Returns:
            Output of shape (batch, seq_len, d_model).
        """
        attn_out, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), mask)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x
