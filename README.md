# Attention Is All You Need

PyTorch implementation of the transformer architecture from [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762). Built from scratch — no `nn.Transformer` shortcuts. Covers the full encoder-decoder stack with multi-head self-attention, positional encoding, and masked attention for autoregressive decoding.

## Architecture

```
Encoder:
  Input Embedding + Positional Encoding
  → N x EncoderLayer
      → Multi-Head Self-Attention (h heads)
      → Add & Norm
      → Feed-Forward (d_model → d_ff → d_model)
      → Add & Norm

Decoder:
  Output Embedding + Positional Encoding
  → N x DecoderLayer
      → Masked Multi-Head Self-Attention
      → Add & Norm
      → Cross-Attention (queries from decoder, keys/values from encoder)
      → Add & Norm
      → Feed-Forward
      → Add & Norm

Linear + Softmax → output distribution
```

Default hyperparameters match the "base" model from the paper: `d_model=512`, `h=8`, `N=6`, `d_ff=2048`, `dropout=0.1`.

## Components

| Module | Description |
|---|---|
| `attention.py` | Scaled dot-product attention and multi-head attention |
| `encoder.py` | Encoder layer and encoder stack |
| `decoder.py` | Decoder layer (masked self-attn + cross-attn) and decoder stack |
| `embeddings.py` | Token embeddings + sinusoidal positional encoding |
| `transformer.py` | Full encoder-decoder model |
| `train.py` | Training loop with label smoothing and learning rate warmup |

## Setup

### Prerequisites

- Python 3.10+
- PyTorch 2.0+

### Install

```bash
git clone https://github.com/quinnhasse/AttentionIsAllYouNeed.git
cd AttentionIsAllYouNeed
pip install torch numpy
```

### Train

```bash
# Train on a toy copy task (default)
python train.py

# Override hyperparameters
python train.py --d_model 256 --n_heads 4 --n_layers 3 --epochs 20
```

### Run inference

```python
from transformer import Transformer

model = Transformer(
    src_vocab_size=1000,
    tgt_vocab_size=1000,
    d_model=512,
    n_heads=8,
    n_layers=6,
    d_ff=2048,
    dropout=0.1,
)

# src: (batch, src_seq_len), tgt: (batch, tgt_seq_len)
output = model(src, tgt, src_mask, tgt_mask)
```

## Key implementation details

- **Scaled dot-product attention**: scores are divided by `sqrt(d_k)` before softmax to prevent gradient vanishing in deep models
- **Sinusoidal positional encoding**: fixed (not learned), using alternating sin/cos at exponentially spaced frequencies
- **Label smoothing**: targets are softened to `epsilon=0.1` during training
- **Learning rate schedule**: warmup for `warmup_steps` steps, then inverse square root decay — matches the paper exactly
- **Padding and causal masks**: separate masks for source padding and target future positions

## Reference

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). *Attention Is All You Need*. NeurIPS 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
