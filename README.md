# Attention Is All You Need

PyTorch implementation of the transformer from [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762),
with a full training run on Multi30k DE→EN, W&B logging, and three ablations.

W&B project: https://wandb.ai/quinnhasse/attention-is-all-you-need

## Results

Multi30k DE→EN, 20 epochs, beam 4, α=0.6:

| Run | Test BLEU | Δ vs base |
|---|---|---|
| base | 36.8 | — |
| no positional encoding | 31.5 | −5.3 |
| no label smoothing | 35.1 | −1.7 |
| no warmup schedule | 28.3 | −8.5 |

Full results: [`results/ablations.md`](results/ablations.md) · [`results/ablations.csv`](results/ablations.csv)

Technical report: [`report.md`](report.md)

## Architecture

```
Encoder:
  src_embed + PositionalEncoding (optional, ablatable)
  → 6 × EncoderLayer
      → Multi-Head Self-Attention (h=8, d_k=64)
      → Add & Norm (pre-norm)
      → FFN (d_model=512 → d_ff=2048 → d_model=512)
      → Add & Norm

Decoder:
  tgt_embed + PositionalEncoding (optional, ablatable)
  → 6 × DecoderLayer
      → Masked Multi-Head Self-Attention
      → Add & Norm
      → Cross-Attention (queries from decoder, k/v from encoder)
      → Add & Norm
      → FFN
      → Add & Norm

Linear (weight-tied to tgt_embed) + log-softmax → output distribution
```

Base model: `d_model=512`, `h=8`, `N=6`, `d_ff=2048`, `dropout=0.1`.

## Setup

```bash
git clone https://github.com/quinnhasse/AttentionIsAllYouNeed.git
cd AttentionIsAllYouNeed
pip install -r requirements.txt
```

## Training

```bash
# Base run (Multi30k DE→EN)
python train.py +experiment=base

# With W&B logging
python train.py +experiment=base wandb.enabled=true

# Ablations
python train.py +experiment=no_pe
python train.py +experiment=no_label_smooth
python train.py +experiment=no_warmup

# Override individual params
python train.py training.epochs=30 training.batch_size=64
```

Outputs go to `outputs/<date>/<time>/` (Hydra default). Each run writes
`checkpoints/best.pt` and `results/summary.json`.

## Evaluation

```bash
# Evaluate a checkpoint (beam search, default beam=4)
python evaluate_bleu.py outputs/<date>/<time>/checkpoints/best.pt

# Greedy decode
python evaluate_bleu.py checkpoints/best.pt --beam 1

# Evaluate on validation set
python evaluate_bleu.py checkpoints/best.pt --split val --beam 4
```

## Tests

```bash
pip install pytest
pytest tests/ -v
```

Coverage:
- `tests/test_attention.py` — scaled dot-product attention, multi-head attention shapes and gradients
- `tests/test_masking.py` — causal mask structure, padding mask, combined mask interactions
- `tests/test_tokenizer.py` — BPE encode/decode, special tokens, truncation
- `tests/test_model.py` — full transformer forward shapes, greedy decode, no-PE variant

## Modules

| File | Description |
|---|---|
| `transformer/attention.py` | Scaled dot-product attention and multi-head attention |
| `transformer/encoder.py` | Encoder layer and stack |
| `transformer/decoder.py` | Decoder layer (masked self-attn + cross-attn) and stack |
| `transformer/embeddings.py` | Token embeddings + sinusoidal positional encoding |
| `transformer/model.py` | Full encoder-decoder Transformer; `use_pe` flag for ablation |
| `transformer/scheduler.py` | Noam learning rate schedule |
| `data/translation.py` | Multi30k data pipeline with BPE tokenization |
| `data/dataset.py` | WikiText-2 pipeline for language modeling |
| `train.py` | Hydra training script (seq2seq on Multi30k) |
| `evaluate_bleu.py` | Checkpoint evaluation with beam search + sacrebleu |
| `conf/` | Hydra configs: base model and three ablation experiments |
| `results/` | Ablation table (markdown + CSV) |
| `report.md` | Method, results, and limitations |

## Key implementation notes

- **Pre-norm** (LayerNorm before each sublayer) rather than the original post-norm.
  More stable for smaller datasets.
- **Sinusoidal PE**: fixed, not learned. Alternating sin/cos at exponentially-spaced
  frequencies. Set `model.use_pe=false` to ablate.
- **Label smoothing**: ε=0.1, distributes 0.1 probability mass uniformly across vocab
  (excluding pad). Set `training.label_smoothing=0.0` for standard CE.
- **Noam schedule**: warmup for 4000 steps then inverse-sqrt decay.
  Set `training.use_warmup=false` to use constant LR at peak Noam value.
- **Weight tying**: output projection shares weights with target embedding matrix.

## Reference

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N.,
Kaiser, L., & Polosukhin, I. (2017). *Attention Is All You Need*. NeurIPS 2017.
[arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
