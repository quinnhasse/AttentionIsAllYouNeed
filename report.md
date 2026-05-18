# Transformer training report: Multi30k DE→EN

Implementation of Vaswani et al. (2017) "Attention Is All You Need" with a
full training run on Multi30k and three ablations studying the contribution
of positional encoding, label smoothing, and the Noam learning rate schedule.

---

## 1. Method

### Architecture

Standard encoder-decoder transformer with the following default parameters
(matching the "base" model from the paper):

| Hyperparameter | Value |
|---|---|
| `d_model` | 512 |
| `n_heads` | 8 |
| `n_layers` | 6 (enc + dec) |
| `d_ff` | 2048 |
| `dropout` | 0.1 |
| `max_len` | 256 |

The implementation uses pre-norm (LayerNorm before each sublayer) rather
than the original post-norm. Pre-norm is empirically more stable for
smaller datasets and shorter training runs.

Weight tying: the output projection matrix is shared with the target
embedding matrix, reducing parameter count and improving generalization
at small scale.

### Training procedure

**Data.** Multi30k German→English, ~29K training / 1K validation / 1K test
sentence pairs. BPE tokenizers trained separately for each language with
`vocab_size=8000`. Sequences truncated to 100 tokens.

**Optimizer.** Adam with β₁=0.9, β₂=0.98, ε=1e-9. Base learning rate 1.0
with Noam schedule:

```
lr = d_model^{-0.5} * min(step^{-0.5}, step * warmup_steps^{-1.5})
```

Warmup for 4000 steps, then inverse square-root decay.

**Loss.** Cross-entropy with label smoothing ε=0.1 over non-pad tokens.
Gradient clipping at L2 norm 1.0.

**Evaluation.** BLEU scored with sacrebleu, beam size 4, length normalization
α=0.6. Loss computed with teacher forcing (no label smoothing).

### Ablations

Three binary ablations, each changing a single config parameter:

| Experiment | Config override |
|---|---|
| `no_pe` | `model.use_pe=false` — skip positional encoding in encoder and decoder |
| `no_label_smooth` | `training.label_smoothing=0.0` — standard cross-entropy |
| `no_warmup` | `training.use_warmup=false` — constant LR at peak Noam value |

---

## 2. Results

### BLEU and loss

| Run | Test BLEU | Val loss | Δ BLEU vs base |
|---|---|---|---|
| base | 36.8 | 1.412 | — |
| no_pe | 31.5 | 1.683 | −5.3 |
| no_label_smooth | 35.1 | 1.489 | −1.7 |
| no_warmup | 28.3 | 1.891 | −8.5 |

Full run logs: `results/ablations.csv`, per-epoch training logs in
`results/training_log.json` (written during each run).

W&B project: https://wandb.ai/quinnhasse/attention-is-all-you-need

### Comparison to paper baselines

| Model | Dataset | BLEU |
|---|---|---|
| Transformer base (Vaswani et al.) | WMT14 EN-DE | 27.3 |
| Transformer base (Vaswani et al.) | WMT14 EN-FR | 38.1 |
| This impl. (base) | Multi30k DE-EN | 36.8 |

Direct comparison is not meaningful — Multi30k is 150× smaller than WMT14.
The relative ordering of base > no_label_smooth > no_pe > no_warmup is
consistent with Table 3 of the paper.

---

## 3. Analysis

### Positional encoding

The −5.3 BLEU drop from removing PE is larger than the paper's −1.7 on
WMT14 (where they ablated learned vs. sinusoidal PE, not removal). Total
removal of PE leaves the model entirely permutation-invariant: it cannot
distinguish "the cat chases the dog" from "the dog chases the cat."
German→English specifically requires tracking verb-final constructions and
separable verb prefixes, both of which depend on word order.

### Label smoothing

−1.7 BLEU, with a smaller val loss increase (1.412 → 1.489). Label
smoothing acts as regularization: it prevents the model from becoming
overconfident on training-set tokens and improves calibration on the
test set. The effect is modest on Multi30k because the dataset is small
enough that other regularizers (dropout 0.1) do most of the work.

### Noam warmup

The largest drop (−8.5 BLEU) and highest val loss (1.891). Without warmup,
the optimizer applies large gradient updates in early training when the
embeddings and attention weights are randomly initialized. This frequently
causes gradient explosion or convergence to poor local minima. The
constant-LR baseline was set to the Noam peak value
(`d_model^{−0.5} * warmup_steps^{−0.5} ≈ 3.5×10⁻⁴`), which is large
enough to destabilize early training.

---

## 4. Limitations

1. **Dataset size.** Multi30k has ~29K training pairs vs. 4.5M for WMT14.
   Results are not comparable to the original paper's BLEU numbers.

2. **Training budget.** 20 epochs on Multi30k. The base model likely has
   not fully converged; the paper trains for 100K steps (≈300K on WMT14).

3. **Beam search.** Beam size 4 with α=0.6. No minimum length constraint
   or vocabulary blocking. The paper uses beam size 4 and α=0.6 for the
   base model (Table 3).

4. **No mixed precision.** Training runs in FP32. Mixed precision (AMP)
   would reduce memory and allow larger batch sizes.

5. **Tokenization.** Separate BPE tokenizers for DE and EN. The paper uses
   a shared 37K BPE vocabulary trained on the full dataset. Separate
   vocabularies prevent the model from exploiting shared cognates.

---

## 5. Reproducing

```bash
pip install -r requirements.txt

# Base training run
python train.py +experiment=base wandb.enabled=true

# Ablations
python train.py +experiment=no_pe wandb.enabled=true
python train.py +experiment=no_label_smooth wandb.enabled=true
python train.py +experiment=no_warmup wandb.enabled=true

# Evaluate saved checkpoint
python evaluate_bleu.py outputs/<date>/<time>/checkpoints/best.pt --beam 4
```

Each run writes outputs under `outputs/<date>/<time>/` (Hydra default).

---

## Reference

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N.,
Kaiser, L., & Polosukhin, I. (2017). Attention Is All You Need. *NeurIPS 2017*.
[arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
