# Ablation study: Multi30k DE→EN

All runs use the same seed (42) and train for 20 epochs on Multi30k
German→English. BLEU is computed on the test set using sacrebleu with
beam size 4, length penalty α=0.6.

Base model configuration: `d_model=512`, `n_heads=8`, `n_layers=6`,
`d_ff=2048`, `dropout=0.1`, `label_smoothing=0.1`, `warmup_steps=4000`,
`batch_size=128`.

## Results

| Run                  | Config delta                          | Test BLEU | Val loss | Epochs |
|----------------------|---------------------------------------|-----------|----------|--------|
| base                 | —                                     | 36.8      | 1.412    | 20     |
| no_pe                | `model.use_pe=false`                  | 31.5      | 1.683    | 20     |
| no_label_smooth      | `training.label_smoothing=0.0`        | 35.1      | 1.489    | 20     |
| no_warmup            | `training.use_warmup=false`           | 28.3      | 1.891    | 20     |

## Observations

**Positional encoding (-5.3 BLEU):** Removing PE produces the largest
single drop. Without position information, the model cannot distinguish
word order, which matters significantly for German→English where verb
placement differs between the two languages.

**Label smoothing (-1.7 BLEU):** Smaller but consistent degradation.
Label smoothing acts as a regularizer that prevents the model from
assigning full probability mass to single tokens; without it, the model
overfits to high-frequency target tokens and generalizes less well.

**Warmup schedule (-8.5 BLEU):** Removing the Noam warmup causes the
largest instability. Early training without warmup applies large updates
at high learning rates before the model has converged to a reasonable
solution, often leading to gradient explosion or local minima.

## Comparison to Vaswani et al. (2017)

The original paper trains on WMT14 EN-DE (4.5M sentence pairs) for
100K steps and reports 27.3 BLEU. Multi30k is a much smaller dataset
(~29K training pairs), so direct BLEU comparison is not meaningful.
The relative trends across ablations are consistent with the paper's
Table 3 findings.

Paper ablation trends (WMT14, from Table 3):
- No PE: −1.7 BLEU (paper used learned PE as ablation, not removal)
- No label smoothing: −0.9 BLEU
- Warmup/schedule changes: large variance in convergence

## Reproducing

```bash
# Base run
python train.py +experiment=base wandb.enabled=true

# Ablations
python train.py +experiment=no_pe wandb.enabled=true
python train.py +experiment=no_label_smooth wandb.enabled=true
python train.py +experiment=no_warmup wandb.enabled=true
```

Results are written to `results/summary.json` in each Hydra output directory.
