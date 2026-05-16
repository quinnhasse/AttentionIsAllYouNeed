"""Train Transformer (enc-dec) on Multi30k DE→EN with Hydra configs and W&B logging.

Usage:
    python train.py                          # base config
    python train.py +experiment=base         # explicit base experiment
    python train.py +experiment=no_pe        # ablation: no positional encoding
    python train.py +experiment=no_label_smooth
    python train.py +experiment=no_warmup

Override individual params:
    python train.py training.epochs=30 training.batch_size=64
    python train.py wandb.enabled=true wandb.run_name=my-run

All outputs go under outputs/<date>/<time>/ (Hydra default).
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from tqdm import tqdm

from transformer.model import Transformer
from transformer.scheduler import get_noam_scheduler
from data.translation import get_translation_dataloaders, BOS_IDX, EOS_IDX, PAD_IDX


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def label_smoothed_ce(
    logits: Tensor,
    targets: Tensor,
    smoothing: float = 0.1,
    pad_idx: int = PAD_IDX,
) -> Tensor:
    """Cross-entropy with label smoothing over non-pad tokens.

    Distributes `smoothing` probability mass uniformly across the
    vocabulary and reserves `1 - smoothing` for the correct label.

    Args:
        logits: (N, vocab_size) where N = batch * seq_len.
        targets: (N,) token ids.
        smoothing: Label smoothing epsilon (0.0 = standard CE).
        pad_idx: Token id to exclude from loss.

    Returns:
        Scalar mean loss over non-pad tokens.
    """
    vocab_size = logits.size(-1)
    log_probs = torch.log_softmax(logits, dim=-1)

    with torch.no_grad():
        smooth_targets = torch.full_like(log_probs, smoothing / (vocab_size - 2))
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing)
        smooth_targets[:, pad_idx] = 0.0
        mask = targets == pad_idx
        smooth_targets[mask] = 0.0

    loss = -(smooth_targets * log_probs).sum(dim=-1)
    n_tokens = (~mask).sum()
    return loss.sum() / n_tokens.clamp(min=1)


# ---------------------------------------------------------------------------
# Evaluation (greedy BLEU)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_bleu(
    model: Transformer,
    loader,
    tgt_tokenizer,
    device: torch.device,
    max_gen_len: int = 100,
) -> float:
    """Compute corpus BLEU on a dataloader using greedy decoding.

    Args:
        model: Trained Transformer in eval mode.
        loader: DataLoader yielding (src_ids, tgt_ids).
        tgt_tokenizer: Target-side tokenizer for decoding ids to strings.
        device: Device for inference.
        max_gen_len: Maximum generated tokens per example.

    Returns:
        Corpus BLEU score (0–100).
    """
    try:
        import sacrebleu
    except ImportError:
        return 0.0

    model.eval()
    hypotheses: list[str] = []
    references: list[str] = []

    for src, tgt in loader:
        src = src.to(device)
        for i in range(src.size(0)):
            src_i = src[i].unsqueeze(0)  # (1, src_len)
            gen = model.greedy_decode(src_i, BOS_IDX, EOS_IDX, max_len=max_gen_len)
            hyp_ids = gen[0].tolist()
            hyp = tgt_tokenizer.decode(hyp_ids)
            hypotheses.append(hyp)

            ref_ids = tgt[i].tolist()
            ref_ids = [t for t in ref_ids if t not in (PAD_IDX, BOS_IDX, EOS_IDX)]
            ref = tgt_tokenizer.decode(ref_ids)
            references.append(ref)

    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return bleu.score


@torch.no_grad()
def evaluate_loss(
    model: Transformer,
    loader,
    device: torch.device,
    pad_idx: int = PAD_IDX,
) -> float:
    """Compute token-level cross-entropy (no label smoothing) on a split.

    Args:
        model: Transformer in eval mode.
        loader: DataLoader yielding (src_ids, tgt_ids).
        device: Inference device.
        pad_idx: Token id to exclude from loss calculation.

    Returns:
        Mean cross-entropy in nats per non-pad token.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    total_loss = 0.0
    total_tokens = 0

    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        # Teacher-forced: feed tgt[:-1] as input, predict tgt[1:]
        tgt_input = tgt[:, :-1]
        tgt_target = tgt[:, 1:]

        logits = model(src, tgt_input)  # (B, T-1, vocab)
        B, T, V = logits.shape
        loss = criterion(logits.reshape(B * T, V), tgt_target.reshape(B * T))
        n_tokens = (tgt_target != pad_idx).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    return total_loss / max(total_tokens, 1)


# ---------------------------------------------------------------------------
# Gradient norm helper
# ---------------------------------------------------------------------------

def _grad_norm(model: nn.Module) -> float:
    """Compute the global L2 norm of all gradients.

    Args:
        model: Model with gradients populated after backward().

    Returns:
        Global gradient L2 norm as a Python float.
    """
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().norm(2).item() ** 2
    return math.sqrt(total)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Train Transformer on Multi30k using Hydra config.

    Args:
        cfg: Merged Hydra config (config.yaml + experiment override).
    """
    print(OmegaConf.to_yaml(cfg))

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)
    print(f"device: {device}")

    # Data
    print("loading Multi30k ...")
    train_loader, val_loader, test_loader, src_tok, tgt_tok = get_translation_dataloaders(
        src_lang=cfg.data.src_lang,
        tgt_lang=cfg.data.tgt_lang,
        vocab_size=cfg.data.vocab_size,
        batch_size=cfg.training.batch_size,
        max_seq_len=cfg.data.max_seq_len,
        seed=cfg.seed,
    )
    src_vocab = src_tok.get_vocab_size()
    tgt_vocab = tgt_tok.get_vocab_size()
    print(f"src vocab: {src_vocab}  tgt vocab: {tgt_vocab}")

    # Model
    model = Transformer(
        src_vocab_size=src_vocab,
        tgt_vocab_size=tgt_vocab,
        d_model=cfg.model.d_model,
        n_heads=cfg.model.n_heads,
        n_layers=cfg.model.n_layers,
        d_ff=cfg.model.d_ff,
        max_len=cfg.model.max_len,
        dropout=cfg.model.dropout,
        pad_idx=PAD_IDX,
        use_pe=cfg.model.use_pe,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"parameters: {n_params:,}")

    # Optimizer + scheduler
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
    )
    if cfg.training.use_warmup:
        scheduler = get_noam_scheduler(
            optimizer,
            d_model=cfg.model.d_model,
            warmup_steps=cfg.training.warmup_steps,
        )
    else:
        # Constant lr schedule: lrate = d_model^{-0.5} * warmup_steps^{-0.5}
        peak = cfg.model.d_model ** -0.5 * cfg.training.warmup_steps ** -0.5
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=peak)

    # W&B (optional)
    wb_run = None
    if cfg.wandb.enabled:
        try:
            import wandb
            run_name = cfg.wandb.run_name or (
                f"d{cfg.model.d_model}-h{cfg.model.n_heads}"
                f"-pe{cfg.model.use_pe}-ls{cfg.training.label_smoothing}"
                f"-wu{cfg.training.use_warmup}"
            )
            wb_run = wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity or None,
                name=run_name,
                config=OmegaConf.to_container(cfg, resolve=True),
            )
        except Exception as exc:
            print(f"W&B init failed: {exc}")

    # Output dirs (Hydra sets cwd to outputs/<date>/<time>)
    save_dir = Path(cfg.training.save_dir)
    results_dir = Path(cfg.training.results_dir)
    save_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    log: list[dict] = []
    best_val_loss = float("inf")
    step = 0

    for epoch in range(1, cfg.training.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0
        t0 = time.time()

        for src, tgt in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            src, tgt = src.to(device), tgt.to(device)
            tgt_input  = tgt[:, :-1]
            tgt_target = tgt[:, 1:]

            optimizer.zero_grad()
            logits = model(src, tgt_input)
            B, T, V = logits.shape
            loss = label_smoothed_ce(
                logits.reshape(B * T, V),
                tgt_target.reshape(B * T),
                smoothing=cfg.training.label_smoothing,
            )
            loss.backward()

            gnorm = _grad_norm(model)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1

            n_tokens = (tgt_target != PAD_IDX).sum().item()
            epoch_loss += loss.item() * n_tokens
            epoch_tokens += n_tokens

            if wb_run and step % 100 == 0:
                wb_run.log({
                    "train/loss_step": loss.item(),
                    "train/grad_norm": gnorm,
                    "train/lr": scheduler.get_last_lr()[0],
                    "step": step,
                })

        train_loss = epoch_loss / max(epoch_tokens, 1)
        val_loss = evaluate_loss(model, val_loader, device)

        # BLEU on validation set every 5 epochs (expensive: greedy decode)
        val_bleu = 0.0
        if epoch % 5 == 0 or epoch == cfg.training.epochs:
            val_bleu = evaluate_bleu(model, val_loader, tgt_tok, device)

        elapsed = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]

        entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_bleu": round(val_bleu, 2),
            "lr": lr_now,
            "grad_norm": round(gnorm, 4),
            "elapsed_s": round(elapsed, 1),
        }
        log.append(entry)
        print(
            f"epoch {epoch:>3} | train {train_loss:.4f} | val {val_loss:.4f}"
            f" | bleu {val_bleu:.1f} | lr {lr_now:.2e} | gnorm {gnorm:.2f}"
            f" | {elapsed:.1f}s"
        )

        if wb_run:
            wb_run.log({
                "train/loss_epoch": train_loss,
                "val/loss": val_loss,
                "val/bleu": val_bleu,
                "train/lr": lr_now,
                "epoch": epoch,
            })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "cfg": OmegaConf.to_container(cfg, resolve=True),
                    "val_bleu": val_bleu,
                },
                save_dir / "best.pt",
            )

        with open(results_dir / "training_log.json", "w") as f:
            json.dump(log, f, indent=2)

    # Final test BLEU from best checkpoint
    ckpt = torch.load(save_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    test_loss = evaluate_loss(model, test_loader, device)
    test_bleu = evaluate_bleu(model, test_loader, tgt_tok, device)

    summary = {
        "best_val_loss": round(best_val_loss, 4),
        "test_loss": round(test_loss, 4),
        "test_bleu": round(test_bleu, 2),
        "n_params": n_params,
        "epochs": cfg.training.epochs,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    print(f"\nbest val loss: {best_val_loss:.4f}")
    print(f"test BLEU:     {test_bleu:.2f}")

    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if wb_run:
        wb_run.summary.update({"test/bleu": test_bleu, "test/loss": test_loss})
        wb_run.finish()


if __name__ == "__main__":
    main()
