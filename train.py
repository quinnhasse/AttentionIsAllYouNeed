"""Train TransformerLM on WikiText-2 with gradient clipping and optional W&B logging.

Usage:
    python train.py
    python train.py --d_model 256 --n_heads 8 --n_layers 4 --epochs 30
    python train.py --wandb  # requires WANDB_API_KEY env var
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

from transformer.model import TransformerLM
from transformer.scheduler import get_noam_scheduler
from data.dataset import get_dataloaders


def label_smoothed_ce(
    logits: Tensor,
    targets: Tensor,
    smoothing: float = 0.1,
    pad_idx: int = 0,
) -> Tensor:
    """Cross-entropy with label smoothing.

    Distributes `smoothing` probability mass uniformly across the
    vocabulary, reserving `1 - smoothing` for the correct label.
    Pad tokens are excluded from the loss.

    Args:
        logits: (batch * seq_len, vocab_size)
        targets: (batch * seq_len,)
        smoothing: Label smoothing epsilon.
        pad_idx: Token id of the padding token.

    Returns:
        Scalar loss tensor.
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


@torch.no_grad()
def evaluate(model: TransformerLM, loader, device: torch.device) -> float:
    """Compute token-level cross-entropy (no label smoothing) on a split.

    Args:
        model: TransformerLM in eval mode.
        loader: DataLoader yielding (input, target) batches.
        device: Device to run on.

    Returns:
        Mean cross-entropy (nats per token).
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    total_loss = 0.0
    total_tokens = 0

    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        logits = model(src)
        B, T, V = logits.shape
        loss = criterion(logits.view(B * T, V), tgt.view(B * T))
        n_tokens = (tgt != 0).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    return total_loss / max(total_tokens, 1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train TransformerLM on WikiText-2")
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--d_ff", type=int, default=1024)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--vocab_size", type=int, default=8000)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--warmup_steps", type=int, default=4000)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--save_dir", type=str, default="checkpoints")
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--wandb", action="store_true", help="Log to W&B (requires WANDB_API_KEY)")
    p.add_argument("--wandb_project", type=str, default="attention-is-all-you-need")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"device: {device}")

    # Data
    print("loading WikiText-2 ...")
    train_loader, val_loader, test_loader, tokenizer = get_dataloaders(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        seed=args.seed,
    )
    vocab_size = tokenizer.get_vocab_size()
    print(f"vocab size: {vocab_size}")

    # Model
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_len=args.max_len,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"parameters: {n_params:,}")

    # Optimizer + scheduler
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = get_noam_scheduler(optimizer, d_model=args.d_model, warmup_steps=args.warmup_steps)

    # W&B (optional)
    wb_run = None
    if args.wandb:
        try:
            import wandb
            wb_run = wandb.init(
                project=args.wandb_project,
                config=vars(args),
                name=f"lm-d{args.d_model}-h{args.n_heads}-l{args.n_layers}",
            )
        except Exception as exc:
            print(f"W&B init failed: {exc}. Continuing without W&B.")

    # Directories
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)

    log: list[dict] = []
    best_val_loss = float("inf")
    step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0
        t0 = time.time()

        for src, tgt in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            src, tgt = src.to(device), tgt.to(device)

            optimizer.zero_grad()
            logits = model(src)
            B, T, V = logits.shape
            loss = label_smoothed_ce(logits.view(B * T, V), tgt.view(B * T))
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1

            n_tokens = (tgt != 0).sum().item()
            epoch_loss += loss.item() * n_tokens
            epoch_tokens += n_tokens

        train_loss = epoch_loss / max(epoch_tokens, 1)
        val_loss = evaluate(model, val_loader, device)
        val_ppl = math.exp(val_loss)
        elapsed = time.time() - t0

        entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_perplexity": round(val_ppl, 2),
            "lr": scheduler.get_last_lr()[0],
            "elapsed_s": round(elapsed, 1),
        }
        log.append(entry)
        print(
            f"epoch {epoch:>3} | train {train_loss:.4f} | val {val_loss:.4f} "
            f"| ppl {val_ppl:.1f} | {elapsed:.1f}s"
        )

        if wb_run:
            wb_run.log(entry)

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "args": vars(args)},
                save_dir / "best.pt",
            )

        # Save log after every epoch
        with open(results_dir / "training_log.json", "w") as f:
            json.dump(log, f, indent=2)

    # Final test evaluation
    best_ckpt = torch.load(save_dir / "best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model"])
    test_loss = evaluate(model, test_loader, device)
    test_ppl = math.exp(test_loss)

    summary = {
        "best_val_loss": round(best_val_loss, 4),
        "best_val_perplexity": round(math.exp(best_val_loss), 2),
        "test_loss": round(test_loss, 4),
        "test_perplexity": round(test_ppl, 2),
        "n_params": n_params,
        "epochs": args.epochs,
    }
    print(f"\nbest val ppl:  {math.exp(best_val_loss):.1f}")
    print(f"test ppl:      {test_ppl:.1f}")

    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if wb_run:
        wb_run.summary.update(summary)
        wb_run.finish()


if __name__ == "__main__":
    main()
