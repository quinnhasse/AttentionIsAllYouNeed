"""Evaluate a saved TransformerLM checkpoint.

Computes token-level cross-entropy and perplexity on the validation
and test splits of WikiText-2.

Usage:
    python evaluate.py --checkpoint checkpoints/best.pt
    python evaluate.py --checkpoint checkpoints/best.pt --split test
"""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from transformer.model import TransformerLM
from data.dataset import get_dataloaders


def perplexity_from_loss(loss: float) -> float:
    """Exponentiate cross-entropy loss to perplexity."""
    return math.exp(loss)


@torch.no_grad()
def compute_perplexity(
    model: TransformerLM,
    loader,
    device: torch.device,
    pad_idx: int = 0,
) -> tuple[float, float]:
    """Run a full evaluation pass and return (loss, perplexity).

    Args:
        model: TransformerLM to evaluate.
        loader: DataLoader yielding (input, target) batches.
        device: Compute device.
        pad_idx: Token id to ignore in loss computation.

    Returns:
        Tuple of (mean_cross_entropy_nats, perplexity).
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction="sum")
    total_loss = 0.0
    total_tokens = 0

    for src, tgt in tqdm(loader, desc="eval", leave=False):
        src, tgt = src.to(device), tgt.to(device)
        logits = model(src)
        B, T, V = logits.shape
        loss = criterion(logits.view(B * T, V), tgt.view(B * T))
        n_tokens = (tgt != pad_idx).sum().item()
        total_loss += loss.item()
        total_tokens += n_tokens

    mean_loss = total_loss / max(total_tokens, 1)
    return mean_loss, perplexity_from_loss(mean_loss)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate TransformerLM checkpoint")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    p.add_argument(
        "--split",
        type=str,
        default="both",
        choices=["val", "test", "both"],
        help="Which split to evaluate",
    )
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output", type=str, default=None, help="Save JSON results to this path")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Load checkpoint and rebuild model
    ckpt = torch.load(args.checkpoint, map_location=device)
    saved_args = ckpt["args"]

    train_loader, val_loader, test_loader, tokenizer = get_dataloaders(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=saved_args.get("vocab_size", 8000),
    )
    vocab_size = tokenizer.get_vocab_size()

    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=saved_args["d_model"],
        n_heads=saved_args["n_heads"],
        n_layers=saved_args["n_layers"],
        d_ff=saved_args["d_ff"],
        max_len=saved_args["max_len"],
        dropout=0.0,  # No dropout at eval time
    ).to(device)
    model.load_state_dict(ckpt["model"])

    results: dict[str, float] = {}

    if args.split in ("val", "both"):
        val_loss, val_ppl = compute_perplexity(model, val_loader, device)
        results["val_loss"] = round(val_loss, 4)
        results["val_perplexity"] = round(val_ppl, 2)
        print(f"val  loss: {val_loss:.4f}  perplexity: {val_ppl:.2f}")

    if args.split in ("test", "both"):
        test_loss, test_ppl = compute_perplexity(model, test_loader, device)
        results["test_loss"] = round(test_loss, 4)
        results["test_perplexity"] = round(test_ppl, 2)
        print(f"test loss: {test_loss:.4f}  perplexity: {test_ppl:.2f}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"saved to {args.output}")


if __name__ == "__main__":
    main()
