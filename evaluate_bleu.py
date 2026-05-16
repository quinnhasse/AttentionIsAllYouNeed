"""Evaluate a saved checkpoint on Multi30k test set and report BLEU.

Usage:
    python evaluate_bleu.py checkpoints/best.pt
    python evaluate_bleu.py checkpoints/best.pt --beam 4 --alpha 0.6
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data.translation import (
    get_translation_dataloaders,
    BOS_IDX,
    EOS_IDX,
    PAD_IDX,
)
from transformer.model import Transformer


def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    max_len: int = 100,
) -> list[int]:
    """Decode a single source sequence with greedy search.

    Args:
        model: Trained Transformer in eval mode.
        src: Source token ids of shape (1, src_len).
        max_len: Maximum output sequence length.

    Returns:
        List of generated token ids (excluding BOS).
    """
    device = src.device
    model.eval()
    with torch.no_grad():
        src_mask = model.make_src_mask(src)
        memory = model.encode(src, src_mask)
        tgt = torch.tensor([[BOS_IDX]], dtype=torch.long, device=device)
        for _ in range(max_len - 1):
            tgt_mask = model.make_tgt_mask(tgt)
            out = model.decode(tgt, memory, tgt_mask, src_mask)
            logits = model.output_proj(out[:, -1, :])
            next_tok = logits.argmax(dim=-1, keepdim=True)
            tgt = torch.cat([tgt, next_tok], dim=1)
            if next_tok.item() == EOS_IDX:
                break
    return tgt[0, 1:].tolist()  # strip BOS


def beam_decode(
    model: Transformer,
    src: torch.Tensor,
    beam_size: int = 4,
    alpha: float = 0.6,
    max_len: int = 100,
) -> list[int]:
    """Decode a single source sequence with beam search.

    Uses length normalization: score = log_prob / length^alpha.

    Args:
        model: Trained Transformer in eval mode.
        src: Source token ids of shape (1, src_len).
        beam_size: Number of beams to maintain.
        alpha: Length normalization exponent.
        max_len: Maximum sequence length.

    Returns:
        Token ids of the highest-scoring complete beam.
    """
    device = src.device
    model.eval()
    with torch.no_grad():
        src_mask = model.make_src_mask(src)
        memory = model.encode(src, src_mask)
        # memory: (1, src_len, d_model) — expand for beams
        memory = memory.expand(beam_size, -1, -1)
        src_mask = src_mask.expand(beam_size, -1, -1, -1)

        # Each beam: (token_ids, log_prob, done)
        beams: list[tuple[list[int], float, bool]] = [
            ([BOS_IDX], 0.0, False)
        ]
        completed: list[tuple[list[int], float]] = []

        for _ in range(max_len):
            if all(b[2] for b in beams):
                break

            candidates: list[tuple[list[int], float]] = []
            for ids, log_p, done in beams:
                if done:
                    candidates.append((ids, log_p))
                    continue
                tgt = torch.tensor([ids], dtype=torch.long, device=device)
                tgt_mask = model.make_tgt_mask(tgt)
                # Use only first memory slice for single beam (batch=1)
                out = model.decode(
                    tgt,
                    memory[:1],
                    tgt_mask,
                    src_mask[:1],
                )
                log_probs = torch.log_softmax(model.output_proj(out[:, -1, :]), dim=-1)
                top_lp, top_ids = log_probs[0].topk(beam_size)
                for lp, tok in zip(top_lp.tolist(), top_ids.tolist()):
                    new_ids = ids + [tok]
                    new_lp = log_p + lp
                    is_done = tok == EOS_IDX
                    if is_done:
                        # Normalize by length before storing
                        length = len(new_ids) - 1  # exclude BOS
                        norm = (5 + length) ** alpha / (5 + 1) ** alpha
                        completed.append((new_ids, new_lp / norm))
                    else:
                        candidates.append((new_ids, new_lp))

            # Keep top beam_size non-completed beams
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = [
                (ids, lp, False)
                for ids, lp in candidates[:beam_size]
            ]

        if not completed and beams:
            best_ids, best_lp = beams[0][0], beams[0][1]
        elif completed:
            best_ids, _ = max(completed, key=lambda x: x[1])
        else:
            best_ids = [BOS_IDX, EOS_IDX]

    return best_ids[1:]  # strip BOS


@torch.no_grad()
def evaluate(
    model: Transformer,
    loader,
    tgt_tokenizer,
    device: torch.device,
    beam_size: int = 1,
    alpha: float = 0.6,
    max_gen_len: int = 100,
) -> dict:
    """Run evaluation loop, return BLEU and token-level CE loss.

    Args:
        model: Trained Transformer.
        loader: DataLoader for the evaluation split.
        tgt_tokenizer: Target-language tokenizer.
        device: Inference device.
        beam_size: 1 for greedy, >1 for beam search.
        alpha: Length normalization exponent (beam search only).
        max_gen_len: Maximum generated sequence length.

    Returns:
        Dict with keys bleu, loss.
    """
    try:
        import sacrebleu
    except ImportError as err:
        raise ImportError("Install sacrebleu: pip install sacrebleu") from err

    import torch.nn as nn
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    total_loss, total_tokens = 0.0, 0
    hypotheses: list[str] = []
    references: list[str] = []

    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)

        # Loss (teacher-forced)
        tgt_input  = tgt[:, :-1]
        tgt_target = tgt[:, 1:]
        logits = model(src, tgt_input)
        B, T, V = logits.shape
        loss = criterion(logits.reshape(B * T, V), tgt_target.reshape(B * T))
        n_tok = (tgt_target != PAD_IDX).sum().item()
        total_loss += loss.item() * n_tok
        total_tokens += n_tok

        # Generation
        for i in range(src.size(0)):
            src_i = src[i].unsqueeze(0)
            if beam_size == 1:
                ids = greedy_decode(model, src_i, max_len=max_gen_len)
            else:
                ids = beam_decode(model, src_i, beam_size=beam_size, alpha=alpha, max_len=max_gen_len)
            hyp = tgt_tokenizer.decode(ids)
            hypotheses.append(hyp)

            ref_ids = tgt[i].tolist()
            ref_ids = [t for t in ref_ids if t not in (PAD_IDX, BOS_IDX, EOS_IDX)]
            ref = tgt_tokenizer.decode(ref_ids)
            references.append(ref)

    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    ce = total_loss / max(total_tokens, 1)
    return {"bleu": round(bleu.score, 2), "loss": round(ce, 4)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a checkpoint on Multi30k test set")
    p.add_argument("checkpoint", type=str, help="Path to .pt checkpoint file")
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--beam", type=int, default=4, help="Beam size (1 = greedy)")
    p.add_argument("--alpha", type=float, default=0.6, help="Length normalization exponent")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_gen_len", type=int, default=100)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg_dict = ckpt.get("cfg", {})

    # Reconstruct data loaders to get tokenizers
    model_cfg = cfg_dict.get("model", {})
    data_cfg  = cfg_dict.get("data", {})

    train_loader, val_loader, test_loader, src_tok, tgt_tok = get_translation_dataloaders(
        src_lang=data_cfg.get("src_lang", "de"),
        tgt_lang=data_cfg.get("tgt_lang", "en"),
        vocab_size=data_cfg.get("vocab_size", 8000),
        batch_size=args.batch_size,
        max_seq_len=data_cfg.get("max_seq_len", 100),
    )

    src_vocab = src_tok.get_vocab_size()
    tgt_vocab = tgt_tok.get_vocab_size()

    model = Transformer(
        src_vocab_size=src_vocab,
        tgt_vocab_size=tgt_vocab,
        d_model=model_cfg.get("d_model", 512),
        n_heads=model_cfg.get("n_heads", 8),
        n_layers=model_cfg.get("n_layers", 6),
        d_ff=model_cfg.get("d_ff", 2048),
        max_len=model_cfg.get("max_len", 256),
        dropout=0.0,
        pad_idx=PAD_IDX,
        use_pe=model_cfg.get("use_pe", True),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    loader = test_loader if args.split == "test" else val_loader
    results = evaluate(
        model,
        loader,
        tgt_tok,
        device,
        beam_size=args.beam,
        alpha=args.alpha,
        max_gen_len=args.max_gen_len,
    )
    results["split"] = args.split
    results["beam"] = args.beam
    results["checkpoint"] = args.checkpoint

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
