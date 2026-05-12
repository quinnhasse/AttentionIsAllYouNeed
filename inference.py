"""Generate text from a trained TransformerLM checkpoint.

Usage:
    python inference.py --checkpoint checkpoints/best.pt --prompt "The history of"
    python inference.py --checkpoint checkpoints/best.pt --prompt "In 1969" \\
        --max_new_tokens 200 --temperature 0.8 --top_k 40
"""

import argparse
from pathlib import Path

import torch

from transformer.model import TransformerLM
from data.dataset import get_dataloaders


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate text with TransformerLM")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--prompt", type=str, default="The")
    p.add_argument("--max_new_tokens", type=int, default=150)
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Sampling temperature. Lower = more deterministic.")
    p.add_argument("--top_k", type=int, default=50,
                   help="Top-k filtering. Set 0 to disable.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    saved_args = ckpt["args"]

    # Rebuild tokenizer (loads from cache)
    _, _, _, tokenizer = get_dataloaders(
        batch_size=1,
        seq_len=16,
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
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model"])

    # Encode prompt
    enc = tokenizer.encode(args.prompt)
    # Strip the BOS/EOS added by post-processor for generation
    prompt_ids = enc.ids
    # Keep only tokens up to max_len - max_new_tokens to avoid overflow
    max_context = saved_args["max_len"] - args.max_new_tokens
    prompt_ids = prompt_ids[:max_context]

    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    generated = model.generate(
        prompt=prompt_tensor,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    output_ids = generated[0].tolist()
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)

    print("--- prompt ---")
    print(args.prompt)
    print("--- generated ---")
    print(output_text)


if __name__ == "__main__":
    main()
