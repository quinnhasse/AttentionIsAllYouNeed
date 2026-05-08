"""WikiText-2 data pipeline with BPE tokenization.

Downloads WikiText-2 via the HuggingFace `datasets` library and trains a
BPE tokenizer with the `tokenizers` library. Returns DataLoaders that emit
(input, target) pairs for causal language modeling.

Usage:
    train_loader, val_loader, test_loader, tokenizer = get_dataloaders(
        batch_size=32,
        seq_len=128,
        vocab_size=8000,
        seed=42,
    )
"""

import os
import json
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

try:
    from datasets import load_dataset
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    _HAVE_DEPS = True
except ImportError:
    _HAVE_DEPS = False


CACHE_DIR = Path(".cache/wikitext2")
TOKENIZER_PATH = CACHE_DIR / "tokenizer.json"


def _build_tokenizer(texts: list[str], vocab_size: int) -> "Tokenizer":
    """Train a BPE tokenizer on the given texts.

    Args:
        texts: List of strings to train on.
        vocab_size: Target vocabulary size.

    Returns:
        Trained Tokenizer instance.
    """
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.post_processor = TemplateProcessing(
        single="<bos> $A <eos>",
        special_tokens=[("<bos>", tokenizer.token_to_id("<bos>")),
                        ("<eos>", tokenizer.token_to_id("<eos>"))],
    )
    return tokenizer


def _tokenize_corpus(texts: list[str], tokenizer: "Tokenizer") -> list[int]:
    """Tokenize a list of strings into a flat list of token ids."""
    ids: list[int] = []
    for encoding in tokenizer.encode_batch(texts):
        ids.extend(encoding.ids)
    return ids


class WikiText2Dataset(Dataset):
    """Sliding-window dataset over a flat token sequence.

    Each item is (input_ids, target_ids) where target is input shifted
    left by one position — standard causal language modeling setup.
    """

    def __init__(self, token_ids: list[int], seq_len: int) -> None:
        self.seq_len = seq_len
        self.data = torch.tensor(token_ids, dtype=torch.long)

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        chunk = self.data[idx : idx + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


def get_dataloaders(
    batch_size: int = 32,
    seq_len: int = 128,
    vocab_size: int = 8000,
    seed: int = 42,
    num_workers: int = 0,
    cache_dir: Optional[Path] = None,
) -> tuple["DataLoader[tuple[Tensor, Tensor]]", ...]:
    """Build train/val/test DataLoaders for WikiText-2 causal LM.

    Downloads the dataset on first call and caches the trained tokenizer
    to `.cache/wikitext2/tokenizer.json`.

    Args:
        batch_size: Number of sequences per batch.
        seq_len: Length of each input sequence.
        vocab_size: BPE vocabulary size.
        seed: Random seed for reproducibility.
        num_workers: DataLoader worker processes.
        cache_dir: Override the default cache directory.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, tokenizer).
    """
    if not _HAVE_DEPS:
        raise ImportError(
            "Install 'datasets' and 'tokenizers' to use the data pipeline: "
            "pip install datasets tokenizers"
        )

    torch.manual_seed(seed)
    _cache = Path(cache_dir) if cache_dir else CACHE_DIR
    _cache.mkdir(parents=True, exist_ok=True)
    tok_path = _cache / "tokenizer.json"

    raw = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_texts = [ex["text"] for ex in raw["train"] if ex["text"].strip()]
    val_texts   = [ex["text"] for ex in raw["validation"] if ex["text"].strip()]
    test_texts  = [ex["text"] for ex in raw["test"] if ex["text"].strip()]

    if tok_path.exists():
        from tokenizers import Tokenizer as _Tok
        tokenizer = _Tok.from_file(str(tok_path))
    else:
        tokenizer = _build_tokenizer(train_texts, vocab_size)
        tokenizer.save(str(tok_path))

    train_ids = _tokenize_corpus(train_texts, tokenizer)
    val_ids   = _tokenize_corpus(val_texts, tokenizer)
    test_ids  = _tokenize_corpus(test_texts, tokenizer)

    train_ds = WikiText2Dataset(train_ids, seq_len)
    val_ds   = WikiText2Dataset(val_ids, seq_len)
    test_ds  = WikiText2Dataset(test_ids, seq_len)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, generator=g, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False,
    )

    return train_loader, val_loader, test_loader, tokenizer
