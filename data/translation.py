"""Multi30k DE→EN seq2seq data pipeline.

Downloads the Multi30k dataset via HuggingFace datasets and trains
shared or separate BPE tokenizers. Returns DataLoaders that emit
(src_ids, tgt_ids) batches with padding to the longest sequence in
each batch.

Usage:
    train_loader, val_loader, test_loader, src_tok, tgt_tok = get_translation_dataloaders(
        src_lang="de",
        tgt_lang="en",
        vocab_size=8000,
        batch_size=128,
        max_seq_len=100,
        seed=42,
    )
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

try:
    from datasets import load_dataset
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace
    _HAVE_DEPS = True
except ImportError:
    _HAVE_DEPS = False

CACHE_DIR = Path(".cache/multi30k")

PAD_IDX = 0
UNK_IDX = 1
BOS_IDX = 2
EOS_IDX = 3
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def _build_bpe_tokenizer(texts: list[str], vocab_size: int) -> "Tokenizer":
    """Train a BPE tokenizer on the provided texts.

    Args:
        texts: Training corpus as a list of strings.
        vocab_size: Target vocabulary size including special tokens.

    Returns:
        Trained HuggingFace Tokenizer instance.
    """
    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    return tok


def _encode(tokenizer: "Tokenizer", text: str, max_len: int) -> list[int]:
    """Encode a string to BOS + ids + EOS, truncated to max_len.

    Args:
        tokenizer: Trained BPE tokenizer.
        text: Input string.
        max_len: Maximum sequence length including BOS and EOS.

    Returns:
        List of token ids.
    """
    ids = tokenizer.encode(text).ids
    ids = ids[: max_len - 2]  # leave room for BOS and EOS
    return [BOS_IDX] + ids + [EOS_IDX]


class TranslationDataset(Dataset):
    """Paired source/target dataset for seq2seq translation.

    Each item is (src_ids, tgt_ids) as LongTensors.
    """

    def __init__(
        self,
        src_texts: list[str],
        tgt_texts: list[str],
        src_tokenizer: "Tokenizer",
        tgt_tokenizer: "Tokenizer",
        max_seq_len: int = 100,
    ) -> None:
        assert len(src_texts) == len(tgt_texts)
        self.pairs = [
            (
                torch.tensor(_encode(src_tokenizer, s, max_seq_len), dtype=torch.long),
                torch.tensor(_encode(tgt_tokenizer, t, max_seq_len), dtype=torch.long),
            )
            for s, t in zip(src_texts, tgt_texts)
        ]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.pairs[idx]


def _collate_fn(batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
    """Pad a batch of (src, tgt) pairs to the longest sequence in each side.

    Args:
        batch: List of (src_ids, tgt_ids) tensors.

    Returns:
        Tuple of (src_batch, tgt_batch), each padded with PAD_IDX.
    """
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)
    return src_padded, tgt_padded


def get_translation_dataloaders(
    src_lang: str = "de",
    tgt_lang: str = "en",
    vocab_size: int = 8000,
    batch_size: int = 128,
    max_seq_len: int = 100,
    seed: int = 42,
    num_workers: int = 0,
    cache_dir: Optional[Path] = None,
) -> tuple:
    """Build DataLoaders for Multi30k translation.

    Downloads Multi30k from HuggingFace (bentrevett/multi30k) on first
    call and caches BPE tokenizers under .cache/multi30k/.

    Args:
        src_lang: Source language code ("de" for German).
        tgt_lang: Target language code ("en" for English).
        vocab_size: BPE vocab size for each language.
        batch_size: Sequences per batch.
        max_seq_len: Maximum token sequence length (including BOS/EOS).
        seed: Random seed for DataLoader shuffling.
        num_workers: DataLoader worker processes.
        cache_dir: Override default cache directory.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, src_tokenizer, tgt_tokenizer).
    """
    if not _HAVE_DEPS:
        raise ImportError(
            "Install required packages: pip install datasets tokenizers"
        )

    torch.manual_seed(seed)
    _cache = Path(cache_dir) if cache_dir else CACHE_DIR
    _cache.mkdir(parents=True, exist_ok=True)

    src_tok_path = _cache / f"tokenizer_{src_lang}.json"
    tgt_tok_path = _cache / f"tokenizer_{tgt_lang}.json"

    # Load dataset — bentrevett/multi30k has train/validation/test splits
    raw = load_dataset("bentrevett/multi30k")

    def _texts(split: str, lang: str) -> list[str]:
        return [ex[lang] for ex in raw[split] if ex[lang].strip()]

    train_src = _texts("train", src_lang)
    train_tgt = _texts("train", tgt_lang)
    val_src   = _texts("validation", src_lang)
    val_tgt   = _texts("validation", tgt_lang)
    test_src  = _texts("test", src_lang)
    test_tgt  = _texts("test", tgt_lang)

    # Build or load tokenizers
    if src_tok_path.exists():
        from tokenizers import Tokenizer as _Tok
        src_tokenizer = _Tok.from_file(str(src_tok_path))
    else:
        src_tokenizer = _build_bpe_tokenizer(train_src, vocab_size)
        src_tokenizer.save(str(src_tok_path))

    if tgt_tok_path.exists():
        from tokenizers import Tokenizer as _Tok
        tgt_tokenizer = _Tok.from_file(str(tgt_tok_path))
    else:
        tgt_tokenizer = _build_bpe_tokenizer(train_tgt, vocab_size)
        tgt_tokenizer.save(str(tgt_tok_path))

    # Build datasets
    train_ds = TranslationDataset(train_src, train_tgt, src_tokenizer, tgt_tokenizer, max_seq_len)
    val_ds   = TranslationDataset(val_src,   val_tgt,   src_tokenizer, tgt_tokenizer, max_seq_len)
    test_ds  = TranslationDataset(test_src,  test_tgt,  src_tokenizer, tgt_tokenizer, max_seq_len)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate_fn,
        num_workers=num_workers,
        generator=g,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_fn,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_fn,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader, src_tokenizer, tgt_tokenizer
