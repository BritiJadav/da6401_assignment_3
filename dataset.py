"""
dataset.py — Multi30k Dataset Loading and Preprocessing
DA6401 Assignment 3: "Attention Is All You Need"

Loads Multi30k (de→en) from HuggingFace, tokenizes with spaCy,
builds vocabularies, and provides a collate_fn for DataLoader usage.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from collections import Counter
import spacy


# ══════════════════════════════════════════════════════════════════════
#   VOCABULARY
# ══════════════════════════════════════════════════════════════════════

class Vocabulary:
    """
    Simple vocabulary class that maps tokens ↔ integer indices.

    Special tokens:
        <unk> : unknown token   (index 0)
        <pad> : padding token   (index 1)
        <sos> : start of seq    (index 2)
        <eos> : end of seq      (index 3)
    """

    PAD_TOKEN = '<pad>'
    UNK_TOKEN = '<unk>'
    SOS_TOKEN = '<sos>'
    EOS_TOKEN = '<eos>'

    PAD_IDX = 1
    UNK_IDX = 0
    SOS_IDX = 2
    EOS_IDX = 3

    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.stoi = {
            self.UNK_TOKEN: self.UNK_IDX,
            self.PAD_TOKEN: self.PAD_IDX,
            self.SOS_TOKEN: self.SOS_IDX,
            self.EOS_TOKEN: self.EOS_IDX,
        }
        self.itos = {v: k for k, v in self.stoi.items()}

    def build_from_counter(self, counter: Counter):
        """Add tokens from a Counter that meet min_freq threshold."""
        for token, freq in counter.items():
            if freq >= self.min_freq and token not in self.stoi:
                idx = len(self.stoi)
                self.stoi[token] = idx
                self.itos[idx] = token

    def __len__(self):
        return len(self.stoi)

    def lookup_token(self, idx: int) -> str:
        return self.itos.get(idx, self.UNK_TOKEN)

    def lookup_indices(self, tokens) -> list:
        return [self.stoi.get(t, self.UNK_IDX) for t in tokens]

    def encode(self, tokens: list) -> list:
        """Add <sos>/<eos> and convert to indices."""
        return (
            [self.SOS_IDX]
            + self.lookup_indices(tokens)
            + [self.EOS_IDX]
        )


# ══════════════════════════════════════════════════════════════════════
#   MULTI30K DATASET
# ══════════════════════════════════════════════════════════════════════

class Multi30kDataset(Dataset):
    """
    Multi30k dataset wrapper.

    Loads the dataset from HuggingFace, builds vocabularies on the
    training split, and tokenizes all splits with spaCy.

    Args:
        split        (str)  : 'train', 'validation', or 'test'
        src_vocab    (Vocabulary): Pre-built source vocab (optional).
        tgt_vocab    (Vocabulary): Pre-built target vocab (optional).
        min_freq     (int)  : Minimum token frequency for vocabulary (default 2).

    Usage:
        train_ds = Multi30kDataset('train')
        src_vocab, tgt_vocab = train_ds.src_vocab, train_ds.tgt_vocab
        val_ds   = Multi30kDataset('validation', train_ds.src_vocab, train_ds.tgt_vocab)
        test_ds  = Multi30kDataset('test',       train_ds.src_vocab, train_ds.tgt_vocab)
    """

    def __init__(
        self,
        split: str = 'train',
        src_vocab: Vocabulary = None,
        tgt_vocab: Vocabulary = None,
        min_freq: int = 2,
    ):
        self.split = split

        # ── Load spaCy tokenizers ───────────────────────────────────
        # Requires: python -m spacy download de_core_news_sm
        #           python -m spacy download en_core_web_sm
        try:
            self.spacy_de = spacy.load('de_core_news_sm')
        except OSError:
            import subprocess, sys
            subprocess.run([sys.executable, '-m', 'spacy', 'download', 'de_core_news_sm'], check=True)
            self.spacy_de = spacy.load('de_core_news_sm')

        try:
            self.spacy_en = spacy.load('en_core_web_sm')
        except OSError:
            import subprocess, sys
            subprocess.run([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'], check=True)
            self.spacy_en = spacy.load('en_core_web_sm')

        # ── Load dataset ─────────────────────────────────────────────
        raw = load_dataset('bentrevett/multi30k', trust_remote_code=True)
        self.raw_data = raw[split]

        # ── Build or reuse vocabularies ───────────────────────────────
        if src_vocab is None or tgt_vocab is None:
            # Build from training data
            self.src_vocab, self.tgt_vocab = self.build_vocab(
                raw['train'], min_freq=min_freq
            )
        else:
            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab

        # ── Tokenize and encode the split ─────────────────────────────
        self.data = self.process_data()

    # ── Tokenizers ───────────────────────────────────────────────────

    def tokenize_de(self, text: str):
        return [tok.text.lower() for tok in self.spacy_de.tokenizer(text)]

    def tokenize_en(self, text: str):
        return [tok.text.lower() for tok in self.spacy_en.tokenizer(text)]

    # ── Vocabulary construction ───────────────────────────────────────

    def build_vocab(self, train_data, min_freq: int = 2):
        """
        Builds vocabulary mappings for src (de) and tgt (en).

        Returns:
            src_vocab (Vocabulary), tgt_vocab (Vocabulary)
        """
        src_counter = Counter()
        tgt_counter = Counter()

        for example in train_data:
            src_counter.update(self.tokenize_de(example['de']))
            tgt_counter.update(self.tokenize_en(example['en']))

        src_vocab = Vocabulary(min_freq=min_freq)
        tgt_vocab = Vocabulary(min_freq=min_freq)

        src_vocab.build_from_counter(src_counter)
        tgt_vocab.build_from_counter(tgt_counter)

        return src_vocab, tgt_vocab

    # ── Data processing ───────────────────────────────────────────────

    def process_data(self):
        """
        Tokenize and convert sentences to integer index lists.

        Returns:
            list of (src_indices, tgt_indices) tuples (as plain Python lists)
        """
        processed = []
        for example in self.raw_data:
            src_tokens = self.tokenize_de(example['de'])
            tgt_tokens = self.tokenize_en(example['en'])

            src_indices = self.src_vocab.encode(src_tokens)
            tgt_indices = self.tgt_vocab.encode(tgt_tokens)

            processed.append((src_indices, tgt_indices))
        return processed

    # ── Dataset interface ─────────────────────────────────────────────

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_indices, tgt_indices = self.data[idx]
        return (
            torch.tensor(src_indices, dtype=torch.long),
            torch.tensor(tgt_indices, dtype=torch.long),
        )


# ══════════════════════════════════════════════════════════════════════
#   COLLATE FUNCTION
# ══════════════════════════════════════════════════════════════════════

def collate_fn(batch, pad_idx: int = Vocabulary.PAD_IDX):
    """
    Collate a list of (src, tgt) tensor pairs into padded batches.

    Args:
        batch   : list of (src_tensor, tgt_tensor)
        pad_idx : index used for padding (default 1)

    Returns:
        src_batch : [batch_size, max_src_len]
        tgt_batch : [batch_size, max_tgt_len]
    """
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)
    return src_batch, tgt_batch


# ══════════════════════════════════════════════════════════════════════
#   CONVENIENCE BUILDER
# ══════════════════════════════════════════════════════════════════════

def build_dataloaders(batch_size: int = 128, min_freq: int = 2):
    """
    Build train / val / test DataLoaders for Multi30k.

    Returns:
        train_loader, val_loader, test_loader, src_vocab, tgt_vocab
    """
    print("Loading Multi30k train split and building vocabularies...")
    train_ds = Multi30kDataset('train', min_freq=min_freq)
    src_vocab = train_ds.src_vocab
    tgt_vocab = train_ds.tgt_vocab

    print(f"Source vocab size: {len(src_vocab)}")
    print(f"Target vocab size: {len(tgt_vocab)}")

    print("Loading validation split...")
    val_ds = Multi30kDataset('validation', src_vocab=src_vocab, tgt_vocab=tgt_vocab)

    print("Loading test split...")
    test_ds = Multi30kDataset('test', src_vocab=src_vocab, tgt_vocab=tgt_vocab)

    _collate = lambda batch: collate_fn(batch, pad_idx=Vocabulary.PAD_IDX)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=_collate, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=_collate
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        collate_fn=_collate
    )

    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab
