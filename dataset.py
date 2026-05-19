import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from collections import Counter
import spacy


class Vocabulary:
    """
    Simple vocabulary class that maps tokens <-> integer indices.

    Special tokens:
        <unk> = 0   (unknown)
        <pad> = 1   (padding)
        <sos> = 2   (start-of-sequence)
        <eos> = 3   (end-of-sequence)
    """

    UNK_TOKEN = '<unk>';  UNK_IDX = 0
    PAD_TOKEN = '<pad>';  PAD_IDX = 1
    SOS_TOKEN = '<sos>';  SOS_IDX = 2
    EOS_TOKEN = '<eos>';  EOS_IDX = 3

    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.stoi = {
            self.UNK_TOKEN: self.UNK_IDX,
            self.PAD_TOKEN: self.PAD_IDX,
            self.SOS_TOKEN: self.SOS_IDX,
            self.EOS_TOKEN: self.EOS_IDX,
        }
        self.itos = {v: k for k, v in self.stoi.items()}

    def build_from_counter(self, counter: Counter) -> None:
        for token, freq in counter.items():
            if freq >= self.min_freq and token not in self.stoi:
                idx = len(self.stoi)
                self.stoi[token] = idx
                self.itos[idx]   = token

    def __len__(self) -> int:
        return len(self.stoi)

    def lookup_token(self, idx: int) -> str:
        return self.itos.get(idx, self.UNK_TOKEN)

    def lookup_indices(self, tokens: list) -> list:
        return [self.stoi.get(t, self.UNK_IDX) for t in tokens]

    def encode(self, tokens: list) -> list:
        """Add <sos>/<eos> and convert to indices."""
        return [self.SOS_IDX] + self.lookup_indices(tokens) + [self.EOS_IDX]


#   MULTI30K DATASET

class Multi30kDataset(Dataset):
    """
    Multi30k dataset wrapper.

    Loads from HuggingFace, builds vocabularies on the training split,
    and tokenizes all splits with spaCy.

    Usage:
        train_ds = Multi30kDataset('train')
        src_vocab, tgt_vocab = train_ds.src_vocab, train_ds.tgt_vocab
        val_ds   = Multi30kDataset('validation', train_ds.src_vocab, train_ds.tgt_vocab)
        test_ds  = Multi30kDataset('test',       train_ds.src_vocab, train_ds.tgt_vocab)
    """

    def __init__(
        self,
        split:     str        = 'train',
        src_vocab: Vocabulary = None,
        tgt_vocab: Vocabulary = None,
        min_freq:  int        = 2,
    ):
        self.split = split

        # Load spaCy tokenizers (auto-download if missing)
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

        raw = load_dataset('bentrevett/multi30k')
        self.raw_data = raw[split]

        # Build or reuse vocabularies
        if src_vocab is None or tgt_vocab is None:
            self.src_vocab, self.tgt_vocab = self.build_vocab(raw['train'], min_freq=min_freq)
        else:
            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab

        # Tokenize and encode the split
        self.data = self.process_data()

    # Tokenizers

    def tokenize_de(self, text: str) -> list:
        return [tok.text.lower() for tok in self.spacy_de.tokenizer(text)]

    def tokenize_en(self, text: str) -> list:
        return [tok.text.lower() for tok in self.spacy_en.tokenizer(text)]

    # Vocabulary construction

    def build_vocab(self, train_data, min_freq: int = 2):
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

    # Data processing

    def process_data(self) -> list:
        processed = []
        for example in self.raw_data:
            src_indices = self.src_vocab.encode(self.tokenize_de(example['de']))
            tgt_indices = self.tgt_vocab.encode(self.tokenize_en(example['en']))
            processed.append((src_indices, tgt_indices))
        return processed

    # Dataset interface

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        src_indices, tgt_indices = self.data[idx]
        return (
            torch.tensor(src_indices, dtype=torch.long),
            torch.tensor(tgt_indices, dtype=torch.long),
        )


#   COLLATE FUNCTION

def collate_fn(batch, pad_idx: int = Vocabulary.PAD_IDX):
    """
    Collate (src, tgt) tensor pairs into padded batch tensors.

    Returns:
        src_batch : [batch_size, max_src_len]
        tgt_batch : [batch_size, max_tgt_len]
    """
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)
    return src_batch, tgt_batch


#   CONVENIENCE BUILDER

def build_dataloaders(batch_size: int = 128, min_freq: int = 2):
    """
    Build train / val / test DataLoaders for Multi30k.

    Returns:
        train_loader, val_loader, test_loader, src_vocab, tgt_vocab
    """
    print("Loading Multi30k train split and building vocabularies...")
    train_ds  = Multi30kDataset('train', min_freq=min_freq)
    src_vocab = train_ds.src_vocab
    tgt_vocab = train_ds.tgt_vocab
    print(f"Source vocab size : {len(src_vocab)}")
    print(f"Target vocab size : {len(tgt_vocab)}")

    print("Loading validation split...")
    val_ds  = Multi30kDataset('validation', src_vocab=src_vocab, tgt_vocab=tgt_vocab)
    print("Loading test split...")
    test_ds = Multi30kDataset('test',       src_vocab=src_vocab, tgt_vocab=tgt_vocab)

    _collate = lambda batch: collate_fn(batch, pad_idx=Vocabulary.PAD_IDX)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=_collate, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              collate_fn=_collate)
    test_loader  = DataLoader(test_ds,  batch_size=1,          shuffle=False,
                              collate_fn=_collate)

    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab
