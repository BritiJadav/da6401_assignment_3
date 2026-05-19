import os
import sys
import math
import copy
import subprocess
from collections import Counter
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F



class Vocabulary:
    """
    Token <-> integer index mapping.

    Special tokens (fixed indices):
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
        self.stoi: dict = {
            self.UNK_TOKEN: self.UNK_IDX,
            self.PAD_TOKEN: self.PAD_IDX,
            self.SOS_TOKEN: self.SOS_IDX,
            self.EOS_TOKEN: self.EOS_IDX,
        }
        self.itos: dict = {v: k for k, v in self.stoi.items()}

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
        """Wrap token list with <sos>/<eos> and convert to indices."""
        return [self.SOS_IDX] + self.lookup_indices(tokens) + [self.EOS_IDX]


#  SCALED DOT-PRODUCT ATTENTION


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product Attention.

        Attention(Q, K, V) = softmax( Q*K^t / sqrtd_k ) * V

    Args:
        Q    : shape (..., seq_q, d_k)
        K    : shape (..., seq_k, d_k)
        V    : shape (..., seq_k, d_v)
        mask : BoolTensor broadcastable to (..., seq_q, seq_k).
               True positions are MASKED OUT (-> -inf before softmax).

    Returns:
        output : shape (..., seq_q, d_v)
        attn_w : shape (..., seq_q, seq_k)  - weights sum to 1 over key dim
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))

    attn_w = F.softmax(scores, dim=-1)
    attn_w = torch.nan_to_num(attn_w, nan=0.0)   # guard against all-masked rows
    output = torch.matmul(attn_w, V)
    return output, attn_w



#  MASK HELPERS

def make_src_mask(src: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    """
    Padding mask for the encoder.

    Args:
        src     : [batch, src_len]
        pad_idx : <pad> index (default 1)
    Returns:
        BoolTensor [batch, 1, 1, src_len]   True -> PAD (masked out)
    """
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    """
    Combined padding + causal (look-ahead) mask for the decoder.

    Args:
        tgt     : [batch, tgt_len]
        pad_idx : <pad> index (default 1)
    Returns:
        BoolTensor [batch, 1, tgt_len, tgt_len]   True -> masked out
    """
    tgt_len  = tgt.size(1)
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)           # [B,1,1,T]
    causal   = torch.triu(
        torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=tgt.device),
        diagonal=1,
    ).unsqueeze(0).unsqueeze(0)                                      # [1,1,T,T]
    return pad_mask | causal


#  MULTI-HEAD ATTENTION

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention - $3.2.2 of "Attention Is All You Need".
    Implemented from scratch; torch.nn.MultiheadAttention is NOT used.

    Args:
        d_model   : total model dim  (must be divisible by num_heads)
        num_heads : number of attention heads
        dropout   : dropout on projected attention output
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

        # Stored for W&B attention-map visualisations
        self.attn_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query / key / value : [batch, seq, d_model]
            mask : BoolTensor broadcastable to [batch, num_heads, seq_q, seq_k]
        Returns:
            [batch, seq_q, d_model]
        """
        B = query.size(0)

        def project_and_split(linear, x):
            return linear(x).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)

        Q = project_and_split(self.W_q, query)
        K = project_and_split(self.W_k, key)
        V = project_and_split(self.W_v, value)

        out, attn_w = scaled_dot_product_attention(Q, K, V, mask)
        self.attn_weights = attn_w.detach()          # save for visualisation

        out = self.dropout(out)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.W_o(out)


#  POSITIONAL ENCODING

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding - $3.5.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Registered as a buffer (not a trainable parameter).
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))   # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : [batch, seq_len, d_model]  ->  same shape"""
        return self.dropout(x + self.pe[:, :x.size(1), :])



#  POSITION-WISE FEED-FORWARD NETWORK


class PositionwiseFeedForward(nn.Module):
    """FFN(x) = max(0, xW1+b1)W2+b2  - $3.3"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))



#  ENCODER LAYER  (Post-LayerNorm as in the original paper)

class EncoderLayer(nn.Module):
    """x -> [MHA -> Add&Norm] -> [FFN -> Add&Norm]"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn       = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, src_mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


#  DECODER LAYER  (Post-LayerNorm)

class DecoderLayer(nn.Module):
    """x -> [Masked MHA -> A&N] -> [Cross-MHA -> A&N] -> [FFN -> A&N]"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn        = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.norm3      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(p=dropout)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, memory, memory, src_mask)))
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x



#  ENCODER / DECODER STACKS

class Encoder(nn.Module):
    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.norm1.normalized_shape[0])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.norm1.normalized_shape[0])

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


#  GDRIVE CONFIG

GDRIVE_FILE_ID      = "1CYFiCWmBSz7nL6nWWQF0XtrhgD30ODfs"  
CHECKPOINT_FILENAME = "best_checkpoint.pt"



#  FULL TRANSFORMER

class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for German->English NMT.

    ALL arguments have defaults so the autograder can call:
        model = Transformer().to(device)
        model.eval()
        english = model.infer(german_sentence)

    __init__ automatically:
        1. Loads spaCy tokenisers (de_core_news_sm, en_core_web_sm)
        2. Builds src/tgt vocabularies from Multi30k training split
        3. Constructs all nn layers using the derived vocab sizes
        4. Downloads trained weights from Google Drive via gdown
        5. Loads those weights into the model

    Args:
        d_model        (int)  : Model dimensionality            (default 512)
        N              (int)  : Encoder/decoder stack depth     (default 6)
        num_heads      (int)  : Attention heads                 (default 8)
        d_ff           (int)  : FFN inner dimensionality        (default 2048)
        dropout        (float): Dropout probability             (default 0.1)
        max_infer_len  (int)  : Max tokens to generate          (default 30)
        weights_path   (str)  : Local path for checkpoint file
        min_freq       (int)  : Vocab minimum token frequency   (default 2)
    """

    def __init__(
        self,
        d_model:        int   = 256,
        N:              int   = 3,
        num_heads:      int   = 8,
        d_ff:           int   = 512,
        dropout:        float = 0.3,
        max_infer_len:  int   = 30,
        weights_path:   str   = CHECKPOINT_FILENAME,
        min_freq:       int   = 2,
    ) -> None:

        #  STEP 1: store inference hyper-params BEFORE super().__init__
        #    so they exist as plain attributes immediately
        self._d_model_val    = d_model
        self._max_infer_len  = max_infer_len
        self._weights_path   = weights_path

        #  STEP 2: load spaCy tokenisers 
        print("[Transformer] Loading spaCy tokenisers...")
        self._spacy_de = self._load_spacy_model('de_core_news_sm')
        self._spacy_en = self._load_spacy_model('en_core_web_sm')

        #  STEP 3: build vocabularies from Multi30k train split 
        print("[Transformer] Building vocabularies from Multi30k...")
        self.src_vocab, self.tgt_vocab = self._build_vocabs(min_freq)
        src_vocab_size = len(self.src_vocab)
        tgt_vocab_size = len(self.tgt_vocab)
        print(f"[Transformer] src vocab={src_vocab_size}  tgt vocab={tgt_vocab_size}")

        #  STEP 4: build nn architecture 
        super().__init__()
        self.d_model = d_model

        self.src_embedding     = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding     = nn.Embedding(tgt_vocab_size, d_model)
        self.src_pos_enc       = PositionalEncoding(d_model, dropout)
        self.tgt_pos_enc       = PositionalEncoding(d_model, dropout)
        self.encoder           = Encoder(EncoderLayer(d_model, num_heads, d_ff, dropout), N)
        self.decoder           = Decoder(DecoderLayer(d_model, num_heads, d_ff, dropout), N)
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        # Weight tying: target embedding ↔ output projection
        self.output_projection.weight = self.tgt_embedding.weight

        # Xavier uniform initialisation
        self._init_parameters()

        #  STEP 5: download & load trained weights 
        self._download_and_load_weights()

    
    #  PRIVATE HELPERS

    @staticmethod
    def _load_spacy_model(model_name: str):
        """Load a spaCy model, auto-downloading if not installed."""
        import spacy
        try:
            return spacy.load(model_name)
        except OSError:
            print(f"[Transformer] Downloading spaCy model: {model_name}")
            subprocess.run(
                [sys.executable, '-m', 'spacy', 'download', model_name],
                check=True, capture_output=True,
            )
            return spacy.load(model_name)

    def _tokenize_de(self, text: str) -> list:
        return [tok.text.lower() for tok in self._spacy_de.tokenizer(text)]

    def _tokenize_en(self, text: str) -> list:
        return [tok.text.lower() for tok in self._spacy_en.tokenizer(text)]

    def _build_vocabs(self, min_freq: int = 2) -> Tuple[Vocabulary, Vocabulary]:
        """
        Build src (de) and tgt (en) vocabularies from Multi30k train split.
        """
        from datasets import load_dataset
        train_data = load_dataset('bentrevett/multi30k')['train']

        src_counter: Counter = Counter()
        tgt_counter: Counter = Counter()
        for example in train_data:
            src_counter.update(self._tokenize_de(example['de']))
            tgt_counter.update(self._tokenize_en(example['en']))

        src_vocab = Vocabulary(min_freq=min_freq)
        tgt_vocab = Vocabulary(min_freq=min_freq)
        src_vocab.build_from_counter(src_counter)
        tgt_vocab.build_from_counter(tgt_counter)
        return src_vocab, tgt_vocab

    def _download_and_load_weights(self) -> None:
        """
        Download checkpoint from Google Drive via gdown (if not cached),
        then load the state-dict into this model.
        """
        if GDRIVE_FILE_ID == "YOUR_GDRIVE_FILE_ID_HERE":
            print(
                "[Transformer] WARNING: GDRIVE_FILE_ID is not set.\n"
                "  Edit GDRIVE_FILE_ID at the top of model.py before submission.\n"
                "  Running with random weights - BLEU will be 0."
            )
            return

        # Download only when file is absent
        if not os.path.exists(self._weights_path):
            print(f"[Transformer] Downloading weights -> '{self._weights_path}' ...")
            try:
                import gdown
            except ImportError:
                print("[Transformer] Installing gdown...")
                subprocess.run(
                    [sys.executable, '-m', 'pip', 'install', 'gdown', '-q'],
                    check=True,
                )
                import gdown

            url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
            gdown.download(url, self._weights_path, quiet=False)
        else:
            print(f"[Transformer] Found cached weights at '{self._weights_path}'")

        # Load state dict — support both raw dict and our save_checkpoint format
        checkpoint = torch.load(
            self._weights_path,
            map_location='cpu',
            weights_only=False,
        )
        state_dict = (
            checkpoint['model_state_dict']
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint
            else checkpoint
        )
        self.load_state_dict(state_dict, strict=True)
        print("[Transformer] Weights loaded successfully")

    def _init_parameters(self) -> None:
        """Xavier uniform init for all weight matrices (dim > 1)."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    
    #  CORE FORWARD METHODS

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Run the encoder stack.

        Args:
            src      : token indices  [batch, src_len]
            src_mask : padding mask   [batch, 1, 1, src_len]
        Returns:
            memory   : [batch, src_len, d_model]
        """
        x = self.src_pos_enc(self.src_embedding(src) * math.sqrt(self.d_model))
        return self.encoder(x, src_mask)

    def decode(
        self,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt:      torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the decoder stack and project to vocabulary logits.

        Args:
            memory   : encoder output  [batch, src_len, d_model]
            src_mask :                 [batch, 1, 1, src_len]
            tgt      : token indices   [batch, tgt_len]
            tgt_mask :                 [batch, 1, tgt_len, tgt_len]
        Returns:
            logits   : [batch, tgt_len, tgt_vocab_size]
        """
        x = self.tgt_pos_enc(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        return self.output_projection(self.decoder(x, memory, src_mask, tgt_mask))

    def forward(
        self,
        src:      torch.Tensor,
        tgt:      torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full encoder-decoder forward pass (used during training).

        Args:
            src      : [batch, src_len]
            tgt      : [batch, tgt_len]
            src_mask : [batch, 1, 1, src_len]
            tgt_mask : [batch, 1, tgt_len, tgt_len]
        Returns:
            logits   : [batch, tgt_len, tgt_vocab_size]
        """
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    
    #  END-TO-END INFERENCE
    

    def infer(self, german_sentence: str) -> str:
        """
        Translate a single German sentence to English, end-to-end.

        Pipeline:
            1. Tokenise German input with spaCy (de_core_news_sm)
            2. Convert tokens -> integer indices via src_vocab
            3. Build source padding mask
            4. Encode with Transformer encoder
            5. Autoregressively generate English tokens (greedy decoding)
            6. Convert predicted indices -> English string (detokenise)

        Args:
            german_sentence (str): Raw German text, e.g.
                "Ein Hund rennt durch das Gras."
        Returns:
            english_sentence (str): e.g. "a dog runs through the grass ."
        """
        self.eval()
        device = next(self.parameters()).device

        PAD = Vocabulary.PAD_IDX
        SOS = Vocabulary.SOS_IDX
        EOS = Vocabulary.EOS_IDX

        # 1. Tokenise
        tokens = self._tokenize_de(german_sentence)

        # 2. Encode tokens -> tensor [1, src_len]
        src = torch.tensor(
            self.src_vocab.encode(tokens),
            dtype=torch.long, device=device,
        ).unsqueeze(0)

        # 3. Source mask [1, 1, 1, src_len]
        src_mask = make_src_mask(src, pad_idx=PAD).to(device)

        with torch.no_grad():
            # 4. Encode
            memory = self.encode(src, src_mask)

            # 5. Greedy autoregressive decode
            ys = torch.tensor([[SOS]], dtype=torch.long, device=device)

            for _ in range(self._max_infer_len - 1):
                tgt_mask = make_tgt_mask(ys, pad_idx=PAD).to(device)
                logits   = self.decode(memory, src_mask, ys, tgt_mask)
                next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                ys = torch.cat([ys, next_tok], dim=1)
                if next_tok.item() == EOS:
                    break

        # 6. Detokenise - strip all special tokens
        skip = {PAD, SOS, EOS, Vocabulary.UNK_IDX}
        words = []
        for idx in ys[0].tolist():
            if idx == EOS:
                break
            if idx not in skip:
                words.append(self.tgt_vocab.lookup_token(idx))

        return ' '.join(words)
