"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""

import os
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from typing import Optional
from tqdm import tqdm

from model import Transformer, make_src_mask, make_tgt_mask
from dataset import build_dataloaders, Vocabulary
from lr_scheduler import NoamScheduler

# Try importing evaluate for BLEU; fall back to sacrebleu
try:
    import evaluate as hf_evaluate
    _HF_EVALUATE = True
except ImportError:
    _HF_EVALUATE = False

try:
    import sacrebleu as _sacrebleu
    _SACREBLEU = True
except ImportError:
    _SACREBLEU = False


# ══════════════════════════════════════════════════════════════════════
#   LABEL SMOOTHING LOSS
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need".

    Smoothed target distribution:
        y_smooth = (1 - eps) * one_hot(y) + eps / (vocab_size - 1)

    Positions corresponding to pad_idx receive 0 probability mass and are
    excluded from the loss computation.

    Args:
        vocab_size (int)  : Number of output classes.
        pad_idx    (int)  : Index of <pad> token — receives 0 probability.
        smoothing  (float): Smoothing factor ε (default 0.1).
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx    = pad_idx
        self.smoothing  = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : shape [batch * tgt_len, vocab_size]  (raw model output)
            target : shape [batch * tgt_len]              (gold token indices)

        Returns:
            Scalar loss value (mean over non-pad positions).
        """
        # Build smoothed target distribution
        # All classes share eps / (V-1) except the true class and pad
        smooth_val = self.smoothing / (self.vocab_size - 2)  # exclude true class and pad
        with torch.no_grad():
            true_dist = torch.full(
                (logits.size(0), self.vocab_size),
                fill_value=smooth_val,
                device=logits.device,
            )
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            true_dist[:, self.pad_idx] = 0.0  # pad gets zero mass
            pad_mask = (target == self.pad_idx)
            true_dist[pad_mask] = 0.0          # entire rows for pad tokens → 0

        # KL-divergence: sum(-true_dist * log_softmax(logits))
        log_probs = torch.log_softmax(logits, dim=-1)
        loss = -(true_dist * log_probs).sum(dim=-1)  # [batch * tgt_len]

        # Average only over non-pad tokens
        non_pad = (~pad_mask).sum()
        if non_pad == 0:
            return loss.sum() * 0.0
        return loss.sum() / non_pad


# ══════════════════════════════════════════════════════════════════════
#   TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    """
    Run one epoch of training or evaluation.

    Args:
        data_iter  : DataLoader yielding (src, tgt) batches of token indices.
        model      : Transformer instance.
        loss_fn    : LabelSmoothingLoss (or any nn.Module loss).
        optimizer  : Optimizer (None during eval).
        scheduler  : NoamScheduler instance (None during eval).
        epoch_num  : Current epoch index (for logging).
        is_train   : If True, perform backward pass and scheduler step.
        device     : 'cpu' or 'cuda'.

    Returns:
        avg_loss : Average loss over the epoch (float).
    """
    model.train() if is_train else model.eval()
    mode = "Train" if is_train else "Val"

    total_loss  = 0.0
    total_tokens = 0

    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        pbar = tqdm(data_iter, desc=f"Epoch {epoch_num} [{mode}]", leave=False)
        for batch_idx, (src, tgt) in enumerate(pbar):
            src = src.to(device)  # [batch, src_len]
            tgt = tgt.to(device)  # [batch, tgt_len]

            # Decoder input: everything except last token
            tgt_input = tgt[:, :-1]
            # Decoder target: everything except first token (<sos>)
            tgt_output = tgt[:, 1:]

            # Build masks
            src_mask = make_src_mask(src, pad_idx=Vocabulary.PAD_IDX).to(device)
            tgt_mask = make_tgt_mask(tgt_input, pad_idx=Vocabulary.PAD_IDX).to(device)

            # Forward pass
            logits = model(src, tgt_input, src_mask, tgt_mask)
            # logits: [batch, tgt_len-1, vocab_size]

            # Flatten for loss
            batch_size, tgt_len, vocab_size = logits.shape
            logits_flat  = logits.contiguous().view(-1, vocab_size)
            targets_flat = tgt_output.contiguous().view(-1)

            loss = loss_fn(logits_flat, targets_flat)

            # Count non-pad tokens for averaging
            non_pad = (targets_flat != Vocabulary.PAD_IDX).sum().item()
            total_loss   += loss.item() * non_pad
            total_tokens += non_pad

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping for stability
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

            # W&B step-level logging
            if is_train and wandb.run is not None:
                wandb.log({
                    f"step_loss": loss.item(),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                })

    avg_loss = total_loss / max(total_tokens, 1)

    # W&B epoch-level logging
    if wandb.run is not None:
        wandb.log({f"{mode.lower()}_loss": avg_loss, "epoch": epoch_num})

    return avg_loss


# ══════════════════════════════════════════════════════════════════════
#   GREEDY DECODING
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.

    Args:
        model        : Trained Transformer.
        src          : Source token indices, shape [1, src_len].
        src_mask     : shape [1, 1, 1, src_len].
        max_len      : Maximum number of tokens to generate.
        start_symbol : Vocabulary index of <sos>.
        end_symbol   : Vocabulary index of <eos>.
        device       : 'cpu' or 'cuda'.

    Returns:
        ys : Generated token indices, shape [1, out_len].
             Includes start_symbol; stops at (and includes) end_symbol
             or when max_len is reached.
    """
    model.eval()
    with torch.no_grad():
        # Encode source once
        memory = model.encode(src, src_mask)  # [1, src_len, d_model]

        # Start with <sos>
        ys = torch.tensor([[start_symbol]], dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys, pad_idx=Vocabulary.PAD_IDX).to(device)
            logits = model.decode(memory, src_mask, ys, tgt_mask)
            # logits: [1, current_len, vocab_size]

            # Greedy: take the most probable token at the last position
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [1, 1]
            ys = torch.cat([ys, next_token], dim=1)  # [1, current_len + 1]

            if next_token.item() == end_symbol:
                break

    return ys


# ══════════════════════════════════════════════════════════════════════
#   BLEU EVALUATION
# ══════════════════════════════════════════════════════════════════════

def _decode_tokens(indices, tgt_vocab: Vocabulary) -> str:
    """Convert a list of token indices to a string, stripping special tokens."""
    special = {
        Vocabulary.PAD_IDX,
        Vocabulary.SOS_IDX,
        Vocabulary.EOS_IDX,
        Vocabulary.UNK_IDX,
    }
    tokens = []
    for idx in indices:
        if idx == Vocabulary.EOS_IDX:
            break
        if idx not in special:
            tokens.append(tgt_vocab.lookup_token(int(idx)))
    return ' '.join(tokens)


def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.

    Args:
        model           : Trained Transformer (in eval mode).
        test_dataloader : DataLoader over the test split.
        tgt_vocab       : Vocabulary object.
        device          : 'cpu' or 'cuda'.
        max_len         : Max decode length per sentence.

    Returns:
        bleu_score : Corpus-level BLEU (float, range 0–100).
    """
    model.eval()

    hypotheses = []
    references = []

    sos_idx = Vocabulary.SOS_IDX
    eos_idx = Vocabulary.EOS_IDX
    pad_idx = Vocabulary.PAD_IDX

    with torch.no_grad():
        for src, tgt in tqdm(test_dataloader, desc="BLEU Evaluation", leave=False):
            src = src.to(device)
            tgt = tgt.to(device)

            src_mask = make_src_mask(src, pad_idx=pad_idx).to(device)

            # Greedy decode
            pred_tokens = greedy_decode(
                model, src, src_mask,
                max_len=max_len,
                start_symbol=sos_idx,
                end_symbol=eos_idx,
                device=device,
            )  # [1, out_len]

            # Hypothesis
            hyp = _decode_tokens(pred_tokens[0].tolist(), tgt_vocab)
            hypotheses.append(hyp)

            # Reference (skip <sos> at index 0)
            ref = _decode_tokens(tgt[0, 1:].tolist(), tgt_vocab)
            references.append([ref])   # wrapped in list for sacrebleu format

    # Compute corpus BLEU
    if _HF_EVALUATE:
        bleu_metric = hf_evaluate.load("bleu")
        result = bleu_metric.compute(
            predictions=hypotheses,
            references=[[r[0]] for r in references],
        )
        bleu_score = result["bleu"] * 100.0
    elif _SACREBLEU:
        bleu = _sacrebleu.corpus_bleu(
            hypotheses,
            [[r[0] for r in references]],
        )
        bleu_score = bleu.score
    else:
        # Fallback: simple sentence-level BLEU average using nltk
        try:
            from nltk.translate.bleu_score import corpus_bleu as nltk_corpus_bleu, SmoothingFunction
            smoother = SmoothingFunction().method1
            flat_refs = [[ref[0].split()] for ref in references]
            flat_hyps = [hyp.split() for hyp in hypotheses]
            bleu_score = nltk_corpus_bleu(flat_refs, flat_hyps,
                                          smoothing_function=smoother) * 100.0
        except ImportError:
            raise ImportError(
                "Install sacrebleu or evaluate for BLEU computation: "
                "pip install sacrebleu"
            )

    return bleu_score


# ══════════════════════════════════════════════════════════════════════
#   CHECKPOINT UTILITIES
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """
    Save model + optimiser + scheduler state to disk.

    Saves a dict with keys:
        'epoch', 'model_state_dict', 'optimizer_state_dict',
        'scheduler_state_dict', 'model_config'
    """
    torch.save(
        {
            'epoch':                epoch,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'model_config': {
                'src_vocab_size': model.src_embedding.num_embeddings,
                'tgt_vocab_size': model.tgt_embedding.num_embeddings,
                'd_model':        model.d_model,
                'N':              len(model.encoder.layers),
                'num_heads':      model.encoder.layers[0].self_attn.num_heads,
                'd_ff':           model.encoder.layers[0].ffn.linear1.out_features,
                'dropout':        model.encoder.layers[0].dropout.p,
            },
        },
        path,
    )
    print(f"Checkpoint saved to {path} (epoch {epoch})")


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model (and optionally optimizer/scheduler) state from disk.

    Returns:
        epoch : The epoch at which the checkpoint was saved (int).
    """
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and checkpoint.get('optimizer_state_dict') is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    print(f"Checkpoint loaded from {path} (epoch {epoch})")
    return epoch


# ══════════════════════════════════════════════════════════════════════
#   EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment() -> None:
    """
    Full training experiment following the assignment specification.
    """

    # ── Hyperparameters ───────────────────────────────────────────────
    config = {
        'd_model':      512,
        'N':            6,
        'num_heads':    8,
        'd_ff':         2048,
        'dropout':      0.1,
        'warmup_steps': 4000,
        'num_epochs':   20,
        'batch_size':   128,
        'min_freq':     2,
        'label_smooth': 0.1,
        'max_len':      100,
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # ── W&B init ──────────────────────────────────────────────────────
    wandb.init(project="da6401-a3", config=config)
    cfg = wandb.config

    # ── Data ──────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = build_dataloaders(
        batch_size=cfg.batch_size,
        min_freq=cfg.min_freq,
    )

    # ── Model ─────────────────────────────────────────────────────────
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=cfg.d_model,
        N=cfg.N,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")
    wandb.log({"num_parameters": num_params})

    # ── Optimizer ─────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1.0,          # base lr=1.0; Noam scheduler scales it
        betas=(0.9, 0.98),
        eps=1e-9,
    )

    # ── Scheduler ─────────────────────────────────────────────────────
    scheduler = NoamScheduler(
        optimizer,
        d_model=cfg.d_model,
        warmup_steps=cfg.warmup_steps,
    )

    # ── Loss ──────────────────────────────────────────────────────────
    loss_fn = LabelSmoothingLoss(
        vocab_size=len(tgt_vocab),
        pad_idx=Vocabulary.PAD_IDX,
        smoothing=cfg.label_smooth,
    )

    # ── Training loop ─────────────────────────────────────────────────
    best_val_loss = float('inf')
    best_ckpt_path = "best_checkpoint.pt"

    for epoch in range(cfg.num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1} / {cfg.num_epochs}")
        print(f"{'='*60}")

        train_loss = run_epoch(
            train_loader, model, loss_fn,
            optimizer, scheduler,
            epoch_num=epoch + 1,
            is_train=True,
            device=device,
        )

        val_loss = run_epoch(
            val_loader, model, loss_fn,
            optimizer=None, scheduler=None,
            epoch_num=epoch + 1,
            is_train=False,
            device=device,
        )

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss":   val_loss,
        })

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch + 1, best_ckpt_path)
            wandb.log({"best_val_loss": best_val_loss})

        # Also save latest checkpoint
        save_checkpoint(model, optimizer, scheduler, epoch + 1, "checkpoint.pt")

    # ── Final BLEU evaluation ─────────────────────────────────────────
    print("\nLoading best checkpoint for BLEU evaluation...")
    load_checkpoint(best_ckpt_path, model)
    model.to(device)

    bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=device, max_len=cfg.max_len)
    print(f"\nTest BLEU: {bleu:.2f}")
    wandb.log({"test_bleu": bleu})

    wandb.finish()


if __name__ == "__main__":
    run_training_experiment()
