import os
import torch
import torch.nn as nn
try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False
from torch.utils.data import DataLoader
from typing import Optional
from tqdm import tqdm

from model import Transformer, make_src_mask, make_tgt_mask
from dataset import build_dataloaders, Vocabulary
from lr_scheduler import NoamScheduler

# BLEU library - try sacrebleu first, then evaluate, then nltk
try:
    import sacrebleu as _sacrebleu
    _SACREBLEU = True
except ImportError:
    _SACREBLEU = False

try:
    import evaluate as hf_evaluate
    _HF_EVALUATE = True
except ImportError:
    _HF_EVALUATE = False



#   LABEL SMOOTHING LOSS


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need".

        y_smooth[true_class] = 1 - eps
        y_smooth[other]      = eps / (vocab_size - 2)   # excluding true & pad
        y_smooth[pad]        = 0

    Args:
        vocab_size (int)  : Number of output classes.
        pad_idx    (int)  : Index of <pad> - receives zero probability.
        smoothing  (float): epsilon (default 0.1).
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
            logits : [N, vocab_size]  (raw, pre-softmax)
            target : [N]              (gold indices)
        Returns:
            Scalar loss averaged over non-pad positions.
        """
        smooth_val = self.smoothing / max(self.vocab_size - 2, 1)

        with torch.no_grad():
            true_dist = torch.full(
                (logits.size(0), self.vocab_size),
                fill_value=smooth_val,
                device=logits.device,
            )
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            true_dist[:, self.pad_idx] = 0.0
            pad_mask = (target == self.pad_idx)
            true_dist[pad_mask] = 0.0

        log_probs = torch.log_softmax(logits, dim=-1)
        loss = -(true_dist * log_probs).sum(dim=-1)

        non_pad = (~pad_mask).sum()
        return loss.sum() / non_pad.clamp(min=1)



#   TRAINING LOOP


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

    Returns:
        avg_loss (float) : token-averaged loss over the epoch.
    """
    model.train() if is_train else model.eval()
    mode = "Train" if is_train else "Val"

    total_loss   = 0.0
    total_tokens = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        pbar = tqdm(data_iter, desc=f"Epoch {epoch_num} [{mode}]", leave=False)
        for src, tgt in pbar:
            src = src.to(device)   # [B, src_len]
            tgt = tgt.to(device)   # [B, tgt_len]

            tgt_input  = tgt[:, :-1]  
            tgt_output = tgt[:, 1:]   

            src_mask = make_src_mask(src,       pad_idx=Vocabulary.PAD_IDX).to(device)
            tgt_mask = make_tgt_mask(tgt_input, pad_idx=Vocabulary.PAD_IDX).to(device)

            logits = model(src, tgt_input, src_mask, tgt_mask)
            # logits: [B, tgt_len-1, vocab_size]

            B, T, V = logits.shape
            loss = loss_fn(logits.contiguous().view(-1, V),
                           tgt_output.contiguous().view(-1))

            non_pad = (tgt_output != Vocabulary.PAD_IDX).sum().item()
            total_loss   += loss.item() * non_pad
            total_tokens += non_pad

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if _WANDB and is_train and wandb.run is not None:
                wandb.log({
                    "step_loss":    loss.item(),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                })

    avg_loss = total_loss / max(total_tokens, 1)
    if _WANDB and wandb.run is not None:
        wandb.log({f"{mode.lower()}_loss": avg_loss, "epoch": epoch_num})
    return avg_loss



#   GREEDY DECODING


def greedy_decode(
    model:        Transformer,
    src:          torch.Tensor,
    src_mask:     torch.Tensor,
    max_len:      int,
    start_symbol: int,
    end_symbol:   int,
    device:       str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.

    Args:
        model        : Trained Transformer.
        src          : [1, src_len]
        src_mask     : [1, 1, 1, src_len]
        max_len      : Maximum tokens to generate.
        start_symbol : <sos> index.
        end_symbol   : <eos> index.
        device       : device string.

    Returns:
        ys : [1, out_len]  (includes start_symbol; stops at/after end_symbol)
    """
    model.eval()
    with torch.no_grad():
        memory = model.encode(src, src_mask)
        ys = torch.tensor([[start_symbol]], dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys, pad_idx=Vocabulary.PAD_IDX).to(device)
            logits   = model.decode(memory, src_mask, ys, tgt_mask)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_tok], dim=1)
            if next_tok.item() == end_symbol:
                break

    return ys



#   BEAM SEARCH DECODING


def beam_search_decode(
    model:        Transformer,
    src:          torch.Tensor,
    src_mask:     torch.Tensor,
    max_len:      int,
    start_symbol: int,
    end_symbol:   int,
    beam_size:    int = 4,
    device:       str = "cpu",
) -> torch.Tensor:
    """
    Beam search decoding - significantly better BLEU than greedy.

    Args:
        model       : Trained Transformer.
        src         : [1, src_len]
        src_mask    : [1, 1, 1, src_len]
        max_len     : Maximum tokens to generate.
        start_symbol: <sos> index.
        end_symbol  : <eos> index.
        beam_size   : Number of beams (default 4).
        device      : device string.

    Returns:
        best sequence as [1, out_len] tensor
    """
    model.eval()
    with torch.no_grad():
        # Encode source once
        memory   = model.encode(src, src_mask)  
        # Expand memory for beam_size candidates
        memory   = memory.expand(beam_size, -1, -1)       
        src_mask = src_mask.expand(beam_size, -1, -1, -1) 

        # Each beam: (log_prob, token_sequence)
        beams = [(0.0, [start_symbol])]
        completed = []

        for _ in range(max_len - 1):
            all_candidates = []

            for log_prob, seq in beams:
                if seq[-1] == end_symbol:
                    completed.append((log_prob, seq))
                    continue

                tgt = torch.tensor([seq], dtype=torch.long, device=device)
                tgt_mask = make_tgt_mask(tgt, pad_idx=Vocabulary.PAD_IDX).to(device)

                # Only use first beam's memory/mask for single sequence
                logits = model.decode(
                    memory[:1], src_mask[:1], tgt, tgt_mask
                )  # [1, seq_len, vocab]

                log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)  # [1, vocab]
                topk_log_probs, topk_indices = log_probs[0].topk(beam_size)

                for i in range(beam_size):
                    new_log_prob = log_prob + topk_log_probs[i].item()
                    new_seq      = seq + [topk_indices[i].item()]
                    all_candidates.append((new_log_prob, new_seq))

            if not all_candidates:
                break

            # Keep top beam_size candidates by score normalised by length
            all_candidates.sort(
                key=lambda x: x[0] / len(x[1]),
                reverse=True
            )
            beams = all_candidates[:beam_size]

            # Stop if all beams ended
            if all(s[-1] == end_symbol for _, s in beams):
                completed.extend(beams)
                break

        # Pick best completed sequence (or best beam if none completed)
        candidates = completed if completed else beams
        best_seq = max(
            candidates,
            key=lambda x: x[0] / max(len(x[1]), 1)
        )[1]

    return torch.tensor([best_seq], dtype=torch.long, device=device)



#   BLEU EVALUATION


def _indices_to_str(indices, tgt_vocab: Vocabulary) -> str:
    """Convert index list → string, stripping special tokens."""
    special = {Vocabulary.PAD_IDX, Vocabulary.SOS_IDX,
               Vocabulary.EOS_IDX, Vocabulary.UNK_IDX}
    words = []
    for idx in indices:
        if idx == Vocabulary.EOS_IDX:
            break
        if idx not in special:
            words.append(tgt_vocab.lookup_token(int(idx)))
    return ' '.join(words)


def evaluate_bleu(
    model:            Transformer,
    test_dataloader:  DataLoader,
    tgt_vocab,
    device:           str = "cpu",
    max_len:          int = 100,
) -> float:
    """
    Corpus-level BLEU score on the test set.

    Returns:
        bleu_score (float) in range 0-100.
    """
    model.eval()
    hypotheses = []
    references = []

    with torch.no_grad():
        for src, tgt in tqdm(test_dataloader, desc="BLEU Eval", leave=False):
            src = src.to(device)
            tgt = tgt.to(device)

            src_mask  = make_src_mask(src, pad_idx=Vocabulary.PAD_IDX).to(device)
            pred      = beam_search_decode(
                model, src, src_mask,
                max_len=max_len,
                start_symbol=Vocabulary.SOS_IDX,
                end_symbol=Vocabulary.EOS_IDX,
                beam_size=4,
                device=device,
            )

            hypotheses.append(_indices_to_str(pred[0].tolist(), tgt_vocab))
            references.append(_indices_to_str(tgt[0, 1:].tolist(), tgt_vocab))

    # Compute corpus BLEU
    if _SACREBLEU:
        bleu = _sacrebleu.corpus_bleu(hypotheses, [references], tokenize='none')
        return bleu.score
    elif _HF_EVALUATE:
        metric = hf_evaluate.load("sacrebleu")
        result = metric.compute(predictions=hypotheses,
                                references=[[r] for r in references])
        return result["score"]
    else:
        # Fallback — nltk
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        flat_refs = [[r.split()] for r in references]
        flat_hyps = [h.split() for h in hypotheses]
        return corpus_bleu(flat_refs, flat_hyps,
                           smoothing_function=SmoothingFunction().method1) * 100.0



#   CHECKPOINT UTILITIES


def save_checkpoint(
    model:     Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch:     int,
    path:      str = "checkpoint.pt",
) -> None:
    """
    Save model + optimiser + scheduler + config to disk.

    Saved dict keys:
        epoch, model_state_dict, optimizer_state_dict,
        scheduler_state_dict, model_config
    """
    torch.save(
        {
            'epoch':                epoch,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'model_config': {
                'd_model':   model.d_model,
                'N':         len(model.encoder.layers),
                'num_heads': model.encoder.layers[0].self_attn.num_heads,
                'd_ff':      model.encoder.layers[0].ffn.linear1.out_features,
                'dropout':   model.encoder.layers[0].dropout.p,
            },
        },
        path,
    )
    print(f"Checkpoint saved -> {path}  (epoch {epoch})")


def load_checkpoint(
    path:      str,
    model:     Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model (and optionally optimizer/scheduler) from checkpoint.

    Returns:
        epoch (int) : epoch at which checkpoint was saved.
    """
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    if optimizer is not None and ckpt.get('optimizer_state_dict'):
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler is not None and ckpt.get('scheduler_state_dict'):
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])

    epoch = ckpt.get('epoch', 0)
    print(f"Checkpoint loaded ← {path}  (epoch {epoch})")
    return epoch



#   EXPERIMENT ENTRY POINT


def run_training_experiment() -> None:
    """
    Full training experiment:
        1. W&B init
        2. Build Multi30k dataloaders
        3. Instantiate Transformer (vocab sizes from data)
        4. Adam + Noam scheduler + LabelSmoothingLoss
        5. Training loop with validation + best-checkpoint saving
        6. Final BLEU on test set
    """
    config = {
        'd_model':      256,
        'N':            3,
        'num_heads':    8,
        'd_ff':         512,
        'dropout':      0.3,
        'warmup_steps': 2000,
        'num_epochs':   50,
        'batch_size':   128,
        'min_freq':     2,
        'label_smooth': 0.1,
        'max_len':      100,
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    if _WANDB:
        wandb.init(project="da6401-a3", config=config)
        cfg = wandb.config
    else:
        from types import SimpleNamespace
        cfg = SimpleNamespace(**config)

    #  Data 
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = build_dataloaders(
        batch_size=cfg.batch_size, min_freq=cfg.min_freq,
    )

    #  Model - use explicit vocab sizes during training 
    # (Transformer() with no args rebuilds vocabs inside; here we pass
    #  sizes explicitly to avoid double-downloading the dataset)
    model = Transformer(
        d_model=cfg.d_model, N=cfg.N, num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,       dropout=cfg.dropout,
    ).to(device)

    # Override the vocabs with the ones already built by the dataloader
    # so the model uses identical mappings during training
    model.src_vocab = src_vocab
    model.tgt_vocab = tgt_vocab

    # Resize embeddings if vocab sizes differ (they won't here since
    # both paths use identical Multi30k train + same min_freq=2)
    assert model.src_embedding.num_embeddings == len(src_vocab), (
        f"src vocab mismatch: model={model.src_embedding.num_embeddings} "
        f"data={len(src_vocab)}"
    )
    assert model.tgt_embedding.num_embeddings == len(tgt_vocab), (
        f"tgt vocab mismatch: model={model.tgt_embedding.num_embeddings} "
        f"data={len(tgt_vocab)}"
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")
    if _WANDB and wandb.run: wandb.log({"num_parameters": n_params})

    #  Optimizer & scheduler
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9,
    )
    scheduler = NoamScheduler(optimizer, d_model=cfg.d_model,
                              warmup_steps=cfg.warmup_steps)
    loss_fn   = LabelSmoothingLoss(len(tgt_vocab), Vocabulary.PAD_IDX,
                                    smoothing=cfg.label_smooth)

    #  Training loop 
    best_ckpt_path = "best_checkpoint.pt"

    start_epoch   = 0
    best_val_loss = float('inf')
    total_epochs  = cfg.num_epochs

    for epoch in range(start_epoch, total_epochs):
        print(f"\n{'='*55}\nEpoch {epoch+1}/{cfg.num_epochs}\n{'='*55}")

        train_loss = run_epoch(train_loader, model, loss_fn,
                               optimizer, scheduler,
                               epoch_num=epoch+1, is_train=True, device=device)

        val_loss = run_epoch(val_loader, model, loss_fn,
                             None, None,
                             epoch_num=epoch+1, is_train=False, device=device)

        print(f"  Train loss: {train_loss:.4f}   Val loss: {val_loss:.4f}")
        if _WANDB and wandb.run: wandb.log({"epoch": epoch+1, "train_loss": train_loss, "val_loss": val_loss, "total_epochs": total_epochs})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch+1, best_ckpt_path)
            if _WANDB and wandb.run: wandb.log({"best_val_loss": best_val_loss})

        save_checkpoint(model, optimizer, scheduler, epoch+1, "checkpoint.pt")

    #  Final BLEU
    print("\nEvaluating BLEU on test set...")
    load_checkpoint(best_ckpt_path, model)
    model.to(device)

    bleu = evaluate_bleu(model, test_loader, tgt_vocab,
                          device=device, max_len=cfg.max_len)
    print(f"Test BLEU: {bleu:.2f}")
    if _WANDB and wandb.run:
        wandb.log({"test_bleu": bleu})
        wandb.finish()


if __name__ == "__main__":
    run_training_experiment()
