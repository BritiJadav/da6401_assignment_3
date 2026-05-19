# Implementing a Transformer for Machine Translation

## Overview

In this assignment, you will implement the landmark architecture from the paper "Attention Is All You Need" from scratch using PyTorch. Transitioning from the convolutional neural networks used in previous assignments, you will now build a purely attention-based sequence-to-sequence model. The goal is to develop a Neural Machine Translation (NMT) system capable of translating text from German to English.

---

## Architecture

This project implements the full Transformer architecture as described in [Vaswani et al., 2017](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf), including:

- **Scaled Dot-Product Attention** — `Attention(Q, K, V) = softmax(QKᵀ / √dₖ) · V`
- **Multi-Head Attention** — 8 parallel attention heads (implemented from scratch, no `nn.MultiheadAttention`)
- **Sinusoidal Positional Encoding** — fixed frequency-based position representations
- **Encoder Stack** — 3 layers of Multi-Head Attention + Feed-Forward Network + Add & Norm
- **Decoder Stack** — 3 layers of Masked Self-Attention + Cross-Attention + FFN + Add & Norm
- **Label Smoothing** — ε = 0.1 to prevent over-confident predictions
- **Noam Scheduler** — linear warmup followed by inverse square root decay

---

## Dataset

**Multi30k** — A multilingual dataset for Neural Machine Translation:
- 29,000 training pairs
- 1,014 validation pairs
- 1,000 test pairs
- Language pair: German (de) → English (en)
- Source: [bentrevett/multi30k](https://huggingface.co/datasets/bentrevett/multi30k)

---

## Model Configuration

| Parameter | Value |
|-----------|-------|
| `d_model` | 256 |
| `N` (layers) | 3 |
| `num_heads` | 8 |
| `d_ff` | 512 |
| `dropout` | 0.3 |
| `warmup_steps` | 2000 |
| `batch_size` | 128 |
| `epochs` | 60 |
| `label_smoothing` | 0.1 |

---

## Results

| Metric | Value |
|--------|-------|
| Test BLEU Score | **42.88** |
| Best Checkpoint Epoch | 2 |

---

## Project Structure

```
├── model.py          # Transformer architecture (Encoder, Decoder, Attention)
├── dataset.py        # Multi30k dataset loading and spaCy tokenization
├── train.py          # Training loop, greedy decoding, BLEU evaluation
├── lr_scheduler.py   # Noam learning rate scheduler
├── requirements.txt  # Dependencies
└── README.md         # This file
```

---

## Installation

```bash
pip install -r requirements.txt
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```

---

## Training

```bash
python train.py
```

---

## Inference

The model supports end-to-end German → English translation:

```python
from model import Transformer

model = Transformer().to(device)
model.eval()

english = model.infer("Ein Hund rennt durch das Gras.")
print(english)  # "a dog runs through the grass ."
```

---

## W&B Report

All experiments are documented in the public W&B report including:
- Noam Scheduler vs Fixed Learning Rate comparison
- Ablation study on the √(1/dₖ) scaling factor
- Attention head specialization heatmaps
- Sinusoidal PE vs Learned Positional Embeddings
- Label Smoothing ε=0.1 vs ε=0.0 analysis

---

## Dependencies

- `torch` — Model implementation and training
- `spacy` — German and English tokenization
- `datasets` — Multi30k dataset loading
- `sacrebleu` — BLEU score evaluation
- `wandb` — Experiment tracking and visualization
- `gdown` — Trained weights download from Google Drive

---

## Reference

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). **Attention Is All You Need**. *Advances in Neural Information Processing Systems*, 30.
