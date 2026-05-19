# Implementing a Transformer for Machine Translation

## Overview

In this assignment, I will implement the landmark architecture from the paper "Attention Is All You Need" from scratch using PyTorch. Transitioning from the convolutional neural networks used in previous assignments, I will now build a purely attention-based sequence-to-sequence model. The goal is to develop a Neural Machine Translation (NMT) system capable of translating text from German to English.

## Github Link

[Github link](https://github.com/BritiJadav/da6401_assignment_3)

## W&B Report Link

[Wandb Report Link](https://api.wandb.ai/links/britijadav-indian-institute-of-technology-madras/fycsdy37)


## Architecture

This project implements the full Transformer architecture as described in [Vaswani et al., 2017](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf), including:

- **Scaled Dot-Product Attention** — `Attention(Q, K, V) = softmax(QKᵀ / √dₖ) · V`
- **Multi-Head Attention** — 8 parallel attention heads (implemented from scratch, no `nn.MultiheadAttention`)
- **Sinusoidal Positional Encoding** — fixed frequency-based position representations
- **Encoder Stack** — 3 layers of Multi-Head Attention + Feed-Forward Network + Add & Norm
- **Decoder Stack** — 3 layers of Masked Self-Attention + Cross-Attention + FFN + Add & Norm
- **Label Smoothing** — ε = 0.1 to prevent over-confident predictions
- **Noam Scheduler** — linear warmup followed by inverse square root decay


## Dataset

**Multi30k** — A multilingual dataset for Neural Machine Translation:
- 29,000 training pairs
- 1,014 validation pairs
- 1,000 test pairs
- Language pair: German (de) → English (en)
- Source: [bentrevett/multi30k](https://huggingface.co/datasets/bentrevett/multi30k)


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


## Results

| Metric | Value |
|--------|-------|
| Test BLEU Score | **42.88** |
| Best Checkpoint Epoch | 2 |


## Project Structure
```
├── model.py          # Transformer architecture (Encoder, Decoder, Attention)
├── dataset.py        # Multi30k dataset loading and spaCy tokenization
├── train.py          # Training loop, greedy decoding, BLEU evaluation
├── lr_scheduler.py   # Noam learning rate scheduler
├── requirements.txt  # Dependencies
└── README.md         # This file
```

## Installation

```bash
pip install -r requirements.txt
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```


## Dependencies

- `torch` — Model implementation and training
- `spacy` — German and English tokenization
- `datasets` — Multi30k dataset loading
- `sacrebleu` — BLEU score evaluation
- `wandb` — Experiment tracking and visualization
- `gdown` — Trained weights download from Google Drive

