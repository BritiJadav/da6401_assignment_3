# Building a Complete Visual Perception Pipeline


[Github link](https://github.com/BritiJadav/da6401_assignment_2)


[Wandb Report Link](https://wandb.ai/britisundarghatak100-iit-madras/da6401-assignment-2/reports/da6401_Assignment_2--VmlldzoxNjQ4ODM0MA?accessToken=lk9wwi6vjezfpggd25ioowbw54n3vvl3qnj4ujirpa6wqh6uno6va27a8k2ovwfg)


In this assignment, I will implement the landmark architecture from the paper "Attention Is All You Need" from scratch using PyTorch. Transitioning from the convolutional neural networks used in previous assignments, I will now build a purely attention-based sequence-to-sequence model. The goal is to develop a Neural Machine Translation (NMT) system capable of translating text from German to English.


Architecture

This project implements the full Transformer architecture as described in Vaswani et al., 2017, including:

Scaled Dot-Product Attention — Attention(Q, K, V) = softmax(QKᵀ / √dₖ) · V
Multi-Head Attention — 8 parallel attention heads (implemented from scratch, no nn.MultiheadAttention)
Sinusoidal Positional Encoding — fixed frequency-based position representations
Encoder Stack — 3 layers of Multi-Head Attention + Feed-Forward Network + Add & Norm
Decoder Stack — 3 layers of Masked Self-Attention + Cross-Attention + FFN + Add & Norm
Label Smoothing — ε = 0.1 to prevent over-confident predictions
Noam Scheduler — linear warmup followed by inverse square root decay

Dataset

Multi30k — A multilingual dataset for Neural Machine Translation:

29,000 training pairs
1,014 validation pairs
1,000 test pairs
Language pair: German (de) → English (en)


Model Configuration

Parameter   Value
d_model     256
N(layers)   3
num_heads   8
d_ff        512
dropout     0.3
warmup_steps2000
batch_size  128
epochs      60
label_smoothing0.1
