# Train LLM From Scratch

A ground-up PyTorch implementation of a modern decoder-only Transformer — the architecture behind GPT, LLaMA, and Mistral — built one component at a time.

Each folder is a standalone, heavily-commented module. The goal is to understand *why* every design decision exists, not just *what* the code does.

> Inspired by [how-to-train-your-gpt](https://github.com/raiyanyahya/how-to-train-your-gpt)

---

## Architecture

This implementation uses the modern "LLaMA-style" Transformer, departing from the original GPT-2 in three key ways: RoPE instead of absolute position embeddings, RMSNorm instead of LayerNorm, and SwiGLU instead of GELU.

```
Raw Text
   │
   ▼
┌─────────────────────────────────┐
│  i_tokenization / tokenizer.py  │  GPT-2 BPE → token IDs
└─────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────┐
│  ii_embedding / embedding.py    │  token IDs → dense vectors  [batch, seq, d_model]
└─────────────────────────────────┘
   │
   ▼  repeated N times
┌─────────────────────────────────────────────────────────────┐
│  v_transformer / transformer.py  —  TransformerBlock        │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐  │
│   │  rms_norm.py  →  multi_head_attention.py  →  + x   │  │
│   │                      ↑                              │  │
│   │          iii_position_encoding / rope.py            │  │
│   └─────────────────────────────────────────────────────┘  │
│   ┌─────────────────────────────────────────────────────┐  │
│   │  rms_norm.py  →  swiglu.py  →  + x                 │  │
│   └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
   │
   ▼
Logits  [batch, seq, vocab_size]  →  next-token prediction
```

---

## Project Structure

```
train_llm_from_scratch/
│
├── system_check.py                 # verify torch, CUDA, tiktoken before you start
│
├── i_tokenization/
│   └── tokenizer.py                # GPT-2 BPE tokenizer wrapper (tiktoken)
│
├── ii_embedding/
│   └── embedding.py                # learnable token embedding table
│
├── iii_position_encoding/
│   └── rope.py                     # Rotary Position Embeddings
│
├── iv_attention/
│   └── multi_head_attention.py     # multi-head self-attention + causal mask + RoPE
│
└── v_transformer/
    ├── rms_norm.py                 # Root Mean Square Layer Normalization
    ├── swiglu.py                   # SwiGLU gated feed-forward network
    └── transformer.py              # complete TransformerBlock (pre-norm)
```

---

## Component Breakdown

| # | Module | What it does | Modern choice |
|---|--------|-------------|---------------|
| i | `tokenizer.py` | Converts raw text → token IDs using GPT-2's 50,257-token BPE vocabulary | tiktoken (fast Rust implementation) |
| ii | `embedding.py` | Maps discrete token IDs → continuous 768-dim vectors the network can process | Learnable `nn.Embedding` table |
| iii | `rope.py` | Encodes position by *rotating* Q and K vectors rather than adding positional vectors | **RoPE** — used in LLaMA, Mistral, Gemma, Qwen |
| iv | `multi_head_attention.py` | 12 attention heads each learn different linguistic patterns; causal mask prevents peeking at future tokens | Combined QKV projection, RoPE applied to Q & K |
| v-a | `rms_norm.py` | Stabilizes activations before each sublayer using `x / rms(x) * weight` | **RMSNorm** — cheaper than LayerNorm, used in LLaMA |
| v-b | `swiglu.py` | Per-token non-linear processing: `W3(SiLU(W1(x)) * W2(x))` | **SwiGLU** — gated activation, used in LLaMA, PaLM, Gemini |
| v-c | `transformer.py` | Stacks RMSNorm → Attention → Residual → RMSNorm → FFN → Residual | Pre-norm architecture (more stable than post-norm) |

---

## Key Design Decisions

### Why RoPE over learned position embeddings?
Absolute position embeddings add a fixed vector per position. RoPE instead *rotates* query and key vectors by a position-dependent angle. The result: the dot-product Q·K naturally depends on the *relative* distance between tokens rather than their absolute indices. This gives better length extrapolation and is why every major modern LLM switched to it.

### Why RMSNorm over LayerNorm?
LayerNorm computes both mean and variance. RMSNorm skips the mean subtraction — empirically, the re-centering step adds little value but meaningful compute cost. Identical training stability at ~10% less FLOPs per normalization.

### Why SwiGLU over GELU?
The gating mechanism (`SiLU(W1(x)) * W2(x)`) lets the network *learn* which features to amplify or suppress before the final projection, rather than applying the same non-linearity everywhere. PaLM, LLaMA, and Gemini all default to SwiGLU.

### Why pre-normalization?
Original Transformer applies LayerNorm *after* the sublayer. Pre-norm applies it *before*: `x = x + sublayer(norm(x))`. This keeps gradient magnitudes stable during early training, making it much easier to train deep stacks (32–80 layers) without learning rate warm-up tricks.

---

## Getting Started

**1. Clone**
```bash
git clone https://github.com/pavansai018/train_llm_from_scratch.git
cd train_llm_from_scratch
```

**2. Install dependencies**
```bash
pip install torch numpy matplotlib tiktoken datasets
```

**3. Verify your environment**
```bash
python system_check.py
```

**4. Run a component**
```python
from i_tokenization.tokenizer import SimpleTokenizer
from ii_embedding.embedding import Embedding
from v_transformer.transformer import TransformerBlock

tokenizer = SimpleTokenizer()
ids = tokenizer.encode("The transformer architecture is elegant.")

import torch
embedding = Embedding(vocab_size=50257, d_model=768)
x = embedding(torch.tensor([ids]))  # [1, seq_len, 768]

block = TransformerBlock(d_model=768, num_heads=12,)
out = block(x)  # [1, seq_len, 768]
print(out.shape)
```

---

## Requirements

| Package | Purpose |
|---------|---------|
| `torch` | Neural network layers, autograd |
| `tiktoken` | Fast GPT-2 BPE tokenizer |
| `numpy` | Numerical utilities |
| `matplotlib` | Visualization (attention maps, loss curves) |
| `datasets` | HuggingFace dataset loading |

---

## Roadmap

- [x] Tokenizer (BPE)
- [x] Token Embedding
- [x] Rotary Position Encoding (RoPE)
- [x] Multi-Head Self-Attention
- [x] RMSNorm
- [x] SwiGLU Feed-Forward
- [x] Transformer Block (pre-norm)
- [ ] Full GPT model (stack N blocks + output projection)
- [ ] Training loop (DataLoader, cross-entropy loss, AdamW)
- [ ] Text generation (greedy / temperature / top-k sampling)
- [ ] Training on a real dataset (TinyShakespeare / OpenWebText)

---

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — Su et al., 2021
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) — Zhang & Sennrich, 2019
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) — Noam Shazeer, 2020
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) — Touvron et al., 2023
