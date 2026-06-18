from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int = 50257
    d_model: int = 256
    num_heads: int = 4
    num_layers: int = 4
    max_seq_len: int = 128
    dropout: float = 0.1
    embd_dropout: float = 0.1
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 50
    max_steps: int = 500
    batch_size: int = 4
    grad_accum_steps: int = 2
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8