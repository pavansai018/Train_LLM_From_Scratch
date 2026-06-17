from dataclasses import dataclass

@dataclass
class GPTConfig:
    """
    All hyperparameters in one place
    changing model size is one line. no hunting through code
    """

    # Architecture
    vocab_size: int = 50257 # 50,257 unique tokens in GPT-2 vocabulary
    d_model: int = 768      # each token becomes a 768 dim vector
                            # bigger  = more nuanced meanings, more compute
    num_heads: int = 12     # 12 attention heads (12 x 64 = 768)
    num_layers: int = 12    # 12 transformer blocks stacked
                            # deeper = better reasoning, harder to train
    max_seq_len: int = 1024 # max tokens model can process at once


    # Regularization (prevent overfitting)
    dropout: float = 0.1    # randomly disable 10% of neurons during training
    embd_dropout: float = 0.1 #dropout applied right after embedding lookup


    # Training
    learning_rate: float = 3e-4  # step size for weight updates
    weight_decay: float = 0.1    # penalize large weights (L2 regularization)
    warmup_steps: int = 2000     # gradually increase LR for first 2000 steps
    max_steps: int = 100000      # total training iterations
    batch_szie: int = 8          # sequences processed per GPU step
    grad_accum_steps: int = 4    # accumulate gradient steps (effective batch = 8x4=32)
    betas: tuple = (0.9, 0.95)   # AdamW momentum coeeficients
    eps: float = 1e-8            # small constant preventing dividing by zero


    def __post_init__(self):
        """Validate configuration consistency"""
        assert self.d_model % self.num_heads == 0, (
            f'd_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})'
        )

        