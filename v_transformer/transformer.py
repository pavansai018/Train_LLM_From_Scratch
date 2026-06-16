import torch
import torch.nn as nn
from rms_norm import RMSNorm
from iv_attention.multi_head_attention import MultiHeadAttention
from swiglu import SwiGLU

class TransformerBlock(nn.Module):
    """
    One complete transformeer layer (attention + FFN with residuals)
    Stack N of these to build a deep language model

    Architecture (Pre Norm):
        x = x + Attention(RMSNorm(x), mask)  <- Mix information BETWEEN tokens
        x = x + SwiGLU(RMSNorm(x))           <- Process information WITHIN tokens

    Each sublayer: normalize FIRST (pre-norm), then compute,
    then ADD back the original (residual connection).

    Without residuals: deep networks cant train (vanishing gradients)
    without pre norm: training is unstable at large depths
    without FFN: no non linear processing per token
    without attention: no information mixing between tokens
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        """
        First normalization - before attention
        pre-norm: clean, well scaled input -> stable attention computation
        """
        self.norm1 = RMSNorm(d_model)

        # multi head self attention with RoPE and causal masking
        # the core mechanism that lets tokens 'talk to' each other
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)

        # second normalization - before FFN
        # FFN expects normalized input for consistent behavior across layers
        self.norm2 = RMSNorm(d_model)

        # SwiGLU feed forward network
        # non linear processing per token. without this, stacking more attention
        # layers would be no more powerful than one layer
        self.ffn = SwiGLU(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        forward pass: norm -> sublayer -> add residual
        executed twice: once for attention and once for FFN
        """

        """
        Sublayer 1: Self attention with resiual
        x = x + Attention(rmsnorm(x))
        the model learns what CHANGES (the delta) to make to x,
        not what to replace x with entirely. This is easier to learn.
        If attention cant improve things, it can output near zero
        """
        x = x + self.attention(self.norm1(x), mask)

        """
        Sublayer 2: Feed forward with residual
        x = x + FFN(rmsnorm(x))
        same residual pattern. After mixing information via attention, each token
        'thinks' independently via the FFN.
        Attention = group discussions. FFN = private reflection
        """
        x = x + self.ffn(self.norm2(x))

        return x
    