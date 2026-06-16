import torch
import torch.nn as nn
import math

class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE)
    Instead of adding positions into embeddings,
    we rotate Q and K vectors by position-dependent angles.
    The dot poduct q_i . k_j then depends only on (j-i),
    which is exactly what attention should care about.

    Used in LLaMA 1/2/3, Mistral, Mixtral, Qwn 1/2, Gemma

    How it works at a glance:
        1. for each pair of dimensions (0,1), (2,3), (4,5),....
        2. rotate by angle = position * frequency
        3. lower dims rotate fast (local position)
        4. higher dims rotate slow (global position)
        5. the dot product naturally depends on relative distance.

    """

    def __init__(self, d_model: int, max_seq_len: int = 2048, theta: float = 10000.0):
        """
        Precompute rotation frequencies for fast lookup.
        Args:
            d_model: Head dimension (e.g. 64 gor GPT-2). Must be even.
            max_seq_len: Precompute angles for positions 0..max_seq_len-1
            theta: Base frequency. 10000 is standard. controls the spread 
                   between fast and slow rotation frequencies
        """
        super().__init__()
        # verify d_model is even (must have pairs to rotate)
        assert d_model % 2 == 0, (
            f'd_model ({d_model}) must be even for RoPE.'
            f'Each pair of dimensions needs a partner to rotate with'
        )

        # create dimension indices: [0, 2, 4, ..., d_model-2]
        # each pair (2i, 2i+1) gets the same rotation frequency
        # we only need half the indices because pairs share
        dim_indices = torch.arange(0, d_model, 2).float()

        # compute rotation frequencies
        # theta_i = 1 / (theta ^ (2i / d_model))
        #   i=0: 1/10000^(0/64) = 1.0 -> fast rotation (local)
        #   i=30: 1/10000^(60/64) = 0.0001 -> slow rotation (global)
        # This multi-scale approach means some dimesnsions capture local word order while
        # others capture long-range position relationships.
        inv_freq = 1.0 / (theta ** (dim_indices / d_model))

        # precompute angles for all positions
        # computing cos/sin during training is epxpensive
        # precomputing them once and caching them is 100x faster
        positions = torch.arange(max_seq_len).float() # [0, 1, 2, ..., 2047]


        # outer product: each position * frequency
        # freqs[p, i] = p * inv_freq[i] = angle for position p, dim pair i
        # shape: [max_seq_len, d_model/2]
        freqs = torch.outer(positions, inv_freq)

        # duplicate to full dimension
        # each pair (2i, 2i+1) gets the same angle
        # so we can copy each angle [theta0, theta1, theta2,... ] -> [theta0, theta0, theta1, theta1,..]
        emb = freqs.repeat_interleave(2, dim=-1) # [max_seq_len, d_model]

        # cache cos and sin for all positions
        # register_buffer means these move with model.to(device)
        # and are saved with model.sate_dict(), but are not trainable params (no gradients needed)
        self.register_buffer('cos_cached', emb.cos()) # cos for each angle
        self.register_buffer('sin_cached', emb.sin()) # sin for each angle

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """
        prepare a vector for the rotation formula
        the rotation formula is x_dash = x*cos + rotate_half(x)*sin
              For vector [x0, x1, x2, x3, x4, x5]:
              rotate_half returns [-x1, x0, -x3, x2, -x5, x4]
              
              Why this works: For pair (x0, x1) rotated by angle θ:
                x0' = x0*cos(θ) - x1*sin(θ)   <- matches: x0*cos + (-x1)*sin
                x1' = x0*sin(θ) + x1*cos(θ)   <- matches: x1*cos + (x0)*sin
              
              So executing (x*cos + rotate_half(x)*sin) performs rotation
              on every dimension pair simultaneously — no loop needed!
        """
        x1 = x[..., : x.shape[-1] // 2] # First half:  [x0, x2, x4, ...]
        x2 = x[..., x.shape[-1] // 2 :]   # Second half: [x1, x3, x5, ...]
        return torch.cat([-x2, x1], dim=-1)  # [-x1, x0, -x3, x2, -x5, x4, ...]
    
    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Apply RoPE to queries or keys
        Input:  [batch, num_heads, seq_len, head_dim]
                x can be either Q or K (NOT V — values don't need position)
        Output: Same shape, rotated by position-dependent angles

        WHY applied only to Q and K:
        The attention score = Q_i · K_j controls WHICH values to attend to.
        We want this score to depend on relative position.
        The VALUE vectors carry content — position is irrelevant for the
        content itself. Position only matters for deciding which tokens
        to pay attention TO.
        """
        # WHAT: Extract cos and sin for current sequence length
        # WHY: If seq_len=512 but max_seq_len=2048, we only need
        #      the first 512 rows of the cached cos/sin tables.
        cos = self.cos_cached[:seq_len]   # [seq_len, head_dim]
        sin = self.sin_cached[:seq_len]   # [seq_len, head_dim]

        # WHAT: Add batch and head dimensions for broadcasting
        # WHY: cos/sin are [seq_len, head_dim]. We need them to
        #      multiply with x [batch, heads, seq_len, head_dim].
        #      unsqueeze(0).unsqueeze(0) adds dims at positions 0 and 1:
        #      [seq_len, head_dim] -> [1, 1, seq_len, head_dim]
        #      Now they broadcast correctly over batch and heads.
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        # WHAT: Execute rotation: x_rotated = x*cos(θ) + rotate_half(x)*sin(θ)
        # WHY: This is mathematically equivalent to applying a 2D rotation
        #      matrix to each pair of dimensions, but implemented in pure
        #      element-wise operations — much faster and parallelizable.
        return (x * cos) + (self.rotate_half(x) * sin)