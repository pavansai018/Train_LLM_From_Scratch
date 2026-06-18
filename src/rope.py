import torch

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, d_model: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()

        assert d_model % 2 == 0
        dim_indices = torch.arange(0, d_model, 2).float()
        inv_freq = 1.0 / (theta ** (dim_indices / d_model))
        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, inv_freq)
        emb = freqs.repeat_interleave(2, dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())

    @staticmethod
    def rotate_half(x):
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)
    
    def forward(self, x, seq_len):
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        return (x * cos) + (self.rotate_half(x) * sin)
    
