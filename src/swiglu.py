import torch

class SwiGLU(torch.nn.Module):
    def __init__(self, d_model, expansion_factor: int =4):
        super().__init__()
        hidden_dim = d_model * expansion_factor
        self.w1 = torch.nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = torch.nn.Linear(d_model, hidden_dim, bias=False)
        self.w3 = torch.nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        return self.w3(torch.nn.functional.silu(self.w1(x)) * self.w2(x))
    