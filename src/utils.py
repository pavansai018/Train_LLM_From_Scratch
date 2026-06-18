import torch

def find_device():
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        return torch.device('cuda')
    else:
        print(f'Using CPU')
        return torch.device('cpu')
    

def create_causal_mask(seq_len, device):
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.view(1, 1, seq_len, seq_len)

