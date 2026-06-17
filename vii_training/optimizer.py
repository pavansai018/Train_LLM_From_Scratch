import torch

def create_optimizer(model, config):
    """
    AdamW with two parameter groups (with/without weight decay)
    Norm layers and biases should not get weight decay - it pushes
    them toward zero, destroying normalization

    Group1 (weight decay > 0): Linear weights, embeddings
    Group2 (weight decay = 0): Biases, RMSNorm, LayerNorm
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() <=1 or 'norm' in name.lower() or 'bias' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(name)

    
    return torch.optim.AdamW(
        [
            {
                'params': decay_params, 'weight_decay': config.weight_decay,
            },
            {
                'params': no_decay_params, 'weight_decay': 0.0,
            },
        ],
        lr=config.learning_rate,
        betas=config.betas,
        eps=config.eps,
    )