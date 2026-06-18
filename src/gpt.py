import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformer import TransformerBlock
from rms_norm import RMSNorm
from utils import create_causal_mask

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.embd_dropout = nn.Dropout(config.embd_dropout)
        self.layers = nn.ModuleList([
            TransformerBlock(config.d_model, config.num_heads, config.dropout)
            for _ in range(config.num_layers)
        ])
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.token_embedding.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, input_ids, targets=None):
        batch_size, seq_len = input_ids.shape
        x = self.token_embedding(input_ids)
        x = self.embd_dropout(x)
        mask = create_causal_mask(seq_len, input_ids.device)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            logits_flat = logits.contiguous().view(-1, self.config.vocab_size)
            targets_flat = targets.contiguous().view(-1)
            loss = F.cross_entropy(logits_flat, targets_flat)
        return logits, loss

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())
    

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        self.eval()
        for _ in range(max_new_tokens):
            if input_ids.shape[1] > self.config.max_seq_len:
                input_ids = input_ids[:, -self.config.max_seq_len:]
            logits, _ = self.forward(input_ids)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = float('-inf')
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cumulative_probs > top_p
                sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
                sorted_mask[:, 0] = False
                mask = sorted_mask.scatter(1, sorted_indices, sorted_mask)
                logits[mask] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids