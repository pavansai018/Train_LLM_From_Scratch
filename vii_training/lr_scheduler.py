import math

class CosineWarmupScheduler:
    """
    3 phase learning rate schedule
    warmup prevents early instability. Cosine decay provides
    smooth convergence. Minimum floor prevents zero learning

    Phase 1: (warmup): LR: 0 -> max_lr (linear increase over warmup_steps)
    Phase 2: (Decay): LR: max_lr -> min_lr (cosine curve)
    Phase 3: (minimum): LR: min_lr (constant)
    """

    def __init__(self, optimizer, warmup_steps, max_steps, max_lr: float = 3e-4, min_lr: float = 1e-5):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.current_step = 0

    def get_lr(self)-> float:
        step = self.current_step
        if step < self.warmup_steps:
            return self.max_lr * step / self.warmup_steps
        if step < self.max_steps:
            progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.min_lr + (self.max_lr - self.min_lr) * cosine_decay
        return self.min_lr
    
    def step(self):
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.current_step += 1
    
    def state_dict(self):
        return {'current_step': self.current_step}
    
    def load_state_dict(self, state_dict):
        self.current_step = state_dict['current_step']

    