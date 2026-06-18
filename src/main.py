import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
from dataclasses import dataclass
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from config import GPTConfig
from utils import find_device
from tokenizer import SimpleTokenizer
from dataset import load_training_data, TextDataset
from gpt import GPT
from optimizer import create_optimizer
from lr_scheduler import CosineWarmupScheduler

def train(model, train_dataset, config, device, steps=5000):
    model = model.to(device)
    model.train()
    dataloader = DataLoader(train_dataset, batch_size=config.batch_size,
                            shuffle=True, drop_last=True)
    optimizer = create_optimizer(model, config)
    scheduler = CosineWarmupScheduler(optimizer, config.warmup_steps, steps, max_lr=config.learning_rate)
    use_amp = device.type != 'cpu'
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp) if use_amp else None
    step = 0
    loss_history = []
    start = time.time()

    while step < steps:
        for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
            if step >= steps:
                break
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            with torch.amp.autocast(device.type, enabled=use_amp):
                _, loss = model(input_ids, target_ids)
            loss = loss / config.grad_accum_steps
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if (batch_idx + 1) % config.grad_accum_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                step += 1
                if step % 50 == 0 or step == 1:
                    elapsed = time.time() - start
                    print(f"Step {step:>4d} | Loss: {loss.item() * config.grad_accum_steps:.4f} | Time: {elapsed:.0f}s")
                    loss_history.append((step, loss.item() * config.grad_accum_steps))
    print(f"\nDone. {time.time() - start:.0f}s total.")
    return loss_history
                


def plot_loss(loss_history, save_path="loss_curve.png"):
    steps, losses = zip(*loss_history)
    plt.figure(figsize=(10, 4))
    plt.plot(steps, losses)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"Loss curve saved to {save_path}")


def main():
    print('How to train your GPT')

    # tiny model
    # config = GPTConfig(
    #     d_model=256, num_heads=4, num_layers=4, max_seq_len=128,
    #     batch_size=4, grad_accum_steps=2, max_steps=50000,
    #     warmup_steps=50, learning_rate=3e-4,
    # )

    # small model (gpt2 scale, needs GPU)
    # config = GPTConfig(
    #     d_model=768, num_heads=12, num_layers=12, max_seq_len=1024,
    #     batch_size=4, grad_accum_steps=8, max_steps=50000,
    #     warmup_steps=2000, learning_rate=3e-4,
    # )

    config = GPTConfig(
        d_model=768, num_heads=6, num_layers=6, max_seq_len=512,
        batch_size=4, grad_accum_steps=8, max_steps=50000,
        warmup_steps=2000, learning_rate=3e-4,
    )

    device = find_device()
    tokenizer = SimpleTokenizer()
    print('Loading training data...')
    texts = load_training_data(max_samples=5000)
    train_dataset = TextDataset(texts, tokenizer, max_seq_len=config.max_seq_len)

    print('creating model ....')
    model = GPT(config=config)
    print(f'Parameters: {model.get_num_params}')
    print('\n' + '='*50)
    loss_history = train(model, train_dataset, config, device, steps=config.max_steps)
    plot_loss(loss_history)

    print("\n" + "=" * 50)
    print("GENERATING")
    print("=" * 50 + "\n")

    prompts = [
        "The history of artificial intelligence",
        "In the beginning the universe",
        "The most important scientific discovery",
    ]

    for prompt in prompts:
        input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        output_ids = model.generate(input_ids, max_new_tokens=50, temperature=0.8, top_k=50)
        text = tokenizer.decode(output_ids[0].tolist())
        print(f"Prompt: {prompt}")
        print(f"Output: {text}")
        print("-" * 50)
        print()

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
    }, "checkpoints/model.pt")
    print("Model saved to checkpoints/model.pt")


if __name__ == "__main__":
    main()