import torch
import time
import os
from optimizer import create_optimizer
from lr_scheduler import CosineWarmupScheduler

def train(model, train_dataset, config, device, save_dir='checkpoints'):
    """
    The main training loop
    Iterates: forward -> backward -> update, logging and saving periodically.
    """
    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device)
    model.train()

    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True, drop_last=True, num_workers=4, pin_memory=True,
    )

    optimizer = create_optimizer(model, config)
    scheduler = CosineWarmupScheduler(
        optimizer=optimizer, warmup_steps=config.warmup_steps,
        max_steps=config.max_steps, max_lr=config.learning_rate,
    )

    use_amp = device.type != 'cpu'
    scaler = torch.amp.grad_scaler(device.type, enabled=use_amp) if use_amp else None

    step = 0
    total_loss = 0.0
    loss_history = []
    best_loss = float('inf')
    start_time = time.time()


    print(f"\n{'='*60}")
    print(f"Training! Params: {model.get_num_params():,} | Device: {device}")
    print(f"Effective batch: {config.batch_size * config.grad_accum_steps}")
    print(f"{'='*60}\n")

    while step < config.max_steps:
        for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
            if step >= config.max_steps:
                break
            input_ids = input_ids.to(device, non_blocking=True)
            target_ids = target_ids.to(device, non_blocking=True)

            # forward: predict next token and measure error
            with torch.amp.autocast_mode(device.type, enabled=use_amp):
                _, loss = model(input_ids, targets=target_ids)
            loss = loss / config.grad_accum_steps

            # backward: calculate how to improve
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            total_loss += loss.item() * config.grad_accum_steps

            # update: every grad_accum_steps, optimize
            if (batch_idx + 1) % config.accum_grad_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                if scaler:
                    scaler.step(optimizer); scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad()
                scheduler.step()
                step += 1

                # Logging every 100 steps
                if step % 100 == 0 or step == 1:
                    avg_loss = total_loss / (100 if step > 0 else 1)
                    elapsed = time.time() - start_time
                    tps = (step * config.batch_size * config.grad_accum_steps
                           * config.max_seq_len) / elapsed
                    print(f"Step {step:>6,}/{config.max_steps:,} | "
                          f"Loss: {avg_loss:.4f} | LR: {scheduler.get_lr():.2e} | "
                          f"Toks/sec: {tps:,.0f}")
                    loss_history.append((step, avg_loss))
                    total_loss = 0.0

                # Save checkpoint every 5000 steps
                if step % 5000 == 0:
                    checkpoint = {
                        "step": step, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "loss": avg_loss, "config": config,
                    }
                    torch.save(checkpoint, f"{save_dir}/checkpoint_step_{step}.pt")
                    print(f"   Saved checkpoint at step {step}")
                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        torch.save(checkpoint, f"{save_dir}/best_model.pt")

                
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Done! {total_time/60:.1f} min | Best loss: {best_loss:.4f}")
        print(f"{'='*60}\n")
        return loss_history
    

def plot_loss(loss_history, save_path="loss_curve.png"):
    """
    WHAT: Visualize training progress.
    WHY: Loss curves diagnose problems:
         ↘ Steady decrease: training is working
         → Flat line: stalled (higher LR, check data)
         ↗ Increasing: overfitting (more dropout, weight decay)
         ⚡ Spikes: unstable (lower LR, longer warmup)
    """
    import matplotlib.pyplot as plt
    steps, losses = zip(*loss_history)
    plt.figure(figsize=(10, 5))
    plt.plot(steps, losses)
    plt.xlabel("Training Step"); plt.ylabel("Loss")
    plt.title("GPT Training Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"Loss curve saved to {save_path}")