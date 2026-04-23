import torch
import time
import math
import os
from model import GPT
from data import TinyStoriesDataset, create_dataloaders
from sparsity import SparsityScheduler


def get_lr(step, warmup_steps, max_steps, max_lr=3e-4, min_lr=3e-5):
    """Cosine learning rate schedule with linear warmup."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, val_loader, device, max_batches=50):
    """Compute validation loss and perplexity."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for x, y in val_loader:
        if n_batches >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            _, loss = model(x, y)
        total_loss += loss.item()
        n_batches += 1
    model.train()
    avg_loss = total_loss / n_batches
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity


def train(
    # Model
    d_model=128,
    n_heads=4,
    n_layers=6,
    d_ff=512,
    max_seq_len=256,
    # Training
    batch_size=64,
    max_tokens=160_000_000,
    n_epochs=1,
    max_lr=3e-4,
    min_lr=3e-5,
    weight_decay=0.1,
    grad_clip=1.0,
    # Sparsity
    target_sparsity=0.0,
    # Logging
    log_every=100,
    eval_every=500,
    save_dir="checkpoints",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    print("Loading data...")
    train_dataset = TinyStoriesDataset(
        split="train", max_tokens=max_tokens, seq_len=max_seq_len
    )
    val_dataset = TinyStoriesDataset(
        split="validation", max_tokens=5_000_000, seq_len=max_seq_len
    )
    from torch.utils.data import DataLoader
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    vocab_size = train_dataset.vocab_size
    n_sequences = len(train_dataset)
    steps_per_epoch = n_sequences // batch_size
    total_steps = steps_per_epoch * n_epochs
    warmup_steps = int(total_steps * 0.1)
    print(f"Vocab size: {vocab_size}, Total steps: {total_steps}, Warmup: {warmup_steps}")

    # Model
    model = GPT(
        vocab_size=vocab_size, max_seq_len=max_seq_len,
        d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Compile for fused kernels (requires Triton — use WSL on Windows)
    if device.type == "cuda":
        model = torch.compile(model)
        print("Model compiled with torch.compile")

    # Optimizer — AdamW with weight decay on non-bias, non-LayerNorm params
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)
    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=max_lr)

    # Sparsity scheduler
    sparsity_scheduler = SparsityScheduler(
        model, target_sparsity=target_sparsity,
        warmup_steps=warmup_steps, total_steps=total_steps
    )

    # Mixed precision
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # Training loop
    os.makedirs(save_dir, exist_ok=True)
    global_step = 0
    model.train()
    t_start = time.time()

    # Sample batches directly from the token tensor — no DataLoader, no epochs,
    # no iterator resets that trigger torch.compile retracing
    tokens = train_dataset.tokens
    # Pre-build an index tensor: (n_sequences, max_seq_len) where each row
    # is [start, start+1, ..., start+seq_len-1]
    offsets = torch.arange(max_seq_len).unsqueeze(0)  # (1, seq_len)
    base = (torch.arange(n_sequences) * max_seq_len).unsqueeze(1)  # (n_seq, 1)
    all_x_idx = base + offsets          # (n_seq, seq_len)
    all_y_idx = base + offsets + 1      # (n_seq, seq_len)

    for global_step in range(total_steps):
        batch_idx = torch.randint(0, n_sequences, (batch_size,))
        x = tokens[all_x_idx[batch_idx]]
        y = tokens[all_y_idx[batch_idx]]
        x, y = x.to(device), y.to(device)

        # Forward + backward (fp16 on GPU)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            _, loss = model(x, y)
        scaler.scale(loss).backward()

        # Gradient clipping (unscale first so clip threshold is in fp32)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # Update LR
        lr = get_lr(global_step, warmup_steps, total_steps, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        # Apply sparsity masks after optimizer step
        sparsity_scheduler.step(global_step)

        # Logging
        if global_step % log_every == 0:
            elapsed = time.time() - t_start
            steps_left = total_steps - global_step
            if global_step > 0:
                eta = elapsed / global_step * steps_left
                eta_str = f"{eta/60:.0f}m"
            else:
                eta_str = "..."
            sparsity_pct = sparsity_scheduler.current_sparsity * 100
            print(f"Step {global_step:>6d}/{total_steps} | Loss {loss.item():.4f} | "
                  f"LR {lr:.2e} | Sparsity {sparsity_pct:.1f}% | "
                  f"Elapsed {elapsed/60:.1f}m | ETA {eta_str}")

        # Evaluation
        if global_step % eval_every == 0:
            val_loss, val_ppl = evaluate(model, val_loader, device)
            print(f"  → Val loss {val_loss:.4f} | Perplexity {val_ppl:.2f}")

    # Final evaluation
    val_loss, val_ppl = evaluate(model, val_loader, device)
    print(f"\nFinal — Val loss {val_loss:.4f} | Perplexity {val_ppl:.2f}")

    # Log final sparsity stats
    if target_sparsity > 0:
        stats = sparsity_scheduler.get_sparsity_stats()
        print("\nFinal sparsity per layer:")
        for name, sparsity in stats.items():
            print(f"  {name}: {sparsity*100:.1f}%")

    # Save checkpoint
    sparsity_tag = f"sparse{int(target_sparsity*100)}" if target_sparsity > 0 else "dense"
    save_path = os.path.join(save_dir, f"model_{sparsity_tag}_{n_epochs}ep.pt")
    # Strip _orig_mod. prefix added by torch.compile
    state_dict = {k.replace("_orig_mod.", ""): v
                  for k, v in model.state_dict().items()}
    torch.save({
        "model_state_dict": state_dict,
        "config": {
            "vocab_size": vocab_size, "max_seq_len": max_seq_len,
            "d_model": d_model, "n_heads": n_heads,
            "n_layers": n_layers, "d_ff": d_ff,
        },
        "val_loss": val_loss,
        "val_perplexity": val_ppl,
        "target_sparsity": target_sparsity,
    }, save_path)
    print(f"\nSaved checkpoint to {save_path}")

    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sparsity", type=float, default=0.0,
                        help="Target sparsity (0.0 = dense, 0.7 = 70%% sparse)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    train(target_sparsity=args.sparsity, n_epochs=args.epochs,
          batch_size=args.batch_size)
