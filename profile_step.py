"""Profile compiled vs uncompiled model."""
import torch
import time
from model import GPT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = 50257

x = torch.randint(0, vocab_size, (64, 256), device=device)
y = torch.randint(0, vocab_size, (64, 256), device=device)


def bench(model, label, n_steps=20):
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    model.train()

    # Warmup (3 steps)
    for _ in range(3):
        with torch.amp.autocast("cuda"):
            _, loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    torch.cuda.synchronize()

    # Timed run
    t0 = time.perf_counter()
    for _ in range(n_steps):
        with torch.amp.autocast("cuda"):
            _, loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    ms_per_step = elapsed / n_steps * 1000
    tok_per_sec = n_steps * 64 * 256 / elapsed
    print(f"  {label}: {ms_per_step:.0f} ms/step, {tok_per_sec:,.0f} tok/sec")


# Eager mode
model = GPT(vocab_size=vocab_size, max_seq_len=256,
            d_model=128, n_heads=4, n_layers=6, d_ff=512).to(device)
bench(model, "eager")

# Compiled mode
model2 = GPT(vocab_size=vocab_size, max_seq_len=256,
             d_model=128, n_heads=4, n_layers=6, d_ff=512).to(device)
model2 = torch.compile(model2)
bench(model2, "compiled")
