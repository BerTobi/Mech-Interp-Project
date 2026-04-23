"""Generate text from a trained checkpoint."""
import torch
import argparse
from model import GPT
from transformers import AutoTokenizer


def generate(model, tokenizer, prompt, max_new_tokens=100, temperature=0.8,
             device="cuda"):
    model.eval()
    token_ids = tokenizer.encode(prompt)
    x = torch.tensor([token_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop to max_seq_len if needed
            x_cond = x[:, -model.max_seq_len:]
            logits, _ = model(x_cond)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, next_token], dim=1)

    return tokenizer.decode(x[0].tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/model_dense.pt")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = GPT(**config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories")

    print(f"Prompt: {args.prompt}\n")
    text = generate(model, tokenizer, args.prompt, args.tokens, args.temperature, device)
    print(text)
