import os
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

CACHE_DIR = "token_cache"


class TinyStoriesDataset(Dataset):
    """Loads TinyStories and serves tokenized chunks of fixed sequence length."""

    def __init__(self, split="train", max_tokens=160_000_000, seq_len=256,
                 tokenizer_name="roneneldan/TinyStories"):
        self.seq_len = seq_len
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # GPT-2 tokenizer has no pad token by default
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Try loading cached tokens
        os.makedirs(CACHE_DIR, exist_ok=True)
        cache_path = os.path.join(CACHE_DIR, f"{split}_{max_tokens}.pt")

        if os.path.exists(cache_path):
            print(f"Loading cached tokens from {cache_path}...")
            self.tokens = torch.load(cache_path, weights_only=True)
        else:
            dataset = load_dataset("roneneldan/TinyStories", split=split)

            # Batch-tokenize (runs in Rust, much faster than one-by-one)
            print(f"Tokenizing {split} split...")
            batch_size = 10_000
            all_tokens = []
            texts = dataset["text"]
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                encoded = self.tokenizer(batch, add_special_tokens=False)
                for ids in encoded["input_ids"]:
                    all_tokens.extend(ids)
                n_tokens = len(all_tokens)
                print(f"  {n_tokens:,} / {max_tokens:,} tokens "
                      f"({n_tokens / max_tokens * 100:.0f}%)", end="\r")
                if n_tokens >= max_tokens:
                    break

            print()
            all_tokens = all_tokens[:max_tokens]
            self.tokens = torch.tensor(all_tokens, dtype=torch.long)
            torch.save(self.tokens, cache_path)
            print(f"Cached tokens to {cache_path}")

        # Number of complete sequences we can make
        self.n_sequences = (len(self.tokens) - 1) // seq_len

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = self.tokens[start : start + self.seq_len]
        y = self.tokens[start + 1 : start + self.seq_len + 1]
        return x, y

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size


def create_dataloaders(batch_size=64, seq_len=256, max_tokens=160_000_000,
                       num_workers=0):
    """Create train and validation dataloaders."""
    train_dataset = TinyStoriesDataset(
        split="train", max_tokens=max_tokens, seq_len=seq_len
    )
    val_dataset = TinyStoriesDataset(
        split="validation", max_tokens=5_000_000, seq_len=seq_len
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False
    )

    return train_loader, val_loader, train_dataset.vocab_size
