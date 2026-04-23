import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from hooks import ActivationCache


# ---------------------------------------------------------------------------
# Keyword-based labeling for TinyStories passages
# ---------------------------------------------------------------------------

CHARACTER_TYPES = {
    "boy":    ["boy", "prince", "king"],
    "girl":   ["girl", "princess", "queen"],
    "dog":    ["dog", "puppy", "puppies"],
    "cat":    ["cat", "kitten", "kittens"],
    "bird":   ["bird", "parrot", "eagle"],
    "bear":   ["bear", "teddy bear"],
    "rabbit": ["rabbit", "bunny", "bunnies"],
}

VALENCE_KEYWORDS = {
    "positive": ["happy", "glad", "excited", "smiled", "laughed", "joy",
                 "love", "fun", "wonderful", "great", "friend"],
    "negative": ["sad", "cried", "angry", "scared", "afraid", "upset",
                 "worried", "alone", "lost", "hurt", "cry"],
}


def label_character_type(text):
    """Assign a character-type label based on keyword presence.

    Returns the character type with the most keyword hits, or None if
    no keywords are found (story is skipped).
    """
    text_lower = text.lower()
    scores = {}
    for char_type, keywords in CHARACTER_TYPES.items():
        scores[char_type] = sum(text_lower.count(kw) for kw in keywords)
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return None
    return best


def label_valence(text):
    """Assign positive/negative valence based on keyword counts.

    Returns "positive", "negative", or None if neither dominates.
    """
    text_lower = text.lower()
    pos = sum(text_lower.count(kw) for kw in VALENCE_KEYWORDS["positive"])
    neg = sum(text_lower.count(kw) for kw in VALENCE_KEYWORDS["negative"])
    if pos > neg:
        return "positive"
    elif neg > pos:
        return "negative"
    return None


# Story phase uses a different approach: instead of labeling whole passages,
# we split each story into thirds (beginning/middle/end) and probe whether
# the model's activations at each third encode narrative position.
# This is handled separately in build_labeled_passages.

PHASE_KEYWORDS = {
    "beginning": ["once upon", "there was", "there lived", "one day",
                  "one morning", "long ago"],
    "middle":    ["then", "suddenly", "but then", "so he", "so she",
                  "decided to", "wanted to", "tried to"],
    "end":       ["happily", "the end", "from that day", "learned",
                  "never again", "ever after", "and they"],
}


def label_story_phase(text):
    """Assign beginning/middle/end based on which phase keywords dominate.

    TinyStories follows a predictable arc: setup -> conflict -> resolution.
    We use keywords typical of each phase.
    """
    text_lower = text.lower()
    scores = {}
    for phase, keywords in PHASE_KEYWORDS.items():
        scores[phase] = sum(text_lower.count(kw) for kw in keywords)
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return None
    # Require the winning phase to have at least 2 more hits than runner-up
    sorted_scores = sorted(scores.values(), reverse=True)
    if sorted_scores[0] - sorted_scores[1] < 2:
        return None
    return best


# ---------------------------------------------------------------------------
# Build a labeled dataset of (passage_text, label) pairs
# ---------------------------------------------------------------------------

def build_labeled_passages(task="character_type", max_passages=2500,
                           max_scan=200_000):
    """Load TinyStories and label passages with keyword heuristics.

    Scans up to max_scan stories to find enough examples, then balances
    classes by undersampling to match the smallest class.

    Args:
        task: "character_type", "valence", or "story_phase"
        max_passages: target total after balancing
        max_scan: max stories to scan for labels
    Returns:
        texts: list of story strings
        labels: list of integer labels
        label_names: list mapping label index -> human-readable name
    """
    dataset = load_dataset("roneneldan/TinyStories", split="train")

    labelers = {
        "character_type": (label_character_type, list(CHARACTER_TYPES.keys())),
        "valence": (label_valence, ["positive", "negative"]),
        "story_phase": (label_story_phase, ["beginning", "middle", "end"]),
    }
    labeler, label_names = labelers[task]
    name_to_id = {name: i for i, name in enumerate(label_names)}

    # Collect into per-class buckets
    buckets = {i: [] for i in range(len(label_names))}
    n_scanned = 0
    for example in dataset:
        text = example["text"]
        label = labeler(text)
        n_scanned += 1
        if label is None:
            continue
        buckets[name_to_id[label]].append(text)
        if n_scanned >= max_scan:
            break

    # Drop classes with too few examples
    min_count = 10
    keep_ids = {idx for idx, texts in buckets.items() if len(texts) >= min_count}
    if len(keep_ids) < len(name_to_id):
        dropped = [n for n, i in name_to_id.items() if i not in keep_ids]
        print(f"Dropping rare classes: {dropped}")

    # Remap to contiguous ids
    kept_sorted = sorted(keep_ids)
    label_names = [label_names[old] for old in kept_sorted]
    name_to_id = {name: i for i, name in enumerate(label_names)}

    # Balance: undersample to match the smallest class
    per_class = min(len(buckets[idx]) for idx in kept_sorted)
    target_per_class = min(per_class, max_passages // len(kept_sorted))

    texts = []
    labels = []
    rng = np.random.RandomState(42)
    for new_id, old_id in enumerate(kept_sorted):
        chosen = rng.choice(len(buckets[old_id]), size=target_per_class, replace=False)
        for i in chosen:
            texts.append(buckets[old_id][i])
            labels.append(new_id)

    # Shuffle so classes aren't in order
    order = rng.permutation(len(texts))
    texts = [texts[i] for i in order]
    labels = [labels[i] for i in order]

    print(f"Labeled {len(texts)} passages for '{task}' "
          f"(scanned {n_scanned}, {target_per_class} per class)")
    for name, idx in name_to_id.items():
        count = labels.count(idx)
        print(f"  {name}: {count} ({count/len(labels)*100:.1f}%)")

    return texts, labels, label_names


# ---------------------------------------------------------------------------
# Extract one activation vector per passage (mean-pool over tokens)
# ---------------------------------------------------------------------------

def extract_passage_activations(model, texts, tokenizer, device, max_seq_len=256):
    """Tokenize each passage, run through the model, mean-pool activations.

    Returns:
        activations: dict of {layer_name: np.array of shape (N, d_model)}
    """
    model.eval()
    cache = ActivationCache(model)

    collected = {}  # layer_name -> list of (d_model,) arrays

    with torch.no_grad():
        for text in texts:
            token_ids = tokenizer.encode(text, truncation=True,
                                         max_length=max_seq_len)
            x = torch.tensor([token_ids], dtype=torch.long, device=device)

            with cache.capture():
                model(x)

            for name, act in cache.activations.items():
                # act shape: (1, seq_len, d_model) -> mean over seq_len
                pooled = act[0].mean(dim=0).cpu().numpy()
                if name not in collected:
                    collected[name] = []
                collected[name].append(pooled)

    activations = {name: np.stack(vecs) for name, vecs in collected.items()}
    return activations


# ---------------------------------------------------------------------------
# Train and evaluate sklearn LogisticRegression probes
# ---------------------------------------------------------------------------

def train_probe(X, y, test_size=0.2, max_iter=1000):
    """Train a LogisticRegression probe on activations X with labels y.

    Returns:
        dict with train_acc, val_acc, and the fitted model
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    probe = LogisticRegression(max_iter=max_iter, solver="lbfgs")
    probe.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, probe.predict(X_train))
    val_acc = accuracy_score(y_val, probe.predict(X_val))

    return {"train_acc": train_acc, "val_acc": val_acc, "probe": probe}


def run_probes(model, device, tasks=("character_type",), max_passages=2500,
               max_seq_len=256):
    """Full probing pipeline: label passages, extract activations, train probes.

    Args:
        model: a trained GPT model
        tasks: which probe tasks to run
        max_passages: passages per task
    Returns:
        {task_name: {layer_name: {train_acc, val_acc}}}
    """
    tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_results = {}

    for task in tasks:
        print(f"\n--- Probe task: {task} ---")
        texts, labels, label_names = build_labeled_passages(task, max_passages)
        labels = np.array(labels)

        print("Extracting activations...")
        activations = extract_passage_activations(
            model, texts, tokenizer, device, max_seq_len
        )

        # Only probe residual stream outputs (block_i), not sub-layers
        block_keys = sorted(k for k in activations if "." not in k)

        task_results = {}
        for layer_name in block_keys:
            X = activations[layer_name]
            res = train_probe(X, labels)
            task_results[layer_name] = {
                "train_acc": res["train_acc"],
                "val_acc": res["val_acc"],
            }
            print(f"  {layer_name}: train={res['train_acc']:.3f}  "
                  f"val={res['val_acc']:.3f}")

        all_results[task] = task_results

    return all_results
