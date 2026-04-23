import torch
import torch.nn.functional as F
import argparse
from model import GPT
from data import create_dataloaders
from hooks import extract_activations
from probes import run_probes


def load_model(checkpoint_path, device):
    """Load a GPT model from a training checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = GPT(**config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, config, ckpt


def cosine_similarity_per_layer(acts_a, acts_b):
    """Compute mean cosine similarity between two models' activations per layer."""
    results = {}
    for name in acts_a:
        if name not in acts_b:
            continue
        a = acts_a[name].flatten(0, 1)
        b = acts_b[name].flatten(0, 1)
        cos = F.cosine_similarity(a, b, dim=-1)
        results[name] = cos.mean().item()
    return results


def linear_cka(acts_a, acts_b):
    """Compute Linear CKA between activations (Kornblith et al., 2019)."""
    results = {}
    for name in acts_a:
        if name not in acts_b:
            continue
        a = acts_a[name].flatten(0, 1).float()
        b = acts_b[name].flatten(0, 1).float()
        if a.shape[0] > 5000:
            idx = torch.randperm(a.shape[0])[:5000]
            a, b = a[idx], b[idx]

        a = a - a.mean(dim=0)
        b = b - b.mean(dim=0)

        ab = torch.norm(a.T @ b, p="fro") ** 2
        aa = torch.norm(a.T @ a, p="fro")
        bb = torch.norm(b.T @ b, p="fro")

        cka = (ab / (aa * bb + 1e-8)).item()
        results[name] = cka
    return results


def compare_models(checkpoints, batch_size=32, max_batches=5,
                   tasks=("character_type",), max_passages=2500):
    """Full comparison pipeline across multiple checkpoints.

    Args:
        checkpoints: list of (label, path) tuples, e.g.
                     [("dense", "checkpoints/model_dense.pt"), ...]
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load all models
    print("Loading models...")
    models = {}
    for label, path in checkpoints:
        model, config, ckpt = load_model(path, device)
        ppl = ckpt.get("val_perplexity", None)
        ppl_str = f"{ppl:.2f}" if ppl else "?"
        print(f"  {label}: sparsity={ckpt.get('target_sparsity', 0):.0%}, "
              f"val_ppl={ppl_str}")
        models[label] = {"model": model, "config": config, "ckpt": ckpt}

    seq_len = list(models.values())[0]["config"]["max_seq_len"]

    # Shared data
    print("Loading data...")
    _, val_loader, _ = create_dataloaders(
        batch_size=batch_size, seq_len=seq_len, max_tokens=5_000_000
    )

    # ---- 1. Extract activations for all models ----
    print("\nExtracting activations...")
    all_acts = {}
    for label, info in models.items():
        acts, _ = extract_activations(
            info["model"], val_loader, device, max_batches=max_batches
        )
        all_acts[label] = acts

    # ---- 2. Pairwise similarity vs the first model (baseline) ----
    baseline_label = checkpoints[0][0]
    block_keys = sorted(k for k in all_acts[baseline_label] if "." not in k)
    baseline_blocks = {k: all_acts[baseline_label][k] for k in block_keys}

    for label in all_acts:
        if label == baseline_label:
            continue
        other_blocks = {k: all_acts[label][k] for k in block_keys}

        print(f"\n=== {baseline_label} vs {label}: Cosine Similarity ===")
        cos = cosine_similarity_per_layer(baseline_blocks, other_blocks)
        for name in sorted(cos):
            print(f"  {name}: {cos[name]:.4f}")

        print(f"\n=== {baseline_label} vs {label}: Linear CKA ===")
        cka = linear_cka(baseline_blocks, other_blocks)
        for name in sorted(cka):
            print(f"  {name}: {cka[name]:.4f}")

    # ---- 3. Probes for all models ----
    all_probe_results = {}
    for label, info in models.items():
        print(f"\n=== Probes: {label} ===")
        all_probe_results[label] = run_probes(
            info["model"], device, tasks=tasks, max_passages=max_passages,
            max_seq_len=seq_len
        )

    # ---- 4. Summary table ----
    labels = [l for l, _ in checkpoints]
    header_models = "".join(f"{l:>10}" for l in labels)
    print(f"\n=== Probe Accuracy Comparison (val) ===")
    print(f"{'Task':<18} {'Layer':<12} {header_models}")
    print("-" * (30 + 10 * len(labels)))

    for task in tasks:
        for layer in block_keys:
            accs = []
            for label in labels:
                acc = all_probe_results[label].get(task, {}).get(
                    layer, {}).get("val_acc", 0)
                accs.append(acc)
            acc_str = "".join(f"{a:>10.3f}" for a in accs)
            print(f"  {task:<16} {layer:<12} {acc_str}")

    return all_probe_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare model representations across sparsity levels"
    )
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="label:path pairs, e.g. dense:checkpoints/model_dense.pt")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_batches", type=int, default=5)
    parser.add_argument("--tasks", nargs="+",
                        default=["character_type"],
                        choices=["character_type", "valence", "story_phase"],
                        help="Probe tasks to run")
    parser.add_argument("--max_passages", type=int, default=2500)
    args = parser.parse_args()

    # Parse label:path pairs
    checkpoints = []
    for item in args.checkpoints:
        if ":" not in item:
            parser.error(f"Expected label:path format, got '{item}'")
        label, path = item.split(":", 1)
        checkpoints.append((label, path))

    compare_models(checkpoints,
                   batch_size=args.batch_size, max_batches=args.max_batches,
                   tasks=tuple(args.tasks), max_passages=args.max_passages)
