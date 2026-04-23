import torch
import torch.nn as nn


def get_prunable_layers(model):
    """Return all linear layers that should be pruned (not embeddings, not LayerNorm)."""
    prunable = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "lm_head" not in name:
            prunable.append((name, module))
    return prunable


def compute_sparsity(current_step, warmup_steps, sparsity_start_step, sparsity_end_step,
                     target_sparsity):
    """Calculate current sparsity level based on training progress.

    - Before warmup_steps: 0% (train normally)
    - warmup_steps to sparsity_end_step: ramp from 0% to target_sparsity (cubic schedule)
    - After sparsity_end_step: hold at target_sparsity
    """
    if current_step < sparsity_start_step:
        return 0.0
    if current_step >= sparsity_end_step:
        return target_sparsity

    # Cubic ramp — starts slow, accelerates, then slows near target
    # This is the standard schedule from Zhu & Gupta (2017) "To prune, or not to prune"
    progress = (current_step - sparsity_start_step) / (sparsity_end_step - sparsity_start_step)
    return target_sparsity * (1 - (1 - progress) ** 3)


def compute_masks(model, current_sparsity):
    """Compute binary masks for all prunable layers based on weight magnitudes.

    For each layer: rank weights by absolute value, keep the top (1 - sparsity) fraction.
    """
    masks = {}
    if current_sparsity <= 0:
        return masks

    for name, module in get_prunable_layers(model):
        weights = module.weight.data
        # Number of weights to keep
        n_total = weights.numel()
        n_keep = int(n_total * (1 - current_sparsity))
        n_keep = max(n_keep, 1)  # always keep at least 1 weight

        # Find the magnitude threshold: keep weights above this value
        magnitudes = weights.abs().flatten()
        threshold = magnitudes.kthvalue(n_total - n_keep).values

        # Binary mask: 1 = keep, 0 = prune
        mask = (weights.abs() >= threshold).float()
        masks[name] = mask

    return masks


def apply_masks(model, masks):
    """Zero out pruned weights by multiplying with masks."""
    if not masks:
        return
    for name, module in get_prunable_layers(model):
        if name in masks:
            module.weight.data *= masks[name]


class SparsityScheduler:
    """Manages the full sparsity lifecycle during training.

    Usage:
        scheduler = SparsityScheduler(model, target_sparsity=0.7,
                                       warmup_steps=2000, total_steps=10000)
        for step in range(total_steps):
            loss.backward()
            optimizer.step()
            scheduler.step(step)  # recompute masks and apply
    """
    def __init__(self, model, target_sparsity, warmup_steps, total_steps):
        self.model = model
        self.target_sparsity = target_sparsity
        # Pruning starts after warmup, ends at 80% of total training
        self.sparsity_start_step = warmup_steps
        self.sparsity_end_step = int(total_steps * 0.8)
        self.masks = {}
        self.current_sparsity = 0.0

    def step(self, current_step, recompute_every=100):
        """Call after each optimizer step.

        Recomputes masks every `recompute_every` steps (not every step, for efficiency).
        Always applies existing masks to prevent pruned weights from reviving.
        """
        self.current_sparsity = compute_sparsity(
            current_step, self.sparsity_start_step, self.sparsity_start_step,
            self.sparsity_end_step, self.target_sparsity
        )

        # Recompute which weights to prune periodically
        if self.current_sparsity > 0 and current_step % recompute_every == 0:
            self.masks = compute_masks(self.model, self.current_sparsity)

        # Always apply masks to keep pruned weights at zero
        apply_masks(self.model, self.masks)

    def get_sparsity_stats(self):
        """Return actual sparsity per layer (for logging)."""
        stats = {}
        for name, module in get_prunable_layers(self.model):
            weights = module.weight.data
            n_zero = (weights == 0).sum().item()
            n_total = weights.numel()
            stats[name] = n_zero / n_total
        return stats
