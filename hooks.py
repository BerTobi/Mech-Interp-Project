import torch
from contextlib import contextmanager
from model import GPT


class ActivationCache:
    """Captures activations from a GPT model using forward hooks.

    Records the residual stream after each transformer block, plus
    the attention and FFN sub-layer outputs individually.

    Usage:
        cache = ActivationCache(model)
        with cache.capture():
            logits, loss = model(input_ids)
        acts = cache.activations  # dict of {name: tensor}
    """

    # Which sub-modules to hook, relative to each TransformerBlock
    HOOK_POINTS = {
        "attn": "attn",       # attention output (before residual add)
        "ffn": "ffn",         # FFN output (before residual add)
    }

    def __init__(self, model: GPT, layers=None):
        """
        Args:
            model: A GPT model instance.
            layers: Which block indices to capture (e.g. [0, 3, 5]).
                    None = all layers.
        """
        self.model = model
        self.n_layers = len(model.blocks)
        self.layers = layers if layers is not None else list(range(self.n_layers))
        self.activations = {}
        self._hooks = []

    def _make_hook(self, name):
        """Create a hook function that stores the output under `name`."""
        def hook_fn(module, input, output):
            # Detach so we don't hold onto the computation graph
            self.activations[name] = output.detach()
        return hook_fn

    @contextmanager
    def capture(self):
        """Context manager that registers hooks, yields, then removes them."""
        self.activations = {}
        try:
            # Hook the residual stream = output of each full TransformerBlock
            for i in self.layers:
                block = self.model.blocks[i]
                name = f"block_{i}"
                h = block.register_forward_hook(self._make_hook(name))
                self._hooks.append(h)

                # Hook attention and FFN sub-layers within each block
                for label, attr in self.HOOK_POINTS.items():
                    submodule = getattr(block, attr)
                    sub_name = f"block_{i}.{label}"
                    h = submodule.register_forward_hook(self._make_hook(sub_name))
                    self._hooks.append(h)

            yield self
        finally:
            for h in self._hooks:
                h.remove()
            self._hooks = []


def extract_activations(model, dataloader, device, layers=None, max_batches=10):
    """Run data through the model and collect activations across batches.

    Returns:
        activations: dict of {hook_name: tensor} with batch dim concatenated
        all_tokens:  (N, seq_len) tensor of input token ids
    """
    model.eval()
    cache = ActivationCache(model, layers=layers)

    collected = {}  # name -> list of tensors
    token_batches = []

    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            if i >= max_batches:
                break
            x = x.to(device)
            with cache.capture():
                model(x)

            token_batches.append(x.cpu())
            for name, act in cache.activations.items():
                if name not in collected:
                    collected[name] = []
                collected[name].append(act.cpu())

    # Concatenate along batch dimension
    activations = {name: torch.cat(tensors, dim=0) for name, tensors in collected.items()}
    all_tokens = torch.cat(token_batches, dim=0)

    return activations, all_tokens
