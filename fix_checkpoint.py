import torch
import sys

path = sys.argv[1]
ckpt = torch.load(path, map_location="cpu", weights_only=False)
ckpt["model_state_dict"] = {
    k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()
}
torch.save(ckpt, path)
print(f"Fixed: {path}")
