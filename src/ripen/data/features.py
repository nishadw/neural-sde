"""Image features from a frozen pretrained backbone, cached to disk once.

The JAX models never touch pixels: they read ``(D,)`` vectors per photo from
an ``.npz`` cache keyed by filename. Extraction needs the optional
``[features]`` extra (torch, torchvision, pillow); fine-tuning the backbone
is out of scope for the CPU path and is left as a documented option.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

BACKBONE_DIMS = {"resnet18": 512, "resnet50": 2048, "convnext_tiny": 768}


def load_feature_cache(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def save_feature_cache(path: str | Path, feats: dict[str, np.ndarray]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **feats)


class HashFeaturizer:
    """Deterministic stand-in for a backbone: features are a hash of the filename. For tests and smoke runs."""

    def __init__(self, dim: int = 16):
        self.dim = dim

    def __call__(self, paths: list[str]) -> dict[str, np.ndarray]:
        out = {}
        for p in paths:
            seed = int.from_bytes(hashlib.sha1(Path(p).name.encode()).digest()[:8], "little")
            out[Path(p).name] = np.random.default_rng(seed).normal(size=self.dim).astype(np.float32)
        return out


def extract_features(
    paths: list[str], backbone: str = "resnet18", batch_size: int = 64, device: str = "cpu", image_size: int = 224
) -> dict[str, np.ndarray]:
    """Run a frozen ImageNet backbone over ``paths``; returns ``{filename: (D,) float32}``.

    Uses global-average-pooled features (the layer before the classifier).
    """
    import torch
    import torchvision
    from PIL import Image

    weights = {
        "resnet18": torchvision.models.ResNet18_Weights.IMAGENET1K_V1,
        "resnet50": torchvision.models.ResNet50_Weights.IMAGENET1K_V2,
        "convnext_tiny": torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
    }[backbone]
    model = getattr(torchvision.models, backbone)(weights=weights)
    if backbone.startswith("resnet"):
        model.fc = torch.nn.Identity()
    else:
        model.classifier[-1] = torch.nn.Identity()
    model.eval().to(device)
    tf = weights.transforms()
    out: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            chunk = paths[i : i + batch_size]
            x = torch.stack([tf(Image.open(p).convert("RGB")) for p in chunk]).to(device)
            f = model(x).float().cpu().numpy()
            for p, v in zip(chunk, f, strict=True):
                out[Path(p).name] = v.astype(np.float32)
    return out
