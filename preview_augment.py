"""Show FitPadInvert + train-time augmentations on a sample image."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from dataset import (DATASET_ROOT, FitPadInvert, IMAGENET_MEAN, IMAGENET_STD,
                     build_transform)

OUT_PATH = Path(__file__).parent / "augment_preview.png"
NUM_SAMPLES = 8


def denormalize(t: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    x = (t * std + mean).clamp(0, 1)
    return x.permute(1, 2, 0).numpy()


def main():
    sample = next((DATASET_ROOT / "train" / "images" / "MIXED").iterdir())
    print(f"Using sample: {sample}")

    original = Image.open(sample).convert("L")
    baseline = FitPadInvert()(Image.open(sample))
    aug_transform = build_transform(augment=True)

    torch.manual_seed(0)
    augmented = [aug_transform(Image.open(sample)) for _ in range(NUM_SAMPLES)]

    cols = 5
    rows = (NUM_SAMPLES + 2 + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 2 * rows))
    axes = axes.flatten()

    axes[0].imshow(original, cmap="gray")
    axes[0].set_title(f"original\n{original.size[0]}x{original.size[1]}")
    axes[0].axis("off")

    axes[1].imshow(baseline, cmap="gray")
    axes[1].set_title("FitPadInvert\n136x68")
    axes[1].axis("off")

    for i, t in enumerate(augmented):
        ax = axes[2 + i]
        ax.imshow(denormalize(t))
        ax.set_title(f"aug #{i + 1}")
        ax.axis("off")

    for ax in axes[2 + NUM_SAMPLES:]:
        ax.axis("off")

    fig.suptitle("Augmentation preview  (affine + brightness/contrast + erasing)")
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=120)
    print(f"Wrote {OUT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
