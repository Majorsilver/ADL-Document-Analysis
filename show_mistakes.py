"""Render a grid of validation samples the binary model got wrong."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from dataset import CrossOutDataset, FitPadInvert
from train import build_model

PROJECT_DIR = Path(__file__).parent


def load_model(ckpt_path: Path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    classes = ckpt["classes"]
    model = build_model(len(classes)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {ckpt_path.name} | classes={classes} | "
          f"best_val_acc={ckpt.get('val_acc', 'n/a')} @ epoch {ckpt.get('epoch', 'n/a')}")
    return model, classes


@torch.no_grad()
def find_mistakes(model, ds, device, batch_size: int, limit: int,
                  seed: int = 0):
    """Stream val in random order, stop once we have enough wrong predictions.

    Keeps track of the original dataset index so we can recover the file path.
    """
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(ds), generator=g).tolist()
    subset = Subset(ds, order)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=2)

    mistakes = []
    pos = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(1)
        wrong = (preds != y).nonzero(as_tuple=True)[0]
        for idx in wrong.tolist():
            true = int(y[idx])
            pred = int(preds[idx])
            mistakes.append({
                "sample_idx": order[pos + idx],
                "true": true,
                "pred": pred,
                "p_true": float(probs[idx, true]),
                "p_pred": float(probs[idx, pred]),
            })
            if len(mistakes) >= limit:
                return mistakes
        pos += y.size(0)
    return mistakes


def plot(mistakes, ds, classes, out_path: Path):
    n = len(mistakes)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 2.4 * rows))
    axes = axes.flatten() if n > 1 else [axes]

    fit = FitPadInvert()
    from PIL import Image
    for i, m in enumerate(mistakes):
        path, _ = ds.samples[m["sample_idx"]]
        img = fit(Image.open(path))
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(
            f"true={classes[m['true']]}\npred={classes[m['pred']]}  "
            f"p={m['p_pred']:.2f}",
            fontsize=9,
        )
        axes[i].axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"{n} misclassified val samples")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"Wrote {out_path}")
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path,
                    default=PROJECT_DIR / "runs" / "binary_best.pt")
    ap.add_argument("--task", choices=["binary", "type"], default="binary")
    ap.add_argument("--num", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--out", type=Path,
                    default=PROJECT_DIR / "mistakes.png")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes = load_model(args.ckpt, device)

    ds = CrossOutDataset("val", classes, augment=False)
    print(f"val samples: {len(ds)}")

    mistakes = find_mistakes(model, ds, device, args.batch_size, args.num)
    print(f"collected {len(mistakes)} mistakes")
    if not mistakes:
        print("No mistakes found — unusual.")
        return
    plot(mistakes, ds, classes, args.out)


if __name__ == "__main__":
    main()
