"""Diagnose the binary model: confusion matrix, per-class accuracy, confidence."""
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import CrossOutDataset
from train import build_model

PROJECT_DIR = Path(__file__).parent


@torch.no_grad()
def run(ckpt_path: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    classes = ckpt["classes"]
    model = build_model(len(classes)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"ckpt: {ckpt_path.name} | classes={classes}")
    print(f"reported best val acc: {ckpt.get('val_acc'):.6f}")

    ds = CrossOutDataset("val", classes, augment=False)
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=2)

    n = len(classes)
    conf = torch.zeros(n, n, dtype=torch.long)
    sum_prob_wrong = [0.0] * n
    count_wrong = [0] * n
    count_per_class = [0] * n

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu()
        preds = logits.argmax(1).cpu()
        y = y.cpu()
        for t, p, prob_row in zip(y.tolist(), preds.tolist(), probs):
            conf[t, p] += 1
            count_per_class[t] += 1
            if t != p:
                sum_prob_wrong[t] += float(prob_row[p])
                count_wrong[t] += 1

    print("\nConfusion matrix (rows=true, cols=pred):")
    header = "            " + "  ".join(f"{c:>10}" for c in classes)
    print(header)
    for i, c in enumerate(classes):
        row = "  ".join(f"{int(conf[i, j]):>10d}" for j in range(n))
        print(f"{c:>10}  {row}")

    print("\nPer-class recall (correct / total for that true class):")
    for i, c in enumerate(classes):
        correct = int(conf[i, i])
        total = count_per_class[i]
        acc = correct / total if total else 0.0
        print(f"  {c:>10}: {correct:>6d}/{total:<6d}  = {acc:.4f}")

    print("\nPer-class precision (correct / total predicted as that class):")
    for j, c in enumerate(classes):
        correct = int(conf[j, j])
        pred_total = int(conf[:, j].sum())
        prec = correct / pred_total if pred_total else 0.0
        print(f"  {c:>10}: {correct:>6d}/{pred_total:<6d}  = {prec:.4f}")

    print("\nAverage model confidence when wrong (prob assigned to the wrong pred):")
    for i, c in enumerate(classes):
        if count_wrong[i]:
            avg = sum_prob_wrong[i] / count_wrong[i]
            print(f"  true={c:>10}: {count_wrong[i]:>5d} mistakes, avg p(wrong)={avg:.3f}")
        else:
            print(f"  true={c:>10}: 0 mistakes")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path,
                    default=PROJECT_DIR / "runs" / "binary_best.pt")
    args = ap.parse_args()
    run(args.ckpt)
