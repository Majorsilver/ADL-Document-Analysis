"""Evaluate any trained checkpoint from runs/ on the test split.

Usage:
  python test_model.py --ckpt runs/cnn5_binary_best.pt --model cnn5
  python test_model.py --ckpt runs/ConvNeXt_type_aug_best.pt --model convnext --no-prune
  python test_model.py --ckpt runs/cnn5_binary_best.pt --model cnn5 --dataset ours
  python test_model.py --ckpt Yolo11_Binary_...pth --model yolo --task binary

For checkpoints saved by train_common (dict with 'model'+'classes' keys) the
class list is read from the file. For raw state_dict checkpoints (e.g. the
YOLOClassifier weights trained externally), pass --task to specify which
class list to use. Use --no-prune to keep the mislabeled CLEAN-twin samples
that CrossOutDataset normally drops.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

import dataset as dataset_mod
from dataset import CrossOutDataset
from train_common import TASKS

PROJECT_DIR = Path(__file__).parent

MODEL_CHOICES = ("cnn3", "cnn3_k7", "cnn5", "cnn5_k7", "cnn_simple", "cnn_deep", "convnext", "yolo")
DATASET_MODE = {
    "cnn3": "cnn5", "cnn3_k7": "cnn5",
    "cnn5": "cnn5", "cnn5_k7": "cnn5",
    "cnn_simple": "cnn5", "cnn_deep": "cnn5",
    "convnext": "convnext",
    "yolo": "yolo128",
}
DATASET_ROOTS = {
    "v2": PROJECT_DIR / "cross_out_dataset_v2",
    "ours": PROJECT_DIR / "our_dataset",
}


def build_model(name: str, num_classes: int) -> nn.Module:
    if name == "cnn3":
        from cnn3 import Cnn3
        return Cnn3(num_classes=num_classes)
    if name == "cnn3_k7":
        from cnn3_first_kernel_7x7 import Cnn3
        return Cnn3(num_classes=num_classes)
    if name == "cnn5":
        from cnn5 import Cnn5
        return Cnn5(num_classes=num_classes)
    if name == "cnn5_k7":
        from cnn5_first_kernel_7x7 import Cnn5
        return Cnn5(num_classes=num_classes)
    if name == "cnn_simple":
        from cnn_simple import CnnSimple
        return CnnSimple(num_classes=num_classes)
    if name == "cnn_deep":
        from cnn_deep import CnnDeep
        return CnnDeep(num_classes=num_classes)
    if name == "convnext":
        from convnext import build_model as convnext_build
        return convnext_build(num_classes)
    if name == "yolo":
        from yolo_classifier import YOLOClassifier
        # num_classes here is the head's literal output width.
        return YOLOClassifier(num_outputs=num_classes)
    raise ValueError(f"Unknown model: {name}")


def _is_raw_state_dict(ck) -> bool:
    """True if `ck` looks like a bare state_dict (parameter names as keys)
    rather than the train_common save format (dict with 'model'/'classes')."""
    if not isinstance(ck, dict):
        return True
    return "model" not in ck or "classes" not in ck


def _detect_yolo_head_outputs(state_dict) -> int:
    w = state_dict.get("head.4.weight")
    if w is None:
        raise KeyError("Expected 'head.4.weight' in YOLOClassifier state_dict")
    return int(w.shape[0])


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        preds = model(x).argmax(1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y.cpu().tolist())
    correct = sum(int(p == t) for p, t in zip(all_preds, all_labels))
    acc = correct / len(all_labels)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return acc, f1, all_preds, all_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True,
                    help="Path to a checkpoint (e.g. runs/cnn5_binary_best.pt)")
    ap.add_argument("--model", choices=MODEL_CHOICES, required=True,
                    help="Architecture to instantiate before loading weights")
    ap.add_argument("--split", choices=("train", "val", "test"), default="test")
    ap.add_argument("--dataset", choices=tuple(DATASET_ROOTS), default="v2",
                    help="Which dataset root to evaluate on (default: v2). "
                         "'ours' only has a test split.")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--no-prune", action="store_true",
                    help="Disable dropping non-CLEAN samples whose pixels equal "
                         "their CLEAN twin (pruning is on by default)")
    ap.add_argument("--task", choices=list(TASKS), default=None,
                    help="Required when --ckpt is a raw state_dict with no "
                         "embedded class list (e.g. YOLOClassifier weights).")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    if _is_raw_state_dict(ck):
        if args.task is None:
            raise ValueError(
                f"{args.ckpt} is a raw state_dict; pass --task to specify classes "
                f"(one of: {list(TASKS)})")
        classes = TASKS[args.task]
        state_dict = ck
        print(f"checkpoint: {args.ckpt} (raw state_dict)")
        print(f"task: {args.task} | classes ({len(classes)}): {classes}")
    else:
        classes = ck["classes"]
        state_dict = ck["model"]
        print(f"checkpoint: {args.ckpt} | classes ({len(classes)}): {classes}")

    # The CLEAN-twin prune is specific to v2's synthetic-generation failures;
    # it has no meaning for real-world datasets and the hash scan trips on
    # any non-loadable PNG, so force it off for non-v2 roots.
    prune = not args.no_prune and args.dataset == "v2"
    if not prune:
        dataset_mod._load_or_build_duplicate_set = lambda _split_root: set()
        reason = "forced off for non-v2 dataset" if args.dataset != "v2" else "keeping CLEAN-twin duplicates"
        print(f"pruning: OFF ({reason})")
    else:
        print("pruning: ON")

    root = DATASET_ROOTS[args.dataset]
    print(f"dataset: {args.dataset} ({root})")
    ds = CrossOutDataset(args.split, classes, root=root, augment=False,
                         mode=DATASET_MODE[args.model])
    print(f"{args.split} samples: {len(ds)}")

    pin = device.type == "cuda"
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=pin)

    # For the YOLO model, head width may differ from len(classes) (e.g. binary
    # BCE-style training uses a single-output head); read it from the weights.
    if args.model == "yolo":
        num_outputs = _detect_yolo_head_outputs(state_dict)
        model = build_model(args.model, num_outputs).to(device)
        print(f"yolo head outputs: {num_outputs} "
              f"({'BCE binary -> 2 logits' if num_outputs == 1 else 'multi-class'})")
    else:
        model = build_model(args.model, len(classes)).to(device)
    model.load_state_dict(state_dict)

    acc, f1, _, _ = evaluate(model, loader, device)
    print(f"{args.split} accuracy: {acc:.4f}")
    print(f"{args.split} f1 (weighted): {f1:.4f}")
    return acc, f1


if __name__ == "__main__":
    main()
