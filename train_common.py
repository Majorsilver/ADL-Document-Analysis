"""Shared training loop for the cross-out classifiers.

Each per-model trainer supplies a `build_model` callable, a dataset `mode`,
and any extra wandb config keys; this module owns the rest.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

import wandb
from dataset import CrossOutDataset

PROJECT_DIR = Path(__file__).parent
WANDB_PROJECT = "cross-out-detection"
WANDB_ENTITY = "linneahejsupergroup-lule-university-of-technology"

TASKS = {
    "binary": ["CLEAN", "MIXED"],
    "type": ["CROSS", "DIAGONAL", "DOUBLE_LINE",
             "SCRATCH", "SINGLE_LINE", "WAVE", "ZIG_ZAG"],
}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_sum = 0.0
    total = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    all_preds: list[int] = []
    all_labels: list[int] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss_sum += criterion(logits, y).item()
        preds = logits.argmax(1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y.cpu().tolist())
        total += y.size(0)
    avg_loss = loss_sum / total
    correct = sum(int(p == t) for p, t in zip(all_preds, all_labels))
    acc = correct / total
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return avg_loss, acc, f1, all_preds, all_labels


def run_training(
    *,
    build_model: Callable[[int], nn.Module],
    dataset_mode: str,
    wandb_extra: dict[str, Any],
    task: str,
    batch_size: int,
    lr: float,
    max_epochs: int,
    patience: int,
    num_workers: int,
    out_dir: Path,
    run_name: str,
    augment: bool = False,
    resume: bool = False,
    use_wandb: bool = True,
):
    classes = TASKS[task]
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = run_name or task

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"task: {task} | classes ({len(classes)}): {classes}")
    print(f"run: {tag} | augment: {augment} | resume: {resume} | wandb: {use_wandb}")

    if use_wandb:
        wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=tag,
            resume="allow" if resume else None,
            config={
                "task": task, "classes": classes, "augment": augment,
                "batch_size": batch_size, "lr": lr, "max_epochs": max_epochs,
                "patience": patience,
                **wandb_extra,
            },
        )

    train_ds = CrossOutDataset("train", classes, augment=augment, mode=dataset_mode)
    val_ds = CrossOutDataset("val", classes, augment=False, mode=dataset_mode)
    print(f"train samples: {len(train_ds)} | val samples: {len(val_ds)}")

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin, persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin, persistent_workers=num_workers > 0)

    model = build_model(len(classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=pin)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5, min_lr=1e-7)

    best_ckpt_path = out_dir / f"{tag}_best.pt"
    last_ckpt_path = out_dir / f"{tag}_last.pt"
    history_path = out_dir / f"{tag}_history.json"

    best_val_loss = float("inf")
    best_val_acc = -1.0
    best_epoch = -1
    epochs_since_improve = 0
    history: list[dict[str, Any]] = []
    start_epoch = 1

    if resume:
        if not last_ckpt_path.exists():
            raise FileNotFoundError(f"--resume given but {last_ckpt_path} does not exist")
        print(f"Resuming from {last_ckpt_path}")
        ck = torch.load(last_ckpt_path, map_location=device, weights_only=False)
        if ck.get("classes") != classes:
            raise ValueError(f"Class mismatch: ckpt={ck.get('classes')} vs current={classes}")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        if "scheduler" in ck:
            scheduler.load_state_dict(ck["scheduler"])
        start_epoch = ck["epoch"] + 1
        best_val_loss = ck["best_val_loss"]
        best_val_acc = ck["best_val_acc"]
        best_epoch = ck["best_epoch"]
        epochs_since_improve = ck["epochs_since_improve"]
        if history_path.exists():
            with open(history_path) as f:
                history = json.load(f)
        print(f"Resumed at epoch {start_epoch} | best loss {best_val_loss:.4f}@{best_epoch} "
              f"| stale {epochs_since_improve}/{patience}")

    for epoch in range(start_epoch, max_epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for step, (x, y) in enumerate(train_loader, 1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=pin):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * y.size(0)
            running_correct += (logits.argmax(1) == y).sum().item()
            running_total += y.size(0)

            if step % 50 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss {running_loss/running_total:.4f} "
                      f"acc {running_correct/running_total:.4f}", flush=True)

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, device)
        dt = time.time() - t0

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_since_improve = 0
            torch.save({"model": model.state_dict(), "classes": classes,
                        "epoch": epoch, "val_loss": val_loss, "val_acc": val_acc},
                       best_ckpt_path)
        else:
            epochs_since_improve += 1

        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1,
            "seconds": dt, "improved": improved,
            "best_val_loss": best_val_loss, "best_val_acc": best_val_acc,
            "lr": current_lr,
        })
        print(f"epoch {epoch:3d} | train loss {train_loss:.4f} acc {train_acc:.4f} "
              f"| val loss {val_loss:.4f} acc {val_acc:.4f} f1 {val_f1:.4f} "
              f"| best loss {best_val_loss:.4f}@{best_epoch} "
              f"| stale {epochs_since_improve}/{patience} "
              f"| lr {current_lr:.1e} | {dt:.1f}s", flush=True)

        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1,
                "lr": current_lr,
                "best_val_loss": best_val_loss, "best_val_acc": best_val_acc,
                "epoch_seconds": dt,
            })

        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "scheduler": scheduler.state_dict(),
            "classes": classes,
            "epoch": epoch,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "epochs_since_improve": epochs_since_improve,
        }, last_ckpt_path)

        if epochs_since_improve >= patience:
            print(f"Early stopping: no val-loss improvement in {patience} epochs.")
            break

    print(f"Best val loss: {best_val_loss:.4f} (acc {best_val_acc:.4f}) "
          f"at epoch {best_epoch}. Checkpoint: {best_ckpt_path}")

    if best_ckpt_path.exists():
        print(f"Loading best checkpoint for test eval: {best_ckpt_path}")
        ck = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
    test_ds = CrossOutDataset("test", classes, augment=False, mode=dataset_mode)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin,
                             persistent_workers=num_workers > 0)
    test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(
        model, test_loader, device)
    print(f"test loss {test_loss:.4f} acc {test_acc:.4f} f1 {test_f1:.4f} "
          f"(n={len(test_ds)})")

    if use_wandb:
        wandb.log({
            "test_loss": test_loss, "test_acc": test_acc, "test_f1": test_f1,
            "conf_mat": wandb.plot.confusion_matrix(
                y_true=test_labels, preds=test_preds, class_names=classes),
        })
        wandb.finish()


def build_arg_parser(*, default_batch_size: int, default_lr: float,
                     default_patience: int) -> argparse.ArgumentParser:
    """Argparse with the args every trainer shares; per-trainer defaults differ."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS), required=True)
    ap.add_argument("--batch-size", type=int, default=default_batch_size)
    ap.add_argument("--lr", type=float, default=default_lr)
    ap.add_argument("--max-epochs", type=int, default=1000)
    ap.add_argument("--patience", type=int, default=default_patience)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "runs")
    ap.add_argument("--augment", action="store_true",
                    help="Enable train-time augmentation")
    ap.add_argument("--run-name", type=str, required=True,
                    help="Output filename stem and wandb run name")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from {run-name}_last.pt")
    ap.add_argument("--no-wandb", action="store_true",
                    help="Disable wandb logging")
    return ap
