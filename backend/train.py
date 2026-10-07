"""
Training Script for Hybrid CNN-RNN Chickpea Disease Detection Model
───────────────────────────────────────────────────────────────────
Trains the model on the dataset (real or synthetic), saves the best
checkpoint, and exports training metrics and plots.

Usage:
    python train.py --data_dir ../dataset --epochs 30 --batch_size 16
"""

import os
import sys
import json
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

# Local imports
sys.path.insert(0, os.path.dirname(__file__))
from model import get_model, DISEASE_CLASSES
from preprocessing import get_training_transform, get_inference_transform, IMAGE_SIZE


# ─── CLI arguments ────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train Hybrid CNN-RNN Chickpea Disease Detector")
    p.add_argument("--data_dir",   default="../dataset",  help="Root dataset directory")
    p.add_argument("--model_dir",  default="../models",   help="Where to save model checkpoints")
    p.add_argument("--epochs",     type=int, default=30,  help="Number of training epochs")
    p.add_argument("--batch_size", type=int, default=16,  help="Batch size")
    p.add_argument("--lr",         type=float, default=1e-3, help="Initial learning rate")
    p.add_argument("--patience",   type=int, default=7,   help="Early-stopping patience")
    p.add_argument("--num_classes",type=int, default=4,   help="Number of disease classes")
    return p.parse_args()


# ─── Dataset loader ───────────────────────────────────────────────────────────

def get_dataloaders(data_dir, batch_size):
    train_ds = datasets.ImageFolder(
        root=os.path.join(data_dir, "train"),
        transform=get_training_transform()
    )
    val_ds = datasets.ImageFolder(
        root=os.path.join(data_dir, "val"),
        transform=get_inference_transform()
    )
    test_ds = datasets.ImageFolder(
        root=os.path.join(data_dir, "test"),
        transform=get_inference_transform()
    )

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=False),
        "val":   DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0),
        "test":  DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0),
    }
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"  Classes: {train_ds.classes}")
    return loaders, train_ds.classes


# ─── One epoch ────────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss    = criterion(outputs, labels)

            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds      = outputs.argmax(dim=1)
            correct   += (preds == labels).sum().item()
            total     += images.size(0)

    return total_loss / total, correct / total


# ─── Evaluation & metrics ─────────────────────────────────────────────────────

def evaluate(model, loader, device, class_names):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            preds  = model(images).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    report = classification_report(all_labels, all_preds,
                                   target_names=class_names, output_dict=True)
    cm     = confusion_matrix(all_labels, all_preds)
    return report, cm, all_preds, all_labels


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_training_curves(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss", linewidth=2)
    ax1.plot(epochs, history["val_loss"],   "r-o", label="Val Loss",   linewidth=2)
    ax1.set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], "b-o", label="Train Acc", linewidth=2)
    ax2.plot(epochs, history["val_acc"],   "r-o", label="Val Acc",   linewidth=2)
    ax2.set_title("Training & Validation Accuracy", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_ylim([0, 1]); ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved training curves → {save_path}")


def plot_confusion_matrix(cm, class_names, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved confusion matrix → {save_path}")


# ─── Main training loop ───────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Chickpea Disease Detection — Hybrid CNN-RNN Trainer")
    print(f"{'='*60}")
    print(f"  Device : {device}")
    print(f"  Epochs : {args.epochs}  |  Batch: {args.batch_size}  |  LR: {args.lr}")
    print(f"  Data   : {args.data_dir}")
    print()

    os.makedirs(args.model_dir, exist_ok=True)

    # Data
    loaders, class_names = get_dataloaders(args.data_dir, args.batch_size)

    # Model
    model     = get_model(num_classes=args.num_classes, image_size=IMAGE_SIZE).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                      factor=0.5, patience=3)

    history   = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    patience_cnt = 0
    best_model_path = os.path.join(args.model_dir, "best_model.pth")

    print(f"\n{'─'*60}")
    print(f"  {'Epoch':>6}  {'TrLoss':>8}  {'TrAcc':>7}  {'VaLoss':>8}  {'VaAcc':>7}  {'LR':>9}")
    print(f"{'─'*60}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc = run_epoch(model, loaders["train"], criterion,
                                    optimizer, device, train=True)
        va_loss, va_acc = run_epoch(model, loaders["val"],   criterion,
                                    optimizer, device, train=False)

        scheduler.step(va_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        flag = ""
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "val_acc":     va_acc,
                "class_names": class_names,
                "num_classes": args.num_classes,
                "image_size":  IMAGE_SIZE,
            }, best_model_path)
            patience_cnt = 0
            flag = " ★"
        else:
            patience_cnt += 1

        elapsed = time.time() - t0
        print(f"  {epoch:>6}  {tr_loss:>8.4f}  {tr_acc:>6.2%}  "
              f"{va_loss:>8.4f}  {va_acc:>6.2%}  {current_lr:>9.2e}"
              f"  ({elapsed:.1f}s){flag}")

        if patience_cnt >= args.patience:
            print(f"\n  Early stopping at epoch {epoch} (patience={args.patience})")
            break

    print(f"{'─'*60}")
    print(f"  Best Val Accuracy: {best_val_acc:.2%}")

    # Final evaluation on test set
    print("\n  Loading best model for test evaluation…")
    ckpt = torch.load(best_model_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    report, cm, preds, labels = evaluate(model, loaders["test"], device, class_names)

    print(f"\n  Test Results:")
    print(f"  {'─'*40}")
    print(f"  Accuracy : {report['accuracy']:.4f}")
    print(f"  Precision: {report['weighted avg']['precision']:.4f}")
    print(f"  Recall   : {report['weighted avg']['recall']:.4f}")
    print(f"  F1-Score : {report['weighted avg']['f1-score']:.4f}")

    # Save plots
    plot_dir = os.path.join(args.model_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plot_training_curves(history, os.path.join(plot_dir, "training_curves.png"))
    plot_confusion_matrix(cm, class_names, os.path.join(plot_dir, "confusion_matrix.png"))

    # Save metrics JSON
    metrics = {
        "best_val_accuracy": best_val_acc,
        "test_accuracy":     report["accuracy"],
        "test_precision":    report["weighted avg"]["precision"],
        "test_recall":       report["weighted avg"]["recall"],
        "test_f1":           report["weighted avg"]["f1-score"],
        "class_names":       class_names,
        "per_class":         {c: report[c] for c in class_names if c in report},
        "history":           history,
    }
    with open(os.path.join(args.model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  Model checkpoint : {best_model_path}")
    print(f"  Metrics JSON     : {os.path.join(args.model_dir, 'metrics.json')}")
    print(f"  Plots directory  : {plot_dir}")
    print(f"\n✓ Training complete!\n")

    return metrics


if __name__ == "__main__":
    args = parse_args()
    train(args)
