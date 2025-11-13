#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_cnn_patches.py

Entrena una CNN sencilla sobre parches NxN organizados en estructura ImageFolder:
data_dir/
    train/clase_i/*.png
    val/clase_i/*.png
    test/clase_i/*.png

- Usa data augmentation en train.
- Pondera las clases con class_weights.
- Aplica early stopping por val_acc.
- Guarda en ./results_cnn:
    - best_cnn.pt
    - metrics_cnn.csv
    - test_report_cnn.txt
    - label_map_cnn.json
    - curves_accuracy_cnn.png
    - curves_loss_cnn.png
    - confusion_matrix_cnn.png
"""

import os, json, argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


# ----------------------- MODELO CNN -----------------------

class ConvNet(nn.Module):
    """
    CNN pensada para parches 32x32 (3 canales).
    Si usas otro patch_size habría que ajustar el tamaño
    de la capa fully-connected (128 * 4 * 4 asume 3 pools).
    """
    def __init__(self, num_classes, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            # 3 x 32 x 32 -> 32 x 32 x 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 32x32 -> 16x16

            # 32 x 16 x 16 -> 64 x 16 x 16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 16x16 -> 8x8

            # 64 x 8 x 8 -> 128 x 8 x 8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 8x8 -> 4x4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),              # 128 * 4 * 4 = 2048
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ----------------------- DATA LOADERS -----------------------

def make_loaders(data_dir, patch_size=32, batch_size=64, num_workers=2):
    """
    Crea loaders para train/val/test con augmentación ligera en train.
    """
    train_tfm = transforms.Compose([
        transforms.Resize((patch_size, patch_size)),

        # === Gaussian Blur como augmentation ===
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        #  - kernel_size=3 funciona bien para parches pequeños
        #  - sigma al azar entre 0.1 y 1.0 (leve desenfoque)

        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),

        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                            std=[0.5, 0.5, 0.5]),
    ])

    eval_tfm = transforms.Compose([
        transforms.Resize((patch_size, patch_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    train_set = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tfm)
    val_set   = datasets.ImageFolder(os.path.join(data_dir, "val"),   transform=eval_tfm)
    test_set  = datasets.ImageFolder(os.path.join(data_dir, "test"),  transform=eval_tfm)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_set, val_set, test_set, train_loader, val_loader, test_loader


def compute_class_weights(dataset):
    ys = [y for _, y in dataset]
    counts = np.bincount(ys)
    counts = counts + 1e-6  # evitar división por cero
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32)


# ----------------------- LOOP DE ENTRENAMIENTO -----------------------

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * x.size(0)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()

    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)

        total_loss += float(loss) * x.size(0)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()

    return total_loss / total, correct / total


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    yy, pp = [], []
    for x, y in loader:
        x = x.to(device)
        out = model(x)
        pred = out.argmax(1).cpu().numpy().tolist()
        pp += pred
        yy += y.numpy().tolist()
    return np.array(yy), np.array(pp)


# ----------------------- MAIN -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True,
                    help="Carpeta raíz del dataset (ej: ./datasets/32x32)")
    ap.add_argument("--patch_size", type=int, default=32,
                    help="Tamaño de los parches NxN (ej: 32 para 32x32)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--results_dir", type=str, default="results_cnn",
                    help="Carpeta donde se guardan los resultados")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Guardando resultados en: {results_dir.resolve()}")

    # Loaders
    train_set, val_set, test_set, train_loader, val_loader, test_loader = make_loaders(
        args.data_dir,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    num_classes = len(train_set.classes)
    print(f"Patch size: {args.patch_size}x{args.patch_size}")
    print(f"num_classes = {num_classes}")
    print("Clases:", train_set.classes)

    # Modelo + pérdida + optimizador
    model = ConvNet(num_classes=num_classes, dropout=args.dropout).to(device)
    weights = compute_class_weights(train_set).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # Map de etiquetas
    with open(results_dir / "label_map_cnn.json", "w", encoding="utf-8") as f:
        json.dump({i: c for i, c in enumerate(train_set.classes)},
                  f, ensure_ascii=False, indent=2)

    best_val = -1.0
    best_state = None
    patience = args.patience
    history = []

    # Loop de entrenamiento
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = eval_epoch(model, val_loader, criterion, device)
        te_loss, te_acc = eval_epoch(model, test_loader, criterion, device)

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss":   va_loss, "val_acc":   va_acc,
            "test_loss":  te_loss, "test_acc":  te_acc,
        })

        print(
            f"[{epoch:03d}] "
            f"train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
            f"val_loss={va_loss:.4f} acc={va_acc:.4f} | "
            f"test_loss={te_loss:.4f} acc={te_acc:.4f}"
        )

        # Early stopping por val_acc
        if va_acc > best_val:
            best_val = va_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience = args.patience
        else:
            patience -= 1
            if patience <= 0:
                print("Early stopping activado.")
                break

    # Guardar historial + mejor modelo
    pd.DataFrame(history).to_csv(results_dir / "metrics_cnn.csv", index=False)
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(model.state_dict(), results_dir / "best_cnn.pt")
        print(f"Guardado {results_dir / 'best_cnn.pt'} (val_acc={best_val:.4f})")

    # Evaluación final en test
    y_true, y_pred = predict_all(model, test_loader, device)
    all_labels = list(range(len(train_set.classes)))
    rep = classification_report(
        y_true,
        y_pred,
        labels=all_labels,
        target_names=train_set.classes,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)

    with open(results_dir / "test_report_cnn.txt", "w", encoding="utf-8") as f:
        f.write(rep + "\n\nConfusion Matrix:\n" + np.array2string(cm))

    print("\n=== Test Report (CNN) ===")
    print(rep)
    print("Confusion Matrix:\n", cm)

    # --- GRÁFICAS: accuracy y loss por época (train/val/test) ---
    try:
        epochs = [h["epoch"] for h in history]
        tr_acc = [h["train_acc"] for h in history]
        va_acc = [h["val_acc"]   for h in history]
        te_acc = [h["test_acc"]  for h in history]

        tr_loss = [h["train_loss"] for h in history]
        va_loss = [h["val_loss"]   for h in history]
        te_loss = [h["test_loss"]  for h in history]

        # Accuracy
        plt.figure()
        plt.plot(epochs, tr_acc, label="train_acc")
        plt.plot(epochs, va_acc, label="val_acc")
        plt.plot(epochs, te_acc, label="test_acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy por época (CNN)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(results_dir / "curves_accuracy_cnn.png", dpi=150)

        # Loss
        plt.figure()
        plt.plot(epochs, tr_loss, label="train_loss")
        plt.plot(epochs, va_loss, label="val_loss")
        plt.plot(epochs, te_loss, label="test_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss por época (CNN)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(results_dir / "curves_loss_cnn.png", dpi=150)

        # Matriz de confusión como heatmap
        plt.figure(figsize=(6, 5))
        im = plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.title("Matriz de confusión (CNN)")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        tick_marks = np.arange(len(train_set.classes))
        plt.xticks(tick_marks, train_set.classes, rotation=45, ha="right")
        plt.yticks(tick_marks, train_set.classes)
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        plt.savefig(results_dir / "confusion_matrix_cnn.png", dpi=150)

        print("Guardadas en results_cnn/: curves_accuracy_cnn.png, curves_loss_cnn.png y confusion_matrix_cnn.png")
    except Exception as e:
        print(f"[AVISO] No se pudieron generar las gráficas: {e}")


if __name__ == "__main__":
    main()