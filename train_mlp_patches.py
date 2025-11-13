#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_mlp_patches.py
Entrena un MLP (perceptrón multicapa) sobre parches 40x40 organizados en ImageFolder.
Genera curvas de accuracy y loss por época (train/val/test) y guarda PNGs.
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

def parse_hidden(s):
    if not s: return [1024,512]
    return [int(x) for x in s.split(",") if x.strip()]

class MLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden, dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, num_classes)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

def make_loaders(data_dir, batch_size=64, num_workers=2):
    tfm = transforms.Compose([
        transforms.Resize((40,40)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ])
    train_set = datasets.ImageFolder(os.path.join(data_dir,"train"), transform=tfm)
    val_set   = datasets.ImageFolder(os.path.join(data_dir,"val"),   transform=tfm)
    test_set  = datasets.ImageFolder(os.path.join(data_dir,"test"),  transform=tfm)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_set, val_set, test_set, train_loader, val_loader, test_loader

def compute_class_weights(dataset):
    ys = [y for _,y in dataset]
    counts = np.bincount(ys)
    counts = counts + 1e-6
    weights = counts.sum() / (len(counts)*counts)
    return torch.tensor(weights, dtype=torch.float32)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train(); total_loss=0.0; correct=0; total=0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        x = x.view(x.size(0), -1)
        out = model(x)
        loss = criterion(out, y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += float(loss)*x.size(0)
        pred = out.argmax(1)
        correct += (pred==y).sum().item()
        total += y.numel()
    return total_loss/total, correct/total

@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval(); total_loss=0.0; correct=0; total=0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        x = x.view(x.size(0), -1)
        out = model(x)
        loss = criterion(out, y)
        total_loss += float(loss)*x.size(0)
        pred = out.argmax(1)
        correct += (pred==y).sum().item()
        total += y.numel()
    return total_loss/total, correct/total

@torch.no_grad()
def predict_all(model, loader, device):
    model.eval(); yy=[]; pp=[]
    for x,y in loader:
        x = x.to(device)
        x = x.view(x.size(0), -1)
        out = model(x)
        pred = out.argmax(1).cpu().numpy().tolist()
        pp += pred; yy += y.numpy().tolist()
    return np.array(yy), np.array(pp)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=str, default="1024,512")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--patience", type=int, default=7)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_set, val_set, test_set, train_loader, val_loader, test_loader = make_loaders(
        args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers
    )
    input_dim = 3*40*40
    num_classes = len(train_set.classes)
    hidden = parse_hidden(args.hidden)

    model = MLP(input_dim, num_classes, hidden, dropout=args.dropout).to(device)
    weights = compute_class_weights(train_set).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # label map
    with open("label_map.json","w",encoding="utf-8") as f:
        json.dump({i:c for i,c in enumerate(train_set.classes)}, f, ensure_ascii=False, indent=2)

    best_val = -1.0; best_state=None; patience=args.patience
    history = []

    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = eval_epoch(model, val_loader, criterion, device)
        # (OPCIONAL y solicitado) medir test en cada época
        te_loss, te_acc = eval_epoch(model, test_loader, criterion, device)

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": va_loss,   "val_acc": va_acc,
            "test_loss": te_loss,  "test_acc": te_acc,
        })
        print(f"[{epoch:03d}] "
              f"train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val_loss={va_loss:.4f} acc={va_acc:.4f} | "
              f"test_loss={te_loss:.4f} acc={te_acc:.4f}")

        # early stopping por val_acc
        if va_acc > best_val:
            best_val = va_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience = args.patience
        else:
            patience -= 1
            if patience <= 0:
                print("Early stopping activado.")
                break

    # guardar historial y mejor modelo
    pd.DataFrame(history).to_csv("metrics.csv", index=False)
    if best_state is not None:
        model.load_state_dict(best_state); torch.save(model.state_dict(), "best_mlp.pt")
        print(f"Guardado best_mlp.pt (val_acc={best_val:.4f})")

    # --- evaluación final en test (con todas las clases del train) ---
    y_true, y_pred = predict_all(model, test_loader, device)
    all_labels = list(range(len(train_set.classes)))
    rep = classification_report(
        y_true, y_pred,
        labels=all_labels,
        target_names=train_set.classes,
        digits=4,
        zero_division=0,
    )
    cm  = confusion_matrix(y_true, y_pred, labels=all_labels)
    with open("test_report.txt","w",encoding="utf-8") as f:
        f.write(rep + "\n\nConfusion Matrix:\n" + np.array2string(cm))
    print("\n=== Test Report ===")
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
        plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy por época")
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig("curves_accuracy.png", dpi=150)

        # Loss
        plt.figure()
        plt.plot(epochs, tr_loss, label="train_loss")
        plt.plot(epochs, va_loss, label="val_loss")
        plt.plot(epochs, te_loss, label="test_loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss por época")
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig("curves_loss.png", dpi=150)

        print("Guardadas: curves_accuracy.png y curves_loss.png")
    except Exception as e:
        print(f"[AVISO] No se pudieron generar las gráficas: {e}")

if __name__ == "__main__":
    main()