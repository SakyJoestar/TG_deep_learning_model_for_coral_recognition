#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_finetune.py

Fine-tuning de modelos preentrenados (VGG, ResNet, AlexNet, MobileNet)
sobre dataset ImageFolder con estructura train/val/test.

Guarda resultados en ./results_finetune/<model_name>/
"""

import os, argparse, json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------- CARGAR MODELO ----------------------

def load_pretrained_model(name, num_classes):
    """
    Carga un modelo preentrenado de torchvision y reemplaza
    la última capa para clasificar num_classes.
    """

    name = name.lower()

    if name == "resnet18":
        model = models.resnet18(weights="IMAGENET1K_V1")
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif name == "vgg16":
        model = models.vgg16(weights="IMAGENET1K_V1")
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    elif name == "alexnet":
        model = models.alexnet(weights="IMAGENET1K_V1")
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    elif name == "mobilenet_v2":
        model = models.mobilenet_v2(weights="IMAGENET1K_V1")
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    else:
        raise ValueError(f"Modelo '{name}' no soportado en este script.")

    return model


# ---------------------- DATA LOADERS ----------------------

def make_loaders(data_dir, img_size=224, batch_size=32, num_workers=2):
    """
    Imagenes se escalan a 224x224 porque los modelos preentrenados lo necesitan.
    """

    train_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    eval_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
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


# ---------------------- ENTRENAMIENTO ----------------------

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
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

    return total_loss/total, correct/total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        total_loss += float(loss) * x.size(0)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return total_loss/total, correct/total


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    yy, pp = [], []
    for x, y in loader:
        x = x.to(device)
        out = model(x)
        pred = out.argmax(1).cpu().numpy()
        yy.extend(y.numpy())
        pp.extend(pred)
    return np.array(yy), np.array(pp)


# ---------------------- MAIN ----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--model", required=True,
                    help="resnet18 | vgg16 | alexnet | mobilenet_v2")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--results_dir", type=str, default="results_finetune")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_set, val_set, test_set, train_loader, val_loader, test_loader = \
        make_loaders(args.data_dir)

    num_classes = len(train_set.classes)

    model = load_pretrained_model(args.model, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # Carpeta de resultados
    result_path = Path(args.results_dir) / args.model
    result_path.mkdir(parents=True, exist_ok=True)

    # Guardar mapa de clases
    with open(result_path / "label_map.json", "w") as f:
        json.dump({i: c for i, c in enumerate(train_set.classes)}, f, indent=2)

    history = []
    best_val_acc = -1
    best_state = None

    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = eval_epoch(model, val_loader, criterion, device)
        te_loss, te_acc = eval_epoch(model, test_loader, criterion, device)

        history.append([epoch, tr_loss, tr_acc, va_loss, va_acc, te_loss, te_acc])

        print(f"[{epoch:03d}] "
              f"train_acc={tr_acc:.3f} val_acc={va_acc:.3f} test_acc={te_acc:.3f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = model.state_dict()

    # Guardar mejor modelo
    torch.save(best_state, result_path / "best_model.pt")

    # Guardar métricas
    df = pd.DataFrame(history, columns=[
        "epoch", "train_loss", "train_acc", "val_loss", "val_acc", "test_loss", "test_acc"
    ])
    df.to_csv(result_path / "metrics.csv", index=False)

    # Evaluación final
    model.load_state_dict(best_state)
    y_true, y_pred = predict_all(model, test_loader, device)

    rep = classification_report(
        y_true, y_pred,
        target_names=train_set.classes, digits=4
    )
    cm = confusion_matrix(y_true, y_pred)

    with open(result_path / "test_report.txt", "w") as f:
        f.write(rep + "\n\n" + str(cm))

    print("\n=== Reporte final ===")
    print(rep)
    print("Confusion matrix:")
    print(cm)


if __name__ == "__main__":
    main()