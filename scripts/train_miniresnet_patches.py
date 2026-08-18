#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_miniresnet_patches.py.

Entrena una Mini-ResNet casera (:class:`MiniResNet`, construida a partir
de :class:`BasicBlock`) sobre parches NxN organizados en estructura
``ImageFolder``::

    data_dir/
        train/clase_i/*.png
        val/clase_i/*.png
        test/clase_i/*.png

Características:
    - CNN tipo Mini-ResNet (bloques residuales).
    - Usa data augmentation en train (rotación, flips, jitter, blur...).
    - Pondera las clases con ``class_weights``.
    - Métrica principal: F1 macro (no accuracy), más robusta ante el
      desbalance de clases del dataset de cobertura coralina.
    - Early stopping por F1 de validación.
    - Scheduler ``ReduceLROnPlateau`` por F1 de validación.
    - Guarda en ``./results_miniresnet`` (o ``--results_dir``):
        - ``best_miniresnet.pt``: pesos del mejor modelo según F1 de validación.
        - ``metrics_miniresnet.csv``: histórico de pérdida/accuracy/F1 por época.
        - ``test_report_miniresnet.txt``: reporte de clasificación + matriz de confusión.
        - ``label_map_miniresnet.json``: mapa índice -> nombre de clase.
        - ``curves_f1_miniresnet.png`` / ``curves_loss_miniresnet.png``: curvas de entrenamiento.
        - ``confusion_matrix_miniresnet.png``: matriz de confusión como heatmap.

Uso típico:
    python scripts/train_miniresnet_patches.py --data_dir ./datasets/32x32 --epochs 40
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
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt


# ======================= MODELO MINI-RESNET =======================

class BasicBlock(nn.Module):
    """
    Bloque residual básico:
      conv3x3 -> BN -> ReLU -> conv3x3 -> BN + skip connection.
    Si cambian canales o stride, usa un atajo 1x1.
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        """Inicializa las dos convoluciones del bloque y el atajo (downsample) si hace falta.

        Args:
            in_channels: Número de canales de entrada.
            out_channels: Número de canales de salida.
            stride: Stride de la primera convolución; ``> 1`` reduce la
                resolución espacial. Si ``stride != 1`` o
                ``in_channels != out_channels``, se añade una proyección
                1x1 en el atajo para igualar formas.
        """
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Atajo (identity o proyección 1x1 si cambian canales/stride)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x):
        """Aplica el bloque residual: dos convoluciones más la conexión de salto.

        Args:
            x (torch.Tensor): Tensor de entrada con forma
                ``(N, in_channels, H, W)``.

        Returns:
            torch.Tensor: Tensor de salida con forma
            ``(N, out_channels, H', W')`` (``H'``/``W'`` dependen del stride).
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class MiniResNet(nn.Module):
    """
    Mini-ResNet para parches NxN (por ejemplo 32x32).

    Arquitectura:
        stem:   Conv(3->32)
        layer1: 2 bloques residuales (32 canales)
        layer2: 2 bloques residuales (64 canales, stride 2)
        layer3: 2 bloques residuales (128 canales, stride 2)
        global avg pool + MLP

    Gracias a AdaptiveAvgPool2d no depende del patch_size exacto,
    mientras sea razonable (>= 16x16).
    """
    def __init__(self, num_classes, dropout=0.3):
        """Construye el stem, las tres etapas residuales y el clasificador final.

        Args:
            num_classes: Número de clases de salida.
            dropout: Probabilidad de dropout en el clasificador final.
        """
        super().__init__()

        # Bloque inicial
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Bloques residuales
        self.layer1 = self._make_layer(32, 32, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(32, 64, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(64, 128, num_blocks=2, stride=2)

        # Global Average Pooling a 1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Clasificador
        self.classifier = nn.Sequential(
            nn.Flatten(),              # 128 * 1 * 1
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        """Crea una secuencia de bloques residuales.

        El primer bloque puede tener stride > 1 para hacer downsampling y
        cambiar el número de canales; el resto mantiene ``out_channels`` y
        stride 1.

        Args:
            in_channels: Canales de entrada del primer bloque.
            out_channels: Canales de salida de todos los bloques de la etapa.
            num_blocks: Número de bloques :class:`BasicBlock` en la etapa.
            stride: Stride del primer bloque (para downsampling).

        Returns:
            nn.Sequential: Secuencia de ``num_blocks`` bloques residuales.
        """
        layers = []
        # Primer bloque (puede cambiar canales / stride)
        layers.append(BasicBlock(in_channels, out_channels, stride=stride))
        # Resto de bloques con stride=1 y mismos canales
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        """Propaga el batch de entrada a través del stem, las etapas residuales y el clasificador.

        Args:
            x (torch.Tensor): Batch de imágenes con forma
                ``(N, 3, patch_size, patch_size)``.

        Returns:
            torch.Tensor: Logits sin normalizar, de forma
            ``(N, num_classes)``.
        """
        x = self.stem(x)       # 3 -> 32 canales
        x = self.layer1(x)     # mantiene resolución
        x = self.layer2(x)     # reduce resolución (stride 2)
        x = self.layer3(x)     # reduce resolución (stride 2)
        x = self.avgpool(x)    # -> 128 x 1 x 1
        x = self.classifier(x)
        return x


# ======================= DATA LOADERS =======================

def make_loaders(data_dir, patch_size=32, batch_size=64, num_workers=2):
    """Crea los ``DataLoader`` de train/val/test a partir de una estructura ``ImageFolder``.

    El split de train recibe augmentación más agresiva que en
    ``train_cnn_patches.py`` (recorte aleatorio, flip, rotación, jitter de
    color y blur gaussiano); val y test solo se redimensionan y normalizan.

    Args:
        data_dir: Carpeta raíz del dataset, con subcarpetas
            ``train/``, ``val/`` y ``test/`` (cada una con una subcarpeta
            por clase).
        patch_size: Lado (en píxeles) al que se redimensionan los parches.
        batch_size: Tamaño de batch para los tres loaders.
        num_workers: Número de procesos worker para la carga de datos.

    Returns:
        tuple: ``(train_set, val_set, test_set, train_loader, val_loader,
        test_loader)``, donde los ``*_set`` son instancias de
        ``torchvision.datasets.ImageFolder`` y los ``*_loader`` son
        ``torch.utils.data.DataLoader``.
    """

    # Normalización genérica; si tienes mean/std de tu dataset, puedes
    # cambiarlas aquí.
    MEAN = [0.5, 0.5, 0.5]
    STD  = [0.5, 0.5, 0.5]

    train_tfm = transforms.Compose([
        transforms.Resize((patch_size, patch_size)),
        transforms.RandomResizedCrop(patch_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    eval_tfm = transforms.Compose([
        transforms.Resize((patch_size, patch_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
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


def compute_class_weights(dataset, max_weight=10.0):
    """Calcula pesos inversamente proporcionales a la frecuencia de cada clase.

    Se usan como ``weight`` de ``nn.CrossEntropyLoss`` para compensar el
    desbalance entre clases.

    Args:
        dataset: Dataset tipo ``ImageFolder`` (iterable de pares
            ``(imagen, etiqueta_entera)``).
        max_weight: Tope superior del peso de cualquier clase. Sin este
            límite, una clase con muy pocos ejemplos (p.ej. 5 imágenes de
            "Esponjas" frente a 20000+ de "Coral") recibe un peso cientos
            de veces mayor al de las clases mayoritarias, lo que
            desestabiliza el entrenamiento sin que el modelo pueda
            realmente aprender esa clase con tan pocos datos.

    Returns:
        torch.Tensor: Tensor 1D de tipo ``float32`` con un peso por clase
        (capado a ``max_weight``), en el mismo orden que ``dataset.classes``.
    """
    ys = [y for _, y in dataset]
    counts = np.bincount(ys)
    counts = counts + 1e-6  # evitar división por cero
    weights = counts.sum() / (len(counts) * counts)
    weights = np.clip(weights, a_min=None, a_max=max_weight)
    return torch.tensor(weights, dtype=torch.float32)


# ======================= LOOP DE ENTRENAMIENTO =======================

def train_epoch(model, loader, criterion, optimizer, device):
    """Entrena el modelo durante una época sobre ``loader``.

    A diferencia de ``train_cnn_patches.py``, aquí se acumulan todas las
    predicciones de la época para calcular, además del accuracy, el F1
    macro (métrica principal de este script).

    Args:
        model (nn.Module): Modelo a entrenar (se pone en modo ``train``).
        loader (DataLoader): Loader del split de entrenamiento.
        criterion: Función de pérdida (p.ej. ``nn.CrossEntropyLoss``).
        optimizer: Optimizador de PyTorch ya asociado a los parámetros del modelo.
        device: Dispositivo (``"cuda"`` o ``"cpu"``) donde mover los tensores.

    Returns:
        tuple[float, float, float]: ``(loss_promedio, accuracy, f1_macro)``
        de la época.
    """
    model.train()
    total_loss = 0.0
    all_y, all_p = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * x.size(0)
        preds = out.argmax(1)

        all_y.append(y.cpu())
        all_p.append(preds.cpu())

    y_true = torch.cat(all_y).numpy()
    y_pred = torch.cat(all_p).numpy()

    acc = (y_true == y_pred).mean()
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return total_loss / len(y_true), acc, f1


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    """Evalúa el modelo (sin actualizar pesos) sobre ``loader``.

    Args:
        model (nn.Module): Modelo a evaluar (se pone en modo ``eval``).
        loader (DataLoader): Loader del split a evaluar (val o test).
        criterion: Función de pérdida usada solo para reportar el valor.
        device: Dispositivo (``"cuda"`` o ``"cpu"``) donde mover los tensores.

    Returns:
        tuple[float, float, float]: ``(loss_promedio, accuracy, f1_macro)``
        sobre todo el split.
    """
    model.eval()
    total_loss = 0.0
    all_y, all_p = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)

        total_loss += float(loss) * x.size(0)
        preds = out.argmax(1)

        all_y.append(y.cpu())
        all_p.append(preds.cpu())

    y_true = torch.cat(all_y).numpy()
    y_pred = torch.cat(all_p).numpy()

    acc = (y_true == y_pred).mean()
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return total_loss / len(y_true), acc, f1


@torch.no_grad()
def predict_all(model, loader, device):
    """Genera predicciones del modelo para todo un ``DataLoader``.

    Args:
        model (nn.Module): Modelo ya entrenado (se pone en modo ``eval``).
        loader (DataLoader): Loader sobre el que predecir (típicamente test).
        device: Dispositivo (``"cuda"`` o ``"cpu"``) donde mover los tensores.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: ``(y_true, y_pred)``, arrays
        1D con las etiquetas verdaderas y las predichas.
    """
    model.eval()
    yy, pp = [], []
    for x, y in loader:
        x = x.to(device)
        out = model(x)
        pred = out.argmax(1).cpu().numpy().tolist()
        pp += pred
        yy += y.numpy().tolist()
    return np.array(yy), np.array(pp)


# ======================= MAIN =======================

def main():
    """Punto de entrada del script: entrena, evalúa y guarda resultados de la Mini-ResNet.

    Lee los argumentos de línea de comandos, construye los loaders y el
    modelo :class:`MiniResNet`, ejecuta el loop de entrenamiento con
    scheduler ``ReduceLROnPlateau`` y early stopping, ambos gobernados por
    el F1 macro de validación, y al terminar evalúa el mejor checkpoint
    sobre el split de test, guardando métricas, reporte de clasificación,
    matriz de confusión y las curvas de F1/loss en ``--results_dir``.

    No recibe argumentos ni devuelve nada directamente: toda la
    configuración se lee de ``sys.argv`` mediante ``argparse`` (ver
    ``python scripts/train_miniresnet_patches.py --help`` para la lista
    completa de opciones).
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True,
                    help="Carpeta raíz del dataset (ej: ./datasets/patches)")
    ap.add_argument("--patch_size", type=int, default=32,
                    help="Tamaño de los parches NxN (ej: 32 para 32x32)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--patience", type=int, default=10,
                    help="Paciencia para early stopping (por F1 de validación)")
    ap.add_argument("--results_dir", type=str, default="results/results_miniresnet",
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

    # Modelo
    model = MiniResNet(num_classes=num_classes, dropout=args.dropout).to(device)

    # Pérdida ponderada + optimizador + scheduler
    weights = compute_class_weights(train_set).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",       # maximizamos F1
        factor=0.5,
        patience=3,
    )

    # Map de etiquetas
    with open(results_dir / "label_map_miniresnet.json", "w", encoding="utf-8") as f:
        json.dump({i: c for i, c in enumerate(train_set.classes)},
                  f, ensure_ascii=False, indent=2)

    best_val_f1 = -1.0
    best_state = None
    patience = args.patience
    history = []

    # Loop de entrenamiento
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc, va_f1 = eval_epoch(model, val_loader, criterion, device)
        te_loss, te_acc, te_f1 = eval_epoch(model, test_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": tr_loss, "train_acc": tr_acc, "train_f1": tr_f1,
            "val_loss":   va_loss, "val_acc":   va_acc,   "val_f1":   va_f1,
            "test_loss":  te_loss, "test_acc":  te_acc,  "test_f1":  te_f1,
        })

        print(
            f"[{epoch:03d}] "
            f"lr={current_lr:.2e} | "
            f"train_f1={tr_f1:.4f} | val_f1={va_f1:.4f} | test_f1={te_f1:.4f}"
        )

        # Scheduler por F1 de validación
        scheduler.step(va_f1)

        # Early stopping por F1 de validación
        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience = args.patience
        else:
            patience -= 1
            if patience <= 0:
                print("Early stopping por F1 de validación.")
                break

    # Guardar historial + mejor modelo
    pd.DataFrame(history).to_csv(results_dir / "metrics_miniresnet.csv", index=False)
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(model.state_dict(), results_dir / "best_miniresnet.pt")
        print(f"Guardado {results_dir / 'best_miniresnet.pt'} (best_val_f1={best_val_f1:.4f})")

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

    with open(results_dir / "test_report_miniresnet.txt", "w", encoding="utf-8") as f:
        f.write(rep + "\n\nConfusion Matrix:\n" + np.array2string(cm))

    print("\n=== Test Report (MiniResNet) ===")
    print(rep)
    print("Confusion Matrix:\n", cm)

    # --- GRÁFICAS: F1 y loss por época (train/val/test) ---
    try:
        epochs = [h["epoch"] for h in history]
        tr_f1 = [h["train_f1"] for h in history]
        va_f1 = [h["val_f1"]   for h in history]
        te_f1 = [h["test_f1"]  for h in history]

        tr_loss = [h["train_loss"] for h in history]
        va_loss = [h["val_loss"]   for h in history]
        te_loss = [h["test_loss"]  for h in history]

        # F1
        plt.figure()
        plt.plot(epochs, tr_f1, label="train_f1")
        plt.plot(epochs, va_f1, label="val_f1")
        plt.plot(epochs, te_f1, label="test_f1")
        plt.xlabel("Epoch")
        plt.ylabel("F1 macro")
        plt.title("F1 por época (MiniResNet)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(results_dir / "curves_f1_miniresnet.png", dpi=150)

        # Loss
        plt.figure()
        plt.plot(epochs, tr_loss, label="train_loss")
        plt.plot(epochs, va_loss, label="val_loss")
        plt.plot(epochs, te_loss, label="test_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss por época (MiniResNet)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(results_dir / "curves_loss_miniresnet.png", dpi=150)

        # Matriz de confusión como heatmap
        plt.figure(figsize=(6, 5))
        im = plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.title("Matriz de confusión (MiniResNet)")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        tick_marks = np.arange(len(train_set.classes))
        plt.xticks(tick_marks, train_set.classes, rotation=45, ha="right")
        plt.yticks(tick_marks, train_set.classes)
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        plt.savefig(results_dir / "confusion_matrix_miniresnet.png", dpi=150)

        print("Guardadas curvas y matriz de confusión en la carpeta de resultados.")
    except Exception as e:
        print(f"[AVISO] No se pudieron generar las gráficas: {e}")


if __name__ == "__main__":
    main()