#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_finetune.py.

Fine-tuning de modelos preentrenados de ``torchvision`` (ResNet18, VGG16,
AlexNet, MobileNetV2) sobre un dataset ``ImageFolder`` con estructura
``train/val/test``.

Permite controlar cuánto del backbone preentrenado se congela mediante
``--freeze_level`` (ver :func:`main`): desde entrenar solo la última capa
hasta reentrenar toda la red.

Guarda resultados en ``./results_finetune/<model_name>/`` (o
``--results_dir/<model_name>/``): mapa de etiquetas, histórico de
métricas, mejor checkpoint, reporte de clasificación en test, curvas de
accuracy/F1 macro/loss por época y matriz de confusión como heatmap (mismo
formato que ``train_cnn_patches.py`` / ``train_mlp_patches.py`` /
``train_miniresnet_patches.py``, para poder comparar los modelos entre sí).

Uso típico:
    python scripts/train_finetune.py --data_dir ./datasets/224x224 \
        --model resnet18 --freeze_level 2
"""

import os, argparse, json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------- CARGAR MODELO ----------------------

def load_pretrained_model(name, num_classes):
    """Carga un modelo preentrenado de torchvision y reemplaza su última capa.

    La capa de clasificación final se sustituye por una nueva capa lineal
    con salida ``num_classes``, dejando el resto del backbone con los
    pesos preentrenados en ImageNet.

    Args:
        name: Nombre del modelo a cargar. Uno de
            ``"resnet18"``, ``"vgg16"``, ``"alexnet"`` o ``"mobilenet_v2"``
            (insensible a mayúsculas).
        num_classes: Número de clases de salida del nuevo clasificador.

    Returns:
        nn.Module: El modelo de ``torchvision`` con la última capa
        reemplazada, listo para fine-tuning.

    Raises:
        ValueError: Si ``name`` no es uno de los modelos soportados.
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
    """Crea los ``DataLoader`` de train/val/test a partir de una estructura ``ImageFolder``.

    Las imágenes se escalan a ``img_size x img_size`` (224x224 por
    defecto) y se normalizan con las estadísticas de ImageNet, porque los
    modelos preentrenados de ``torchvision`` lo requieren. El split de
    train recibe augmentación ligera (flip horizontal y rotación).

    Args:
        data_dir: Carpeta raíz del dataset, con subcarpetas
            ``train/``, ``val/`` y ``test/`` (cada una con una subcarpeta
            por clase).
        img_size: Lado (en píxeles) al que se redimensionan las imágenes.
        batch_size: Tamaño de batch para los tres loaders.
        num_workers: Número de procesos worker para la carga de datos.

    Returns:
        tuple: ``(train_set, val_set, test_set, train_loader, val_loader,
        test_loader)``, donde los ``*_set`` son instancias de
        ``torchvision.datasets.ImageFolder`` y los ``*_loader`` son
        ``torch.utils.data.DataLoader``.
    """

    train_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                             std  = [0.229, 0.224, 0.225]),
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


def compute_class_weights(dataset, max_weight=10.0):
    """Calcula pesos inversamente proporcionales a la frecuencia de cada clase.

    Se usan como ``weight`` de ``nn.CrossEntropyLoss`` para compensar el
    desbalance entre clases (mismo criterio que ``train_cnn_patches.py`` /
    ``train_mlp_patches.py`` / ``train_miniresnet_patches.py``).

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


# ---------------------- ENTRENAMIENTO ----------------------

def train_epoch(model, loader, criterion, optimizer, device):
    """Entrena el modelo durante una época sobre ``loader``.

    Solo se actualizan los parámetros con ``requires_grad=True`` (los que
    ``main()`` dejó "descongelados" según ``--freeze_level``), gracias a
    que ``optimizer`` se construyó ya filtrado. Además del accuracy,
    acumula todas las predicciones de la época para calcular también el
    F1 macro (más robusto ante el desbalance de clases), como en
    ``train_miniresnet_patches.py``.

    Args:
        model (nn.Module): Modelo a entrenar (se pone en modo ``train``).
        loader (DataLoader): Loader del split de entrenamiento.
        criterion: Función de pérdida (p.ej. ``nn.CrossEntropyLoss``).
        optimizer: Optimizador de PyTorch ya asociado a los parámetros
            entrenables del modelo.
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
        pred = out.argmax(1).cpu().numpy()
        yy.extend(y.numpy())
        pp.extend(pred)
    return np.array(yy), np.array(pp)


# ---------------------- MAIN ----------------------

def main():
    """Punto de entrada del script: hace fine-tuning, evalúa y guarda resultados.

    Carga el modelo preentrenado indicado en ``--model`` mediante
    :func:`load_pretrained_model`, aplica el esquema de congelamiento de
    capas indicado en ``--freeze_level``:

    - ``0``: entrena todo el modelo (sin congelar nada).
    - ``1``: congela todo excepto la última capa de clasificación.
    - ``2`` (por defecto): congela el backbone y entrena las últimas capas
      (p.ej. ``layer4`` + ``fc`` en ResNet18) junto con el clasificador.
    - ``3``: congela solo las capas más tempranas y entrena la mitad final
      de la red junto con el clasificador.

    Después entrena con ``AdamW`` (pérdida ponderada por clase, como en
    los demás scripts de entrenamiento) sobre los parámetros no
    congelados, con early stopping por ``val_acc`` (``--patience``), guarda
    el mejor checkpoint, y evalúa el modelo final sobre el split de test,
    escribiendo el mapa de etiquetas, el histórico de métricas, el
    checkpoint, el reporte de clasificación y las curvas de accuracy/F1
    macro/loss + matriz de confusión en ``--results_dir/<model>/``.

    No recibe argumentos ni devuelve nada directamente: toda la
    configuración se lee de ``sys.argv`` mediante ``argparse`` (ver
    ``python scripts/train_finetune.py --help`` para la lista completa de
    opciones).
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--model", required=True,
                    help="resnet18 | vgg16 | alexnet | mobilenet_v2")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--results_dir", type=str, default="results/results_finetune")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--patience", type=int, default=10,
                help="Paciencia para early stopping (por val_acc)")
    ap.add_argument("--freeze_level", type=int, default=2,
                help="0 = sin congelar (todo entrenable), 1 = solo última capa, "
                     "2 = últimas capas del backbone, 3 = mitad de la red")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_set, val_set, test_set, train_loader, val_loader, test_loader = \
        make_loaders(args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

    num_classes = len(train_set.classes)

    model = load_pretrained_model(args.model, num_classes).to(device)

    # ============================================================
#   FREEZE LEVEL SYSTEM
#   freeze_level: 1, 2 o 3
# ============================================================

    m = args.model.lower()
    freeze = args.freeze_level

    print(f"Usando freeze_level = {freeze}")

    # ------------------------------------------
    # FUNCIONES AUXILIARES PARA FREEZING
    # ------------------------------------------

    def freeze_all(model):
        for p in model.parameters():
            p.requires_grad = False

    def unfreeze(params):
        for p in params:
            p.requires_grad = True

    # ------------------------------------------
    # FREEZE LEVEL 1
    # Solo última capa
    # ------------------------------------------
    
    if freeze == 1:
        freeze_all(model)

        if m == "resnet18":
            unfreeze(model.fc.parameters())

        elif m == "vgg16" or m == "alexnet":
            unfreeze(model.classifier.parameters())

        elif m == "mobilenet_v2":
            unfreeze(model.classifier.parameters())

    # ------------------------------------------
    # FREEZE LEVEL 2
    # Congelar backbone, entrenar últimas capas
    # ------------------------------------------
    elif freeze == 2:
        freeze_all(model)

        if m == "resnet18":
            unfreeze(model.layer4.parameters())   # último bloque residual
            unfreeze(model.fc.parameters())

        elif m == "vgg16" or m == "alexnet":
            # permitir entrenamiento del classifier completo
            unfreeze(model.classifier.parameters())

        elif m == "mobilenet_v2":
            unfreeze(model.features[-1].parameters())   # último bloque conv
            unfreeze(model.classifier.parameters())

    # ------------------------------------------
    # FREEZE LEVEL 3
    # Congelar primeras capas, entrenar mitad final
    # ------------------------------------------
    elif freeze == 3:

        if m == "resnet18":
            # congelar los bloques tempranos
            for p in model.layer1.parameters(): p.requires_grad = False
            for p in model.layer2.parameters(): p.requires_grad = False

            # permitir capa media, final y fc
            unfreeze(model.layer3.parameters())
            unfreeze(model.layer4.parameters())
            unfreeze(model.fc.parameters())

        elif m == "vgg16":
            # congelar primeras 10 conv (features[:16])
            for p in model.features[:16].parameters():
                p.requires_grad = False

            # entrenar conv profundas + classifier
            unfreeze(model.features[16:].parameters())
            unfreeze(model.classifier.parameters())

        elif m == "alexnet":
            for p in model.features[:4].parameters(): p.requires_grad = False
            unfreeze(model.features[4:].parameters())
            unfreeze(model.classifier.parameters())

        elif m == "mobilenet_v2":
            # congelar mitad inicial de bloques
            for p in model.features[:7].parameters(): p.requires_grad = False
            unfreeze(model.features[7:].parameters())
            unfreeze(model.classifier.parameters())

    elif freeze == 0:
        print("Entrenando TODO el modelo (sin freeze).")

    else:
        raise ValueError("--freeze_level debe ser 0, 1, 2 o 3")

    # ------------------------------------------
    # OPTIMIZADOR solo para parámetros entrenables
    # ------------------------------------------
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )
    weights = compute_class_weights(train_set).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    # optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # Carpeta de resultados
    result_path = Path(args.results_dir) / args.model
    result_path.mkdir(parents=True, exist_ok=True)

    # Guardar mapa de clases
    with open(result_path / "label_map.json", "w", encoding="utf-8") as f:
        json.dump({i: c for i, c in enumerate(train_set.classes)},
                  f, ensure_ascii=False, indent=2)

    history = []
    best_val_acc = -1
    best_state = None
    patience = args.patience

    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc, tr_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc, va_f1 = eval_epoch(model, val_loader, criterion, device)
        te_loss, te_acc, te_f1 = eval_epoch(model, test_loader, criterion, device)

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss, "train_acc": tr_acc, "train_f1": tr_f1,
            "val_loss":   va_loss, "val_acc":   va_acc,   "val_f1":   va_f1,
            "test_loss":  te_loss, "test_acc":  te_acc,  "test_f1":  te_f1,
        })

        print(f"[{epoch:03d}] "
              f"train_acc={tr_acc:.3f} f1={tr_f1:.3f} | "
              f"val_acc={va_acc:.3f} f1={va_f1:.3f} | "
              f"test_acc={te_acc:.3f} f1={te_f1:.3f}")

        # Early stopping por val_acc
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            # .cpu() crea una copia real; sin esto, best_state queda como
            # referencia a los tensores del modelo y termina reflejando la
            # última época (no la mejor) porque el optimizador los sigue
            # actualizando in-place tras esta asignación.
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience = args.patience
        else:
            patience -= 1
            if patience <= 0:
                print("Early stopping activado.")
                break

    # Guardar mejor modelo
    torch.save(best_state, result_path / "best_model.pt")
    print(f"Guardado {result_path / 'best_model.pt'} (val_acc={best_val_acc:.4f})")

    # Guardar métricas
    df = pd.DataFrame(history)
    df.to_csv(result_path / "metrics.csv", index=False)

    # Evaluación final
    model.load_state_dict(best_state)
    y_true, y_pred = predict_all(model, test_loader, device)

    all_labels = list(range(len(train_set.classes)))
    rep = classification_report(
        y_true, y_pred,
        labels=all_labels, target_names=train_set.classes,
        digits=4, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)

    with open(result_path / "test_report.txt", "w", encoding="utf-8") as f:
        f.write(rep + "\n\nConfusion Matrix:\n" + np.array2string(cm))

    print("\n=== Reporte final ===")
    print(rep)
    print("Confusion matrix:")
    print(cm)

    # --- GRÁFICAS: accuracy y loss por época (train/val/test) + matriz de confusión ---
    try:
        epochs = [h["epoch"] for h in history]
        tr_acc_h = [h["train_acc"] for h in history]
        va_acc_h = [h["val_acc"]   for h in history]
        te_acc_h = [h["test_acc"]  for h in history]

        tr_loss_h = [h["train_loss"] for h in history]
        va_loss_h = [h["val_loss"]   for h in history]
        te_loss_h = [h["test_loss"]  for h in history]

        # Accuracy
        plt.figure()
        plt.plot(epochs, tr_acc_h, label="train_acc")
        plt.plot(epochs, va_acc_h, label="val_acc")
        plt.plot(epochs, te_acc_h, label="test_acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy por época ({args.model})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(result_path / "curves_accuracy.png", dpi=150)

        # F1 macro
        plt.figure()
        plt.plot(epochs, [h["train_f1"] for h in history], label="train_f1")
        plt.plot(epochs, [h["val_f1"]   for h in history], label="val_f1")
        plt.plot(epochs, [h["test_f1"]  for h in history], label="test_f1")
        plt.xlabel("Epoch")
        plt.ylabel("F1 macro")
        plt.title(f"F1 macro por época ({args.model})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(result_path / "curves_f1.png", dpi=150)

        # Loss
        plt.figure()
        plt.plot(epochs, tr_loss_h, label="train_loss")
        plt.plot(epochs, va_loss_h, label="val_loss")
        plt.plot(epochs, te_loss_h, label="test_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Loss por época ({args.model})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(result_path / "curves_loss.png", dpi=150)

        # Matriz de confusión como heatmap
        plt.figure(figsize=(6, 5))
        im = plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.title(f"Matriz de confusión ({args.model})")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        tick_marks = np.arange(len(train_set.classes))
        plt.xticks(tick_marks, train_set.classes, rotation=45, ha="right")
        plt.yticks(tick_marks, train_set.classes)
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        plt.savefig(result_path / "confusion_matrix.png", dpi=150)

        print(f"Guardadas en {result_path}/: curves_accuracy.png, curves_f1.png, curves_loss.png y confusion_matrix.png")
    except Exception as e:
        print(f"[AVISO] No se pudieron generar las gráficas: {e}")


if __name__ == "__main__":
    main()