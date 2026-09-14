#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_cnn_patches.py.

Entrena una CNN sencilla (:class:`ConvNet`) sobre parches NxN organizados
en estructura ``ImageFolder``::

    data_dir/
        train/clase_i/*.png
        val/clase_i/*.png
        test/clase_i/*.png

Características:
    - Usa data augmentation en train (blur, flip, rotación, color jitter).
    - Pondera las clases con ``class_weights`` para compensar el
      desbalance entre categorías de cobertura.
    - Aplica early stopping por ``val_acc``.
    - Guarda en ``./results_cnn`` (o ``--results_dir``):
        - ``best_cnn.pt``: pesos del mejor modelo según val_acc.
        - ``metrics_cnn.csv``: histórico de pérdida/accuracy/F1 macro por época.
        - ``test_report_cnn.txt``: reporte de clasificación + matriz de confusión.
        - ``label_map_cnn.json``: mapa índice -> nombre de clase.
        - ``curves_accuracy_cnn.png`` / ``curves_f1_cnn.png`` / ``curves_loss_cnn.png``: curvas de entrenamiento.
        - ``confusion_matrix_cnn.png``: matriz de confusión como heatmap.

Uso típico:
    python scripts/train_cnn_patches.py --data_dir ./datasets/32x32 --epochs 40
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

from patch_dataset import make_hierarchical_loaders, hierarchical_breakdown


def get_device():
    """Elige cuda > DirectML (GPU AMD/Intel en Windows) > cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_directml
        if torch_directml.is_available():
            return torch_directml.device()
    except ImportError:
        pass
    return torch.device("cpu")


# ----------------------- MODELO CNN -----------------------

class ConvNet(nn.Module):
    """
    CNN pensada para parches 32x32 (3 canales).
    Si usas otro patch_size habría que ajustar el tamaño
    de la capa fully-connected (128 * 4 * 4 asume 3 pools).
    """
    def __init__(self, num_classes, dropout=0.3):
        """Inicializa las capas convolucionales y el clasificador.

        Args:
            num_classes: Número de clases de salida (tamaño de la capa
                final totalmente conectada).
            dropout: Probabilidad de dropout aplicada antes de la última
                capa lineal del clasificador.
        """
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
        """Propaga el batch de entrada a través de la red.

        Args:
            x (torch.Tensor): Batch de imágenes con forma
                ``(N, 3, patch_size, patch_size)``.

        Returns:
            torch.Tensor: Logits sin normalizar, de forma
            ``(N, num_classes)``.
        """
        x = self.features(x)
        x = self.classifier(x)
        return x


# ----------------------- DATA LOADERS -----------------------

class _ClassLockedImageFolder(datasets.ImageFolder):
    """``ImageFolder`` cuyo mapeo clase->índice es fijo (pasado
    explícitamente), en vez de auto-descubrirse a partir de las subcarpetas
    presentes en ``root``.

    Necesario porque no todas las clases tienen ejemplos en todos los
    splits (p.ej. una clase rara puede no tener ninguna imagen en val o
    test): si cada split auto-descubre su propio ``class_to_idx``, dos
    splits con distinto conjunto de clases presentes terminan con índices
    desalineados entre sí, y las métricas y la matriz de confusión de ese
    split quedan silenciosamente mal etiquetadas aunque el entrenamiento
    no falle.

    Definida a nivel de módulo (no como clase anidada) porque
    ``DataLoader(num_workers>0)`` en Windows usa ``spawn`` y necesita poder
    hacer pickle del dataset para mandarlo a los procesos worker; una clase
    anidada dentro de una función no es picklable.
    """

    def __init__(self, root, class_to_idx, transform=None):
        self._fixed_class_to_idx = dict(class_to_idx)
        super().__init__(root, transform=transform, allow_empty=True)

    def find_classes(self, directory):
        return list(self._fixed_class_to_idx.keys()), dict(self._fixed_class_to_idx)


def make_loaders(data_dir, patch_size=32, batch_size=64, num_workers=2,
                  task="major", min_class_count=0, debug=False):
    """Crea los ``DataLoader`` de train/val/test.

    El split de train recibe augmentación (blur gaussiano, flip horizontal,
    rotación y jitter de color); val y test solo se redimensionan y
    normalizan.

    Args:
        data_dir: Carpeta raíz del dataset, con subcarpetas
            ``train/``, ``val/`` y ``test/`` (cada una con una subcarpeta
            por clase).
        patch_size: Lado (en píxeles) al que se redimensionan los parches.
        batch_size: Tamaño de batch para los tres loaders.
        num_workers: Número de procesos worker para la carga de datos.
        task: ``"major"`` (por defecto) usa ``ImageFolder`` normal y
            clasifica por categoría mayor, igual que siempre (el
            aprendizaje jerárquico queda "apagado"). ``"species"``
            clasifica por categoría+subcategoría y ``"condition"``
            clasifica la condición de salud del coral (sano/blanqueado/
            etc.) agrupando todas las especies — ver ``patch_dataset.py``.
        min_class_count: Solo aplica a ``task in {"species","condition"}``:
            excluye clases con menos de este total de parches (ver
            DECISIONES.md, exclusión por umbral).
        debug: Si es ``True``, imprime detalle de clases excluidas por umbral.

    Returns:
        tuple: ``(train_set, val_set, test_set, train_loader, val_loader,
        test_loader, dropped)``. Los ``*_set`` son ``ImageFolder`` (task
        ``"major"``) o ``PatchFolder`` (``"species"``/``"condition"``);
        ``dropped`` es la lista de clases excluidas por el umbral (vacía
        en ``task="major"``).
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
        transforms.Normalize(mean = [0.4144, 0.3986, 0.2989],
                             std  = [0.2069, 0.2155, 0.2175]),
    ])

    eval_tfm = transforms.Compose([
        transforms.Resize((patch_size, patch_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.4144, 0.3986, 0.2989],
                             std  = [0.2069, 0.2155, 0.2175]),
    ])

    if task != "major":
        return make_hierarchical_loaders(
            data_dir, stage=task, train_tfm=train_tfm, eval_tfm=eval_tfm,
            min_class_count=min_class_count, batch_size=batch_size,
            num_workers=num_workers, debug=debug,
        )

    train_set = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tfm)
    # val/test heredan el mapeo clase->índice de train (ver _locked_image_folder).
    val_set   = _ClassLockedImageFolder(os.path.join(data_dir, "val"),  train_set.class_to_idx, transform=eval_tfm)
    test_set  = _ClassLockedImageFolder(os.path.join(data_dir, "test"), train_set.class_to_idx, transform=eval_tfm)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_set, val_set, test_set, train_loader, val_loader, test_loader, []


def compute_class_weights(dataset, max_weight=10.0):
    """Calcula pesos inversamente proporcionales a la frecuencia de cada clase.

    Se usan como ``weight`` de ``nn.CrossEntropyLoss`` para compensar el
    desbalance entre clases (p.ej. muchos más parches de "Coral" que de
    "Otros organismos").

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


# ----------------------- LOOP DE ENTRENAMIENTO -----------------------

def train_epoch(model, loader, criterion, optimizer, device):
    """Entrena el modelo durante una época sobre ``loader``.

    Además del accuracy, acumula todas las predicciones de la época para
    calcular también el F1 macro (más robusto que accuracy ante el
    desbalance de clases del dataset), como en ``train_miniresnet_patches.py``.

    Args:
        model (nn.Module): Modelo a entrenar (se pone en modo ``train``).
        loader (DataLoader): Loader del split de entrenamiento.
        criterion: Función de pérdida (p.ej. ``nn.CrossEntropyLoss``).
        optimizer: Optimizador de PyTorch ya asociado a los parámetros del modelo.
        device: Dispositivo (``"cuda"`` o ``"cpu"``) donde mover los tensores.

    Returns:
        tuple[float, float, float]: ``(loss_promedio, accuracy, f1_macro)``
        de la época, promediados sobre todas las muestras de ``loader``.
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
        1D con las etiquetas verdaderas y las predichas, en el mismo orden
        en que se recorrió el loader.
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


def subcategory_breakdown(dataset, y_true, y_pred, class_names):
    """Desglosa el acierto de test por subcategoría real, aunque el modelo
    clasifique por categoría mayor.

    El dataset guarda cada parche en ``<major>/<subcategoria>/archivo.jpg``
    (o solo ``<major>/archivo.jpg`` si esa categoría mayor no tiene
    subcategoría propia, p.ej. TAPE). El modelo solo predice la categoría
    mayor (ver nota en :func:`make_loaders`), así que esto no mide una
    predicción de subcategoría - mide, dentro de cada subcategoría real
    (p.ej. "Psammocora stellata"), qué tan seguido el modelo acertó la
    categoría mayor a la que pertenece. Sirve para detectar que el modelo
    le va mal específicamente a una especie/subcategoría aunque en
    promedio le vaya bien a su categoría mayor.

    Args:
        dataset: ``ImageFolder`` (o :class:`_ClassLockedImageFolder`) del
            split de test, con ``.samples`` en el mismo orden que
            ``y_true``/``y_pred`` (requiere ``shuffle=False`` en el loader).
        y_true: Índices de categoría mayor verdaderos (de :func:`predict_all`).
        y_pred: Índices de categoría mayor predichos (de :func:`predict_all`).
        class_names: Nombres de categoría mayor en orden de índice
            (``train_set.classes``).

    Returns:
        pandas.DataFrame: una fila por subcategoría, con columnas
        ``major_category``, ``subcategory``, ``n``, ``correct``, ``accuracy``,
        ordenado por categoría mayor y luego por accuracy ascendente (las
        subcategorías más problemáticas primero).
    """
    root = Path(dataset.root)
    rows = []
    for (filepath, _), yt, yp in zip(dataset.samples, y_true, y_pred):
        rel_parts = Path(filepath).relative_to(root).parts[:-1]  # sin el archivo
        subcat = rel_parts[1] if len(rel_parts) >= 2 else rel_parts[0]
        rows.append({
            "major_category": class_names[yt],
            "subcategory": subcat,
            "correct": int(yt == yp),
        })
    df = pd.DataFrame(rows)
    summary = df.groupby(["major_category", "subcategory"]).agg(
        n=("correct", "size"), correct=("correct", "sum")
    ).reset_index()
    summary["accuracy"] = summary["correct"] / summary["n"]
    return summary.sort_values(["major_category", "accuracy"]).reset_index(drop=True)


# ----------------------- MAIN -----------------------

def main():
    """Punto de entrada del script: entrena, evalúa y guarda resultados de la CNN.

    Lee los argumentos de línea de comandos, construye los loaders y el
    modelo :class:`ConvNet`, ejecuta el loop de entrenamiento con early
    stopping por ``val_acc``, y al terminar evalúa el mejor checkpoint
    sobre el split de test, guardando métricas, reporte de clasificación,
    matriz de confusión y las curvas de accuracy/F1 macro/loss en ``--results_dir``.

    No recibe argumentos ni devuelve nada directamente: toda la
    configuración se lee de ``sys.argv`` mediante ``argparse`` (ver
    ``python scripts/train_cnn_patches.py --help`` para la lista completa
    de opciones).
    """
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
    ap.add_argument("--results_dir", type=str, default="results/results_cnn",
                    help="Carpeta donde se guardan los resultados")
    ap.add_argument("--task", choices=["major", "species", "condition"], default="major",
                    help="'major' (por defecto) = comportamiento de siempre, clasifica por "
                         "categoría mayor. 'species' = categoría+subcategoría. 'condition' = "
                         "condición de salud del coral (sano/blanqueado/etc.), agrupando "
                         "especies. 'species'/'condition' son el aprendizaje jerárquico "
                         "('encendido'); 'major' lo deja 'apagado'.")
    ap.add_argument("--min_class_count", type=int, default=50,
                    help="Solo para --task species/condition: excluye clases con menos de "
                         "este total de parches (train+val+test). Ver DECISIONES.md.")
    args = ap.parse_args()

    device = get_device()
    print("Device:", device)

    results_dir = Path(args.results_dir) / args.task
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Guardando resultados en: {results_dir.resolve()}")

    # Loaders
    train_set, val_set, test_set, train_loader, val_loader, test_loader, dropped = make_loaders(
        args.data_dir,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        task=args.task,
        min_class_count=args.min_class_count,
        debug=True,
    )
    if dropped:
        with open(results_dir / "excluded_classes.txt", "w", encoding="utf-8") as f:
            f.write(f"task={args.task} min_class_count={args.min_class_count}\n")
            for c, n in dropped:
                f.write(f"{c}\t{n}\n")

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
        tr_loss, tr_acc, tr_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc, va_f1 = eval_epoch(model, val_loader, criterion, device)
        te_loss, te_acc, te_f1 = eval_epoch(model, test_loader, criterion, device)

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss, "train_acc": tr_acc, "train_f1": tr_f1,
            "val_loss":   va_loss, "val_acc":   va_acc,   "val_f1":   va_f1,
            "test_loss":  te_loss, "test_acc":  te_acc,  "test_f1":  te_f1,
        })

        print(
            f"[{epoch:03d}] "
            f"train_loss={tr_loss:.4f} acc={tr_acc:.4f} f1={tr_f1:.4f} | "
            f"val_loss={va_loss:.4f} acc={va_acc:.4f} f1={va_f1:.4f} | "
            f"test_loss={te_loss:.4f} acc={te_acc:.4f} f1={te_f1:.4f}"
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

    # Desglose por eje secundario: en task="major" es subcategoría real
    # (subcategory_breakdown, la clasificación sigue siendo por categoría
    # mayor); en task="species"/"condition" es la condición/especie real
    # (hierarchical_breakdown, ver patch_dataset.py).
    if args.task == "major":
        sub_df = subcategory_breakdown(test_set, y_true, y_pred, train_set.classes)
    else:
        sub_df = hierarchical_breakdown(test_set, y_true, y_pred, train_set.classes)
    sub_df.to_csv(results_dir / "test_report_by_subcategory_cnn.csv", index=False)
    print("\n=== Desglose por subcategoría (test) ===")
    print(sub_df.to_string(index=False))

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

        # F1 macro
        plt.figure()
        plt.plot(epochs, [h["train_f1"] for h in history], label="train_f1")
        plt.plot(epochs, [h["val_f1"]   for h in history], label="val_f1")
        plt.plot(epochs, [h["test_f1"]  for h in history], label="test_f1")
        plt.xlabel("Epoch")
        plt.ylabel("F1 macro")
        plt.title("F1 macro por época (CNN)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(results_dir / "curves_f1_cnn.png", dpi=150)

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

        print("Guardadas en results_cnn/: curves_accuracy_cnn.png, curves_f1_cnn.png, curves_loss_cnn.png y confusion_matrix_cnn.png")
    except Exception as e:
        print(f"[AVISO] No se pudieron generar las gráficas: {e}")


if __name__ == "__main__":
    main()