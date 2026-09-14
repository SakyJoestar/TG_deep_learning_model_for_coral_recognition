#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_mlp_patches.py.

Entrena un :class:`MLP` (perceptrón multicapa) sobre parches NxN
organizados en estructura ``ImageFolder``, aplanando cada imagen a un
vector de ``3 * patch_size * patch_size`` valores. Sirve como línea base
sencilla frente a los modelos convolucionales (:mod:`train_cnn_patches`,
:mod:`train_miniresnet_patches`).

Genera curvas de accuracy, F1 macro y loss por época (train/val/test),
matriz de confusión y guarda todo (PNGs, CSV de métricas, reporte de test
y mejor checkpoint) en la carpeta ``./results``.

Uso típico:
    python scripts/train_mlp_patches.py --data_dir ./datasets/32x32 --hidden 1024,512
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


def parse_hidden(s):
    """Convierte la cadena ``--hidden`` (p.ej. ``"1024,512"``) en una lista de tamaños de capa.

    Args:
        s: Cadena con los tamaños de las capas ocultas separados por comas.
            Si está vacía o es ``None``, se usa ``[1024, 512]`` por defecto.

    Returns:
        list[int]: Tamaños de las capas ocultas, en orden.
    """
    if not s:
        return [1024, 512]
    return [int(x) for x in s.split(",") if x.strip()]


class MLP(nn.Module):
    """Perceptrón multicapa simple: bloques Linear -> ReLU -> Dropout apilados.

    La entrada esperada es un vector plano (imagen aplanada), y la salida
    son los logits de clasificación.
    """
    def __init__(self, input_dim, num_classes, hidden, dropout=0.3):
        """Construye las capas ocultas y la capa de salida del MLP.

        Args:
            input_dim: Dimensión del vector de entrada (p.ej.
                ``3 * patch_size * patch_size`` para una imagen RGB aplanada).
            num_classes: Número de clases de salida.
            hidden: Lista con el tamaño de cada capa oculta, en orden
                (ver :func:`parse_hidden`).
            dropout: Probabilidad de dropout aplicada tras cada capa oculta.
        """
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, num_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """Propaga el batch de entrada a través del MLP.

        Args:
            x (torch.Tensor): Batch de vectores aplanados, forma ``(N, input_dim)``.

        Returns:
            torch.Tensor: Logits sin normalizar, de forma ``(N, num_classes)``.
        """
        return self.net(x)


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

    A diferencia de los scripts convolucionales, aquí no se aplica data
    augmentation: solo redimensionado y normalización, ya que el MLP se
    usa como línea base simple.

    Args:
        data_dir: Carpeta raíz del dataset, con subcarpetas
            ``train/``, ``val/`` y ``test/`` (cada una con una subcarpeta
            por clase).
        patch_size: Lado (en píxeles) al que se redimensionan los parches.
        batch_size: Tamaño de batch para los tres loaders.
        num_workers: Número de procesos worker para la carga de datos.
        task: ``"major"`` (por defecto, aprendizaje jerárquico "apagado")
            clasifica por categoría mayor con ``ImageFolder`` normal.
            ``"species"``/``"condition"`` usan ``patch_dataset.py``.
        min_class_count: Solo para ``task in {"species","condition"}``:
            excluye clases con menos de este total de parches.
        debug: Si es ``True``, imprime detalle de clases excluidas.

    Returns:
        tuple: ``(train_set, val_set, test_set, train_loader, val_loader,
        test_loader, dropped)``.
    """
    tfm = transforms.Compose([
        transforms.Resize((patch_size, patch_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if task != "major":
        return make_hierarchical_loaders(
            data_dir, stage=task, train_tfm=tfm, eval_tfm=tfm,
            min_class_count=min_class_count, batch_size=batch_size,
            num_workers=num_workers, debug=debug,
        )

    train_set = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=tfm)
    # val/test heredan el mapeo clase->índice de train (ver _locked_image_folder).
    val_set   = _ClassLockedImageFolder(os.path.join(data_dir, "val"),  train_set.class_to_idx, transform=tfm)
    test_set  = _ClassLockedImageFolder(os.path.join(data_dir, "test"), train_set.class_to_idx, transform=tfm)

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


def train_epoch(model, loader, criterion, optimizer, device):
    """Entrena el modelo durante una época sobre ``loader``.

    Cada batch de imágenes se aplana (``x.view(x.size(0), -1)``) antes de
    pasarlo al MLP, ya que este espera vectores planos. Además del
    accuracy, acumula todas las predicciones de la época para calcular
    también el F1 macro (más robusto ante el desbalance de clases), como
    en ``train_miniresnet_patches.py``.

    Args:
        model (nn.Module): Modelo :class:`MLP` a entrenar (modo ``train``).
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
        x = x.view(x.size(0), -1)
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
        model (nn.Module): Modelo :class:`MLP` a evaluar (modo ``eval``).
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
        x = x.view(x.size(0), -1)
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
    yy = []
    pp = []
    for x, y in loader:
        x = x.to(device)
        x = x.view(x.size(0), -1)
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


def main():
    """Punto de entrada del script: entrena, evalúa y guarda resultados del MLP.

    Lee los argumentos de línea de comandos, construye los loaders y el
    modelo :class:`MLP`, ejecuta el loop de entrenamiento con early
    stopping por ``val_acc``, y al terminar evalúa el mejor checkpoint
    sobre el split de test, guardando métricas, reporte de clasificación,
    matriz de confusión y las curvas de accuracy/F1 macro/loss en ``results/``.

    No recibe argumentos ni devuelve nada directamente: toda la
    configuración se lee de ``sys.argv`` mediante ``argparse`` (ver
    ``python scripts/train_mlp_patches.py --help`` para la lista completa
    de opciones).
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True,
                    help="Carpeta raíz del dataset (ej: ./datasets/32x32)")
    ap.add_argument("--patch_size", type=int, default=32,
                    help="Tamaño de los parches NxN (ej: 32 para 32x32)")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=str, default="1024,512")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--results_dir", type=str, default="results/results_mlp",
                    help="Carpeta donde se guardan los resultados")
    ap.add_argument("--task", choices=["major", "species", "condition"], default="major",
                    help="'major' (por defecto) = comportamiento de siempre. 'species'/"
                         "'condition' encienden el aprendizaje jerárquico (especie o "
                         "condición de salud del coral). Ver patch_dataset.py.")
    ap.add_argument("--min_class_count", type=int, default=50,
                    help="Solo para --task species/condition: excluye clases con menos de "
                         "este total de parches. Ver DECISIONES.md.")
    args = ap.parse_args()

    device = get_device()
    print("Device:", device)

    # Carpeta de resultados
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

    patch_size = args.patch_size
    input_dim = 3 * patch_size * patch_size
    num_classes = len(train_set.classes)
    hidden = parse_hidden(args.hidden)

    print(f"Patch size: {patch_size}x{patch_size}")
    print(f"Input dim = {input_dim}, num_classes = {num_classes}")
    print("Clases:", train_set.classes)

    model = MLP(input_dim, num_classes, hidden, dropout=args.dropout).to(device)
    weights = compute_class_weights(train_set).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # Map de etiquetas
    with open(results_dir / "label_map.json", "w", encoding="utf-8") as f:
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
    pd.DataFrame(history).to_csv(results_dir / "metrics.csv", index=False)
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(model.state_dict(), results_dir / "best_mlp.pt")
        print(f"Guardado results/best_mlp.pt (val_acc={best_val:.4f})")

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

    with open(results_dir / "test_report.txt", "w", encoding="utf-8") as f:
        f.write(rep + "\n\nConfusion Matrix:\n" + np.array2string(cm))

    print("\n=== Test Report ===")
    print(rep)
    print("Confusion Matrix:\n", cm)

    # Desglose por eje secundario (subcategoría real en task="major";
    # condición/especie real en task="species"/"condition").
    if args.task == "major":
        sub_df = subcategory_breakdown(test_set, y_true, y_pred, train_set.classes)
    else:
        sub_df = hierarchical_breakdown(test_set, y_true, y_pred, train_set.classes)
    sub_df.to_csv(results_dir / "test_report_by_subcategory.csv", index=False)
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
        plt.title("Accuracy por época")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(results_dir / "curves_accuracy.png", dpi=150)

        # F1 macro
        plt.figure()
        plt.plot(epochs, [h["train_f1"] for h in history], label="train_f1")
        plt.plot(epochs, [h["val_f1"]   for h in history], label="val_f1")
        plt.plot(epochs, [h["test_f1"]  for h in history], label="test_f1")
        plt.xlabel("Epoch")
        plt.ylabel("F1 macro")
        plt.title("F1 macro por época (MLP)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(results_dir / "curves_f1.png", dpi=150)

        # Loss
        plt.figure()
        plt.plot(epochs, tr_loss, label="train_loss")
        plt.plot(epochs, va_loss, label="val_loss")
        plt.plot(epochs, te_loss, label="test_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss por época")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(results_dir / "curves_loss.png", dpi=150)

        # Matriz de confusión como heatmap
        plt.figure(figsize=(6, 5))
        im = plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.title("Matriz de confusión")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        tick_marks = np.arange(len(train_set.classes))
        plt.xticks(tick_marks, train_set.classes, rotation=45, ha="right")
        plt.yticks(tick_marks, train_set.classes)
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        plt.savefig(results_dir / "confusion_matrix.png", dpi=150)

        print("Guardadas en results/: curves_accuracy.png, curves_f1.png, curves_loss.png y confusion_matrix.png")
    except Exception as e:
        print(f"[AVISO] No se pudieron generar las gráficas: {e}")


if __name__ == "__main__":
    main()
