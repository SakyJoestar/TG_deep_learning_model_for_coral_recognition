#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""patch_dataset.py.

Utilidades para entrenar sobre los parches en distintos "niveles" de
etiqueta, sin necesidad de volver a correr el recorte. Los parches ya
están guardados como ``<split>/<major>[/<subcategoria>[/<condicion>]]``
(ver ``batch_crop_and_label_from_excel.py``); este módulo decide, para
cada parche, cuál nivel de esa ruta usar como clase según la tarea:

- ``"major"``: categoría mayor (p.ej. ``"CORAL"``). Es el nivel que ya
  usan los scripts de entrenamiento vía ``torchvision.datasets.ImageFolder``
  (``find_classes`` solo mira el primer nivel de subcarpetas); este
  módulo no cambia nada para esa tarea, es el "apagado" del modelo
  jerárquico.
- ``"species"``: categoría + subcategoría (p.ej.
  ``"CORAL/POCILLOPORA_SPP"``), ignorando el nivel de condición de salud
  si existe (un parche "sano" y uno "blanqueado" de la misma especie
  caen en la misma clase).
- ``"condition"``: solo parches de ``CORAL`` que sí tengan un tercer
  nivel (condición de salud: sano/blanqueado/recién muerto/
  fluorescencia), etiquetados por esa condición y agrupando todas las
  especies. Ver DECISIONES.md: agrupar por condición a través de
  especies es lo que hace entrenable ``CORAL_BLANQUEADO`` (25 en una
  sola especie -> ~1,500 agrupado).

En ``"species"`` y ``"condition"`` se excluyen las clases con menos de
``min_class_count`` parches en total (train+val+test combinados): con
muy pocos ejemplos ninguna técnica de balanceo las hace aprendibles, así
que en vez de intentar balancearlas se sacan del dataset (ver
DECISIONES.md, entrada de exclusión por umbral).
"""
import os
from collections import Counter
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMG_EXTS = (".jpg", ".jpeg", ".png")
STAGES = ("major", "species", "condition")


def _rel_parts(filepath: Path, root: Path):
    """Partes de carpeta de ``filepath`` relativas a ``root``, sin el nombre de archivo."""
    return Path(filepath).relative_to(root).parts[:-1]


def _label_and_extra(parts, stage):
    """Resuelve ``(etiqueta, etiqueta_secundaria)`` para un parche según ``stage``.

    La etiqueta secundaria no se usa para entrenar, solo para desglosar
    resultados después (ver :func:`hierarchical_breakdown`): en
    ``"species"`` es la condición (o ``"(sin_condicion)"``), en
    ``"condition"`` es la especie.

    Returns:
        tuple: ``(label, extra)``, o ``(None, None)`` si el parche no
        aplica a esta tarea (p.ej. un parche que no es de coral, en la
        tarea ``"condition"``).
    """
    if stage == "major":
        return parts[0], (parts[1] if len(parts) >= 2 else "(sin_subcategoria)")
    if stage == "species":
        label = parts[0] if len(parts) < 2 else f"{parts[0]}/{parts[1]}"
        extra = parts[2] if len(parts) >= 3 else "(sin_condicion)"
        return label, extra
    if stage == "condition":
        if parts[0] != "CORAL" or len(parts) < 3:
            return None, None
        return parts[2], parts[1]
    raise ValueError(f"stage desconocido: {stage!r} (usar uno de {STAGES})")


def _scan_split(split_root: Path, stage):
    """Recorre un split y devuelve ``[(path, label, extra), ...]`` para lo que aplica a ``stage``."""
    items = []
    for dirpath, _, filenames in os.walk(split_root):
        for fn in filenames:
            if not fn.lower().endswith(IMG_EXTS):
                continue
            fp = Path(dirpath) / fn
            parts = _rel_parts(fp, split_root)
            label, extra = _label_and_extra(parts, stage)
            if label is not None:
                items.append((fp, label, extra))
    return items


def scan_and_filter(data_dir, stage, min_class_count=0, debug=False):
    """Escanea train/val/test y arma los samples filtrados para ``stage``.

    Las clases (y sus conteos) se calculan sobre train+val+test
    combinados: si una clase queda por debajo de ``min_class_count`` en
    el total, se excluye de los tres splits a la vez (para no dejar una
    clase "huérfana" que en un split tiene ejemplos y en otro no).

    Args:
        data_dir: Carpeta raíz con subcarpetas ``train/``, ``val/``, ``test/``.
        stage: Uno de :data:`STAGES`.
        min_class_count: Mínimo de parches totales (los 3 splits sumados)
            para conservar una clase. ``0`` desactiva el filtro.
        debug: Si es ``True``, imprime las clases excluidas.

    Returns:
        tuple:

        - ``filtered`` (dict): ``{"train": [...], "val": [...], "test": [...]}``,
          listas de ``(path, label_idx, extra)``.
        - ``class_to_idx`` (dict): solo clases que sobrevivieron el filtro.
        - ``dropped`` (list): ``[(clase, conteo), ...]`` excluidas, ordenado
          por conteo ascendente.
    """
    data_dir = Path(data_dir)
    raw = {split: _scan_split(data_dir / split, stage) for split in ("train", "val", "test")}

    counts = Counter()
    for split in raw:
        counts.update(label for _, label, _ in raw[split])

    kept = sorted(c for c, n in counts.items() if n >= min_class_count)
    dropped = sorted(((c, n) for c, n in counts.items() if n < min_class_count), key=lambda x: x[1])
    class_to_idx = {c: i for i, c in enumerate(kept)}

    filtered = {
        split: [(p, class_to_idx[l], e) for p, l, e in raw[split] if l in class_to_idx]
        for split in raw
    }

    if debug or dropped:
        print(f"[patch_dataset] stage={stage}: {len(kept)} clases retenidas, "
              f"{len(dropped)} excluidas (<{min_class_count} parches en total)")
        for c, n in dropped:
            print(f"  excluida: {c} ({n} parches)")

    return filtered, class_to_idx, dropped


class PatchFolder(Dataset):
    """Dataset de parches a partir de una lista fija de ``(path, label_idx, extra)``.

    Equivalente a ``torchvision.datasets.ImageFolder`` pero con la lista
    de muestras ya resuelta externamente (por :func:`scan_and_filter`),
    para poder usar niveles de etiqueta (especie, condición) y exclusión
    por umbral que ``ImageFolder`` no soporta.
    """

    def __init__(self, samples, classes, class_to_idx, transform=None):
        self.samples = [(p, y) for p, y, _ in samples]
        self.extras = [e for _, _, e in samples]
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def make_hierarchical_loaders(data_dir, stage, train_tfm, eval_tfm, min_class_count=0,
                               batch_size=32, num_workers=2, debug=False):
    """Arma datasets + loaders de train/val/test para ``stage`` (``"species"`` o ``"condition"``).

    Args:
        data_dir: Carpeta raíz del dataset (con ``train/``, ``val/``, ``test/``).
        stage: ``"species"`` o ``"condition"`` (para ``"major"`` seguir
            usando ``ImageFolder`` normal, este módulo no aporta nada ahí).
        train_tfm, eval_tfm: Transforms de torchvision (mismos que ya usa
            cada script para su modelo).
        min_class_count: Mínimo de parches totales para conservar una clase.
        batch_size, num_workers: Igual que en ``make_loaders`` de cada script.
        debug: Si es ``True``, imprime detalle de clases excluidas.

    Returns:
        tuple: ``(train_set, val_set, test_set, train_loader, val_loader,
        test_loader, dropped)``, donde ``dropped`` es la lista de clases
        excluidas por el umbral (para loguear/guardar en resultados).
    """
    filtered, class_to_idx, dropped = scan_and_filter(data_dir, stage, min_class_count, debug)
    classes = sorted(class_to_idx, key=class_to_idx.get)

    train_set = PatchFolder(filtered["train"], classes, class_to_idx, transform=train_tfm)
    val_set = PatchFolder(filtered["val"], classes, class_to_idx, transform=eval_tfm)
    test_set = PatchFolder(filtered["test"], classes, class_to_idx, transform=eval_tfm)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_set, val_set, test_set, train_loader, val_loader, test_loader, dropped


def hierarchical_breakdown(dataset, y_true, y_pred, class_names):
    """Desglosa el acierto de test por la etiqueta secundaria (``dataset.extras``).

    Para ``stage="species"`` la etiqueta secundaria es la condición de
    salud (o ``"(sin_condicion)"``); para ``stage="condition"`` es la
    especie. Equivalente a ``subcategory_breakdown`` de los scripts de
    entrenamiento, pero genérico para estas dos tareas nuevas.

    Args:
        dataset: ``PatchFolder`` del split de test (con ``shuffle=False``
            en su loader, para que el orden coincida con ``y_true``/``y_pred``).
        y_true, y_pred: Arrays de :func:`predict_all`.
        class_names: ``dataset.classes`` (nombres en orden de índice).

    Returns:
        pandas.DataFrame: una fila por (clase, etiqueta secundaria), con
        columnas ``class``, ``secondary``, ``n``, ``correct``, ``accuracy``,
        ordenado por clase y luego por accuracy ascendente.
    """
    import pandas as pd

    rows = []
    for extra, yt, yp in zip(dataset.extras, y_true, y_pred):
        rows.append({
            "class": class_names[yt],
            "secondary": extra,
            "correct": int(yt == yp),
        })
    df = pd.DataFrame(rows)
    summary = df.groupby(["class", "secondary"]).agg(
        n=("correct", "size"), correct=("correct", "sum")
    ).reset_index()
    summary["accuracy"] = summary["correct"] / summary["n"]
    return summary.sort_values(["class", "accuracy"]).reset_index(drop=True)


def category_rollup(y_true, y_pred, class_names):
    """Agrega el acierto de test por categoría mayor, para ``stage="species"``.

    En ``"species"`` cada clase es ``"MAJOR"`` o ``"MAJOR/SUBCATEGORIA"``
    (ver :func:`_label_and_extra`). Esto agrupa las clases por su
    ``MAJOR`` y calcula, para cada una, el **promedio simple de la
    accuracy de sus subcategorías** (cada especie pesa igual sin importar
    cuántos parches tenga) junto con la accuracy ponderada real (todos
    los parches de esa categoría mayor juntos), para poder comparar
    ambas lecturas.

    Args:
        y_true, y_pred: Arrays de :func:`predict_all` sobre el split de test.
        class_names: ``dataset.classes`` (nombres en orden de índice,
            formato ``"species"``).

    Returns:
        pandas.DataFrame: una fila por categoría mayor, con columnas
        ``major``, ``n_subcategorias``, ``accuracy_promedio_subcategorias``
        (macro, sin ponderar), ``n_parches``, ``accuracy_ponderada``
        (micro, todos los parches juntos), ordenado por
        ``accuracy_promedio_subcategorias`` ascendente.
    """
    import pandas as pd

    rows = [{"class": class_names[yt], "correct": int(yt == yp)} for yt, yp in zip(y_true, y_pred)]
    df = pd.DataFrame(rows)
    df["major"] = df["class"].str.split("/").str[0]

    per_class = df.groupby(["major", "class"]).agg(
        n=("correct", "size"), correct=("correct", "sum")
    ).reset_index()
    per_class["accuracy"] = per_class["correct"] / per_class["n"]

    rollup = per_class.groupby("major").agg(
        n_subcategorias=("class", "nunique"),
        accuracy_promedio_subcategorias=("accuracy", "mean"),
        n_parches=("n", "sum"),
        correct_parches=("correct", "sum"),
    ).reset_index()
    rollup["accuracy_ponderada"] = rollup["correct_parches"] / rollup["n_parches"]
    rollup = rollup.drop(columns="correct_parches")

    return rollup.sort_values("accuracy_promedio_subcategorias").reset_index(drop=True)
