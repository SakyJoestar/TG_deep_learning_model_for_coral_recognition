#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_crop_and_label_from_excel.py

Para cada imagen en --images_dir busca su Excel/CSV homónimo en --excels_dir,
lee X/Y y una columna de etiqueta (--label_col), recorta parches (40x40 por defecto)
y los guarda en estructura ImageFolder (train/val/test/<clase>/...).

Requisitos:
  pip install pandas pillow openpyxl
"""

import os, argparse, random
from pathlib import Path
import pandas as pd
from PIL import Image

IMG_EXTS = (".jpg",".jpeg",".png",".tif",".tiff",".bmp")
XLS_EXTS = (".xlsx",".xls")
CSV_EXTS = (".csv",)

def find_col(cols, candidates):
    cl = {str(c).lower(): c for c in cols}
    for cand in candidates:
        cand = cand.lower()
        for c in cols:
            if cand in str(c).lower():
                return cl[str(c).lower()]
    return None

def normalize_numeric_series(s):
    s = s.astype(str).str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")

def read_coords_labels_from_excel(path, sheet_hint="", x_col_hint=None, y_col_hint=None, lbl_hint=None, debug=False):
    import pandas as pd
    from pathlib import Path

    xls = pd.ExcelFile(path)
    sheets = xls.sheet_names

    def sheet_has_xy(sh):
        tmp = pd.read_excel(path, sheet_name=sh)
        cols = [str(c).lower() for c in tmp.columns]
        has_x = any(c.strip().startswith("x") or "x (pixel" in c for c in cols)
        has_y = any(c.strip().startswith("y") or "y (pixel" in c for c in cols)
        return has_x and has_y, tmp

    # 1) Si pasas --sheet y existe, se usa (solo si tiene X/Y)
    df = None
    used_sheet = None
    if sheet_hint and sheet_hint in sheets:
        ok, tmp = sheet_has_xy(sheet_hint)
        if ok:
            df = tmp
            used_sheet = sheet_hint

    # 2) Intento auto: prioriza hojas que terminen en "_archive" con X/Y
    if df is None:
        for sh in sheets:
            if str(sh).lower().endswith("_archive"):
                ok, tmp = sheet_has_xy(sh)
                if ok:
                    df = tmp
                    used_sheet = sh
                    break

    # 3) Fallback auto: cualquier hoja con X/Y
    if df is None:
        for sh in sheets:
            ok, tmp = sheet_has_xy(sh)
            if ok:
                df = tmp
                used_sheet = sh
                break

    if df is None:
        raise ValueError(f"Excel '{Path(path).name}': no encontré ninguna hoja con columnas X/Y. Hojas: {sheets}")

    if debug:
        print(f"[DEBUG] Excel {Path(path).name}: hoja usada='{used_sheet}' columnas: {list(df.columns)}")

    x_col = x_col_hint or find_col(df.columns, ["x (pixel","x_pixel","xpixel","x [pix","x pix"," x","x)","x"])
    y_col = y_col_hint or find_col(df.columns, ["y (pixel","y_pixel","ypixel","y [pix","y pix"," y","y)","y"])
    if x_col is None and "X" in df.columns: x_col = "X"
    if y_col is None and "Y" in df.columns: y_col = "Y"
    if x_col is None or y_col is None:
        raise ValueError(f"Excel '{Path(path).name}': no encontré columnas X/Y en hoja '{used_sheet}'. Cols: {list(df.columns)}")

    # -------- columna de etiqueta --------
    if lbl_hint and lbl_hint in df.columns:
        lbl_col = lbl_hint
    else:
        lbl_col = (find_col(df.columns, ["code","major category","minor category","label","etiqueta"]) or "unknown")
        if lbl_col == "unknown":
            df["unknown"] = "unknown"

    out = df[[x_col, y_col, lbl_col]].copy()
    out.columns = ["x","y","label"]
    out["x"] = normalize_numeric_series(out["x"])
    out["y"] = normalize_numeric_series(out["y"])
    out = out.dropna(subset=["x","y"]).reset_index(drop=True)

    if debug:
        print(f"[DEBUG] filas válidas={len(out)} (sheet='{used_sheet}', x='{x_col}', y='{y_col}', label='{lbl_col}')")
    return out


def clamp(v, lo, hi): return max(lo, min(hi, v))

def crop_centered_patch(im, cx, cy, size=40, pad_edge=True):
    W,H = im.size; half = size//2
    x0,y0 = int(round(cx-half)), int(round(cy-half))
    x1,y1 = int(round(cx+half)), int(round(cy+half))
    ix0,iy0 = clamp(x0,0,W), clamp(y0,0,H)
    ix1,iy1 = clamp(x1,0,W), clamp(y1,0,H)
    patch = im.crop((ix0,iy0,ix1,iy1))
    if not pad_edge or patch.size==(size,size): return patch
    canvas = Image.new("RGB",(size,size),(0,0,0))
    canvas.paste(patch,(ix0-x0,iy0-y0))
    return canvas

def ensure_dir(p:Path): p.mkdir(parents=True, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--excels_dir", required=True)
    ap.add_argument("--out_dir", default="./dataset")
    ap.add_argument("--sheet", default="", help="Hoja del Excel (p.ej. DSCN9411_archive)")
    ap.add_argument("--x_col", default="", help="Nombre exacto columna X (opcional)")
    ap.add_argument("--y_col", default="", help="Nombre exacto columna Y (opcional)")
    ap.add_argument("--label_col", default="", help="Columna de etiqueta (Code/Major/Minor/Label)")
    ap.add_argument("--patch_size", type=int, default=40)
    ap.add_argument("--pad_edge", type=str, default="true")
    ap.add_argument("--scale_x", type=float, default=1.0)
    ap.add_argument("--scale_y", type=float, default=1.0)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--debug", type=str, default="false")
    args = ap.parse_args()

    pad_edge = args.pad_edge.lower() in ("true","1","yes","y","t")
    debug = args.debug.lower() in ("true","1","yes","y","t")
    random.seed(args.seed)

    images_dir = Path(args.images_dir)
    excels_dir = Path(args.excels_dir)

    # Carpeta raíz según tamaño de parche, p.ej. ./datasets/32x32
    root_out_dir = Path(args.out_dir) / f"{args.patch_size}x{args.patch_size}"
    out_dir = root_out_dir

    ensure_dir(root_out_dir)

    # recolecta pares imagen-excel homónimos
    images = sorted(
    [p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS],
    key=lambda p: p.name.lower()
    )
    pairs = []
    for img in images:
        stem = img.stem
        excel = None
        for ext in XLS_EXTS + CSV_EXTS:
            cand = excels_dir / f"{stem}{ext}" 
            if cand.exists():
                excel = cand; break
        if excel: pairs.append((img, excel))
        elif debug:
            print(f"[AVISO] No encontré Excel/CSV para {img.name}")

    # split por imagen (evita fuga)
    random.shuffle(pairs)
    n = len(pairs)
    n_test = int(round(n*args.test_ratio))
    n_val  = int(round(n*args.val_ratio))
    test_pairs  = pairs[:n_test]
    val_pairs   = pairs[n_test:n_test+n_val]
    train_pairs = pairs[n_test+n_val:]

    def split_of(img_path):
        if img_path in set(p[0] for p in test_pairs): return "test"
        if img_path in set(p[0] for p in val_pairs):  return "val"
        return "train"

    total = 0
    for img_path, excel_path in pairs:
        # --- CSV: columnas fijas Major Category, X, Y ---
        if excel_path.suffix.lower() in CSV_EXTS:
            df = pd.read_csv(excel_path)

            # nombres esperados
            xcol = args.x_col or "X"
            ycol = args.y_col or "Y"
            lcol = args.label_col or "Major Category"

            # por si acaso, si no están exactos, intenta encontrarlos por nombre (case-insensitive)
            if xcol not in df.columns:
                xcol = find_col(df.columns, ["x"])
            if ycol not in df.columns:
                ycol = find_col(df.columns, ["y"])
            if lcol not in df.columns:
                lcol = find_col(df.columns, ["major category"])

            if xcol is None or ycol is None or lcol is None \
               or xcol not in df.columns or ycol not in df.columns or lcol not in df.columns:
                print(f"[ERROR] {excel_path.name}: necesito columnas 'Major Category', 'X' y 'Y'")
                continue

            df = df[[xcol, ycol, lcol]].copy()
            df.columns = ["x", "y", "label"]
            df["x"] = normalize_numeric_series(df["x"])
            df["y"] = normalize_numeric_series(df["y"])
            df["label"] = df["label"].astype(str).str.strip()
            df = df.dropna(subset=["x", "y"]).reset_index(drop=True)

            if debug:
                print(f"[DEBUG] CSV {excel_path.name}: filas válidas={len(df)} (x='{xcol}', y='{ycol}', label='{lcol}')")

        # --- Excels normales ---
        else:
            df = read_coords_labels_from_excel(
                excel_path,
                sheet_hint=(args.sheet or ""),
                x_col_hint=(args.x_col or None),
                y_col_hint=(args.y_col or None),
                lbl_hint=(args.label_col or None),
                debug=debug,
            )

        if df.empty:
            if debug: print(f"[AVISO] {excel_path.name}: sin filas válidas")
            continue

        im = Image.open(img_path).convert("RGB")
        subset = split_of(img_path)

        for idx, r in df.iterrows():
            cx = float(r["x"])*args.scale_x
            cy = float(r["y"])*args.scale_y
            label = str(r.get("label","unknown")).strip() or "unknown"

            # carpeta destino
            dst = out_dir / subset / label
            ensure_dir(dst)

            patch = crop_centered_patch(im, cx, cy, size=args.patch_size, pad_edge=pad_edge)
            name = f"{img_path.stem}_p{idx+1:03d}_x{int(round(cx))}_y{int(round(cy))}.png"
            patch.save(dst / name)
            total += 1

    print(f"Listo. Parches totales: {total}. Estructura en: {out_dir}")
    print("Ej.: dataset/train/<clase>/*.png, dataset/val/<clase>/*.png, dataset/test/<clase>/*.png")

if __name__ == "__main__":
    main()